"""
Fast OLS/WLS estimator with optional robust covariance.

Implements a minimal OLS/WLS core aimed at permutation-intensive workflows.
The estimator caches (XᵀX)⁻¹ once and derives a c-vector so that permutation
statistics reduce to dot-products. Robust covariance (HC1 or CRV1) is computed
at most once per fit. Permutation evaluation uses a pre-weighted c-vector
and avoids rebuilding design matrices.

Public interface
----------------
FastOLS(y, X, treat_idx, *, weights=None, cluster=None, compute_vcov=True)

Attributes
----------
beta_hat : float
    Treatment coefficient.
se : float
    Robust standard error (NaN if compute_vcov=False).
vcov : array or None
    Robust covariance matrix.
c_vector : ndarray
    Vector satisfying β̂ = cᵀ y   (or cᵀ yw for weighted LS).
c_perm_vector : ndarray
    c-vector in the correct metric for permutation draws.
K : float
    cᵀ T_ where T_ is the treatment column in the metric of X (weighted if WLS).
t_metric : ndarray
    Treatment regressor in the same metric as c_vector.

Notes
-----
- No missing-value handling; upstream validation is required.
- For permutation tests, use ``permuted_stats(Y_perm)`` on an instance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ------------------------------------------------------------------
# Robust covariance: HC1   (pure NumPy)
# ------------------------------------------------------------------
def _vcov_white(
    res: np.ndarray,
    X: np.ndarray,
    xtx_inv: np.ndarray | None = None,
) -> np.ndarray:
    """
    HC1 (Eicker-Huber-White) sandwich:

        V = (n/(n-k)) · (XᵀX)⁻¹ (Xᵀ diag(e²) X) (XᵀX)⁻¹

    Parameters
    ----------
    res : (n,) float64
        Residuals (in the same metric as X; i.e., weighted if WLS).
    X : (n,k) float64
        Design matrix (weighted if WLS).
    xtx_inv : (k,k) float64, optional
        Precomputed inverse of (XᵀX). If not provided, it is computed here
        via Cholesky-based solves. Supplying it avoids a redundant factorization.

    Returns
    -------
    vcov : (k,k) float64
        HC1 robust covariance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    res = np.asarray(res, dtype=np.float64)
    n, k = X.shape
    if xtx_inv is None:
        try:
            xtx = X.T @ X
            L = np.linalg.cholesky(xtx)
            xtx_inv = np.linalg.solve(
                L.T,
                np.linalg.solve(L, np.eye(k, dtype=np.float64)),
            )
        except np.linalg.LinAlgError as err:  # pragma: no cover
            raise ValueError(
                "Cholesky factorization failed in _vcov_white: (X'X) is not SPD. "
                "This typically indicates collinearity or an ill-conditioned design."
            ) from err
    Xe = X * res.reshape(-1, 1)
    S = Xe.T @ Xe
    return xtx_inv @ S @ xtx_inv * (n / (n - k))


# ------------------------------------------------------------------
# Robust covariance: CRV1  (vectorised NumPy)
# ------------------------------------------------------------------
def _vcov_cluster(
    res: np.ndarray,
    X: np.ndarray,
    cluster: np.ndarray,
    xtx_inv: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    CRV1 (small-sample-corrected cluster-robust variance):

    .. math::

        V
        = \frac{G}{G-1}\frac{n-1}{n-k}
          (X^\top X)^{-1}
          \Bigl(\sum_{g=1}^{G} (X^\top e)_g (X^\top e)_g^\top\Bigr)
          (X^\top X)^{-1},

    where :math:`(X^\top e)_g = \sum_{i\in g} X_i e_i`.

    Parameters
    ----------
    res : (n,) float64
        Residuals (in the same metric as X; i.e., weighted if WLS).
    X : (n,k) float64
        Design matrix (weighted if WLS).
    cluster : (n,) int
        Cluster labels (same length as y/X).
    xtx_inv : (k,k) float64, optional
        Precomputed inverse of (XᵀX). If not provided, it is computed here
        via Cholesky-based solves. Supplying it avoids a redundant factorization.

    Returns
    -------
    vcov : (k,k) float64
        CRV1 robust covariance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    res = np.asarray(res, dtype=np.float64)
    cluster = np.asarray(cluster)
    n, k = X.shape
    clusters, pos = np.unique(cluster, return_inverse=True)
    G = clusters.size

    # Accumulate cluster scores   S_g = Σ_{i∈g} X_i e_i
    scores = np.zeros((G, k), dtype=np.float64)
    Xe = X * res.reshape(-1, 1)
    np.add.at(scores, pos, Xe)

    middle = scores.T @ scores  # Σ_g  S_g S_gᵀ

    if xtx_inv is None:
        try:
            xtx = X.T @ X
            L = np.linalg.cholesky(xtx)
            xtx_inv = np.linalg.solve(
                L.T,
                np.linalg.solve(L, np.eye(k, dtype=np.float64)),
            )
        except np.linalg.LinAlgError as err:  # pragma: no cover
            raise ValueError(
                "Cholesky factorization failed in _vcov_cluster: (X'X) is not SPD. "
                "This typically indicates collinearity or an ill-conditioned design."
            ) from err
    scale = (G / (G - 1)) * ((n - 1) / (n - k))
    return xtx_inv @ middle @ xtx_inv * scale


# ------------------------------------------------------------------
# Permutation helper – optionally JIT-accelerated
# ------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)  # pragma: no cover
    def fast_permuted_stats(c_vec: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute z_r = cᵀ y_perm[r] for each of R permutations.

        Advanced: normally you should use `FastOLS.permuted_stats(Y_perm)`,
        which selects the correct `c` (pre-weighted if WLS). This low-level
        kernel assumes `c_vec` already matches the metric of `Y`.
        """
        if False:  # Numba keeps a branch but never executes it
            print("RUNNING UNJITTED FUNCTION")

        R = Y.shape[0]
        out = np.empty(R, dtype=np.float64)
        for r in prange(R):
            out[r] = c_vec @ Y[r]
        return out

    NUMBA_OK = True

except ImportError:  # pragma: no cover

    def fast_permuted_stats(c_vec: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorised fallback (still fast).

        Advanced: normally you should use `FastOLS.permuted_stats(Y_perm)`,
        which selects the correct `c` (pre-weighted if WLS). This function
        assumes `c_vec` already matches the metric of `Y`.
        """
        return np.asarray(Y, dtype=np.float64) @ np.asarray(c_vec, dtype=np.float64)

    NUMBA_OK = False


# ------------------------------------------------------------------
# Main estimator
# ------------------------------------------------------------------
class FastOLS:
    """
    High-performance OLS / WLS with robust SEs.

    Parameters
    ----------
    y : (n,) arraylike
        Outcome vector.
    X : (n,k) arraylike
        Design matrix. (Include intercept yourself if needed.)
    treat_idx : int
        Column index of the treatment variable inside ``X``.
    weights : (n,) arraylike, optional
        Analytic weights (WLS). If ``None`` ordinary LS is used.
    cluster : arraylike, optional
        Cluster identifiers for CRV1. If ``None`` ⇒ HC1.
    compute_vcov : bool, default True
        If False, skip residual and robust covariance calculations.
        Useful for permutation fits, where only β̂, c-vectors, K and t_metric
        are needed.

    Notes
    -----
    - No missing-value handling is done here; validation lives upstream.
    - When ``compute_vcov=False``, attributes ``vcov=None`` and ``se=np.nan``.
    - For permutation tests, call ``permuted_stats(Y_perm)`` on the instance.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        treat_idx: int,
        *,
        weights: Optional[np.ndarray] = None,
        cluster: Optional[np.ndarray] = None,
        compute_vcov: bool = True,
    ):
        # ----- Basic integrity checks ------------------------------------
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n, k = X.shape
        if y.shape[0] != n:
            raise ValueError("y and X must have the same number of rows")
        if not (0 <= treat_idx < k):
            raise IndexError("treat_idx out of bounds")
        if n <= k:
            raise ValueError("X'X is not invertible: need n > k")

        # ----- Optional analytic weights ---------------------------------
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if (not np.all(np.isfinite(w))) or (np.any(w <= 0.0)):
                raise ValueError("weights must be strictly positive finite numbers")
            w_sqrt = np.sqrt(w, dtype=np.float64)
            Xw = X * w_sqrt[:, None]
            yw = y * w_sqrt
        else:
            w_sqrt = None
            Xw = X
            yw = y

        # ----- Optional cluster checks -----------------------------------
        if cluster is not None:
            cluster = np.asarray(cluster)
            if cluster.shape[0] != n:
                raise ValueError("cluster must have the same length as y/X")
            # G checked below only if compute_vcov=True

        # ----- Cache (XᵀX)⁻¹ via Cholesky solves -------------------------
        try:
            xtx = Xw.T @ Xw
            L = np.linalg.cholesky(xtx)
            xtx_inv = np.linalg.solve(
                L.T,
                np.linalg.solve(L, np.eye(k, dtype=np.float64)),
            )
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "Cholesky factorization failed: (X'X) is not SPD. "
                "This typically indicates collinearity, duplicated columns, "
                "or an ill-conditioned design."
            ) from err
        self._XtX_inv = xtx_inv

        # ----- c-vector & point estimate ---------------------------------
        # β̂ = cᵀ y    (OLS)   or   β̂ = cᵀ yw (WLS)
        at = xtx_inv[treat_idx, :]  # 1×k row
        c_vec = at @ Xw.T  # (n,)
        self.c_vector = c_vec
        self.beta_hat = float(c_vec @ yw)

        # Precompute the permutation vector: c_perm = c (OLS) or c ∘ √w (WLS)
        if w_sqrt is None:
            self.c_perm_vector = c_vec
        else:
            self.c_perm_vector = c_vec * w_sqrt

        # ----- Constant  K = cᵀ T_  (perm-CI shift) ----------------------
        # T_ must live in the same metric as c:
        #   OLS: T_ = X[:, treat_idx]
        #   WLS: T_ = X[:, treat_idx] ∘ √w
        T_metric = Xw[:, treat_idx]
        self.K = float(c_vec @ T_metric)
        # Expose the treatment column in the same metric as `c_vector`.
        # This lets the orchestrator compute K_r = c_rᵀ T_obs,metric for each permutation.
        self.t_metric = T_metric

        # ----- Residuals & robust covariance (optional) ------------------
        if compute_vcov:
            # Solve (XᵀX) β = Xᵀ y  via the same Cholesky factors
            beta_full = np.linalg.solve(L.T, np.linalg.solve(L, Xw.T @ yw))
            resid = yw - Xw @ beta_full

            if cluster is None:
                vcov = _vcov_white(resid, Xw, xtx_inv)
            else:
                G = int(np.unique(cluster).size)
                if G < 2:
                    raise ValueError("CRV1 requires at least 2 clusters")
                vcov = _vcov_cluster(resid, Xw, cluster, xtx_inv)

            self.vcov = vcov
            self.se = float(np.sqrt(vcov[treat_idx, treat_idx]))
        else:
            self.vcov = None  # type: ignore[assignment]
            self.se = float("nan")

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def permuted_stats(self, Y: np.ndarray) -> np.ndarray:
        """
        Return z_r = cᵀ y_perm[r] for each row of Y.

        Uses the precomputed ``c_perm_vector`` so callers never need to think
        about weights; this is the canonical way to compute permutation stats.

        Accepts either a 2-D array of shape (R, n) or a 1-D vector (n,),
        which will be treated as a single permutation. Dtypes are coerced
        to float64 without copying when possible.
        """
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            if Y.size != self.c_perm_vector.size:
                raise ValueError(
                    f"Y has length {Y.size}, expected {self.c_perm_vector.size}"
                )
            Y = Y.reshape(1, -1)
        elif Y.ndim == 2:
            if Y.shape[1] != self.c_perm_vector.size:
                raise ValueError(
                    f"Y has shape (R, {Y.shape[1]}), expected second dim "
                    f"{self.c_perm_vector.size}"
                )
        else:  # pragma: no cover
            raise ValueError("Y must be 1-D or 2-D")

        return fast_permuted_stats(self.c_perm_vector, Y)

    def coef(self) -> float:  # noqa: D401
        """Return the treatment coefficient β̂."""
        return self.beta_hat

    def se_robust(self) -> float:
        """Return the robust standard error (√diag of sandwich)."""
        return self.se


__all__ = ["FastOLS", "fast_permuted_stats", "NUMBA_OK"]
