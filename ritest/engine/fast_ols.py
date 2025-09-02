"""
ritest.engine.fast_ols
======================

A minimal yet *fast* OLS / WLS estimator implemented in pure NumPy.
Robust variance–covariance matrices are computed once; permutation
loops then reuse a cached *c-vector* so that each permuted statistic
is just a dot-product.

Why this exists
---------------
`statsmodels` is feature-rich but far too slow to refit thousands of
times during randomisation inference.  FastOLS:

1.  Caches  ``(XᵀX)⁻¹``  once.
2.  Derives  ``c_vector``  so that  **β̂ = cᵀ y**.
3.  Computes a *robust* sandwich (HC1 or CRV1) exactly once.
4.  Optionally jit-accelerates the permutation dot-products.

Robust-Inference Policy
-----------------------
*If* ``cluster`` is supplied ⟶ CRV1 (Stata default).
Otherwise ⟶ HC1 (White).  We never assume homoskedasticity.

Public Interface
----------------
FastOLS(y, X, treat_idx, *, weights=None, cluster=None)

Attributes
~~~~~~~~~~
beta_hat : float             point estimate of treatment effect
se       : float             robust SE = √vcov[treat_idx, treat_idx]
vcov     : (k+1,k+1) array   full robust sandwich
c_vector : (n,) array        β = cᵀ y  (re-used for permutations)
K        : float             cᵀ T (constant across permutations)

Helper
~~~~~~
fast_permuted_stats(c_vector, Y_perm) → β for each permuted y
(JIT-parallel if Numba available)

Example
~~~~~~~
>>> ols = FastOLS(y, Z, treat_idx=2, cluster=school_id)
>>> beta_obs = ols.beta_hat
>>> z_perm   = fast_permuted_stats(ols.c_vector, Y_perm)   # RI loop
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ------------------------------------------------------------------
# Robust covariance: HC1   (pure NumPy, no Numba needed)
# ------------------------------------------------------------------
def _vcov_white(res: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    HC1 (Eicker-Huber-White) sandwich:

        V = (n/(n-k)) · (XᵀX)⁻¹ (Xᵀ diag(e²) X) (XᵀX)⁻¹
    """
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    S = (X * res.reshape(-1, 1)).T @ (X * res.reshape(-1, 1))
    return xtx_inv @ S @ xtx_inv * n / (n - k)


# ------------------------------------------------------------------
# Robust covariance: CRV1  (vectorised NumPy, no Numba)
# ------------------------------------------------------------------
def _vcov_cluster(res: np.ndarray, X: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    r"""
    CRV1 (small-sample-corrected cluster-robust variance):

    .. math::

        V
        = \frac{G}{G-1}\frac{n-1}{n-k}
          (X^\top X)^{-1}
          \Bigl(\sum_{g=1}^{G} (X^\top e)_g (X^\top e)_g^\top\Bigr)
          (X^\top X)^{-1},

    where :math:`(X^\top e)_g = \sum_{i\in g} X_i e_i`.
    """
    n, k = X.shape
    clusters, pos = np.unique(cluster, return_inverse=True)
    G = clusters.size

    # Accumulate cluster scores   S_g = Σ_{i∈g} X_i e_i
    scores = np.zeros((G, k))
    np.add.at(scores, pos, X * res[:, None])

    middle = scores.T @ scores  # Σ_g  S_g S_gᵀ

    xtx_inv = np.linalg.inv(X.T @ X)
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
        Compute  z_r = cᵀ y_perm[r]  for each of R permutations in parallel.
        """
        if False:  # <-- will be stripped by Numba in compiled mode
            print("RUNNING UNJITTED FUNCTION")  # never prints if JIT compiled

        R = Y.shape[0]
        out = np.empty(R)
        for r in prange(R):
            out[r] = c_vec @ Y[r]
        return out

    NUMBA_OK = True

except ImportError:  # pragma: no cover

    def fast_permuted_stats(c_vec: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Vectorised fallback (still fast)."""
        return Y @ c_vec

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
        Design matrix.  (Include intercept yourself if needed.)
    treat_idx : int
        Column index of the treatment variable inside ``X``.
    weights : (n,) arraylike, optional
        Analytic weights (WLS).  If *None* ordinary LS is used.
    cluster : arraylike, optional
        Cluster identifiers for CRV1.  If *None* ⇒ HC1.

    Notes
    -----
    * No missing-value handling is done here – validation lives upstream.
    * Robust covariance is computed **once**; permutation tests then use
      only ``c_vector`` which makes each draw *O(n)* not *O(n³)*.
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
    ):
        # ----- Basic integrity checks ------------------------------------
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        n, k = X.shape
        if y.shape[0] != n:
            raise ValueError("y and X must have the same number of rows")
        if not 0 <= treat_idx < k:
            raise IndexError("treat_idx out of bounds")

        # ----- Optional analytic weights ---------------------------------
        if weights is not None:
            w = np.sqrt(weights).reshape(-1, 1)
            Xw = X * w
            yw = y * w.ravel()
        else:
            Xw = X
            yw = y

        # ----- Cache (XᵀX)⁻¹ --------------------------------------------
        xtx_inv = np.linalg.inv(Xw.T @ Xw)
        self._XtX_inv = xtx_inv

        # ----- c-vector & point estimate ---------------------------------
        self.c_vector = c_vec = xtx_inv[treat_idx] @ Xw.T
        self.beta_hat = float(c_vec @ yw)

        # ----- Residuals -------------------------------------------------
        beta_full = xtx_inv @ (Xw.T @ yw)
        resid = yw - Xw @ beta_full

        # ----- Robust sandwich covariance --------------------------------
        if cluster is None:
            vcov = _vcov_white(resid, Xw)
        else:
            vcov = _vcov_cluster(resid, Xw, np.asarray(cluster))

        self.vcov = vcov
        self.se = float(np.sqrt(vcov[treat_idx, treat_idx]))

        # ----- Constant  K = cᵀ T  (perm-CI shift) -----------------------
        self.K = float(c_vec @ X[:, treat_idx])

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------
    def coef(self) -> float:  # noqa: D401
        """Return the treatment coefficient β̂."""
        return self.beta_hat

    def se_robust(self) -> float:
        """Return the robust standard error (√diag of sandwich)."""
        return self.se


__all__ = ["FastOLS", "fast_permuted_stats", "NUMBA_OK"]
