"""
Coefficient CI utilities for permutation-based inference.

Supports the fast path for linear OLS/WLS models via precomputed
permutation quantities. Evaluations for candidate β₀ values reduce to
vectorised arithmetic to produce either the full (β₀, p-value) curve or
CI bounds.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

Alt = Literal["two-sided", "left", "right"]


# ------------------------------------------------------------------ #
# 1) FAST band: full p(β₀) curve (vectorised)
# ------------------------------------------------------------------ #
def coef_ci_band_fast(
    beta_obs: float,
    beta_perm: np.ndarray,
    K_obs: float,
    K_perm: np.ndarray,
    se: float,
    *,
    ci_range: float,
    ci_step: float,
    alternative: Alt = "two-sided",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a vectorised (β₀, p-value) curve for linear models.

    For each candidate β₀, the observed and permuted statistics are shifted via:

        β_obs(β₀) = β_obs − β₀ · K_obs
        β_r(β₀)   = β_r   − β₀ · K_r

    where K_obs = c_obsᵀ T_metric and K_r = c_rᵀ T_metric.

    Parameters
    ----------
    beta_obs : float
        Observed treatment coefficient.
    beta_perm : (R,) array
        Permuted coefficients.
    K_obs : float
        Observed cᵀ T_metric value.
    K_perm : (R,) array
        Permuted cᵀ T_metric values.
    se : float
        Standard error used to size the grid in SE units.
    ci_range : float
        Half-width of the grid, in SE units.
    ci_step : float
        Spacing of the grid, in SE units.
    alternative : {"two-sided", "left", "right"}
        Which tail to use for the p-value definition.

    Returns
    -------
    grid : ndarray
        Grid of β₀ values.
    pvals : ndarray
        Corresponding permutation p-values.
    """
    # Basic consistency checks (detailed validation happens upstream)
    beta_perm = np.asarray(beta_perm, dtype=np.float64)
    K_perm = np.asarray(K_perm, dtype=np.float64)
    if beta_perm.ndim != 1 or K_perm.ndim != 1 or beta_perm.shape != K_perm.shape:
        raise ValueError("beta_perm and K_perm must be 1-D arrays of the same shape")
    if not np.isfinite(beta_obs) or not np.isfinite(K_obs):
        raise ValueError("beta_obs and K_obs must be finite")
    if not (np.isfinite(beta_perm).all() and np.isfinite(K_perm).all()):
        raise ValueError("beta_perm and K_perm must be finite")
    if not (se > 0.0 and ci_range > 0.0 and ci_step > 0.0):
        raise ValueError("se, ci_range, and ci_step must be positive")

    # If both K_obs and all K_perm are zero, the statistic is invariant to β₀.
    if K_obs == 0.0 and np.all(K_perm == 0.0):
        grid = (
            np.arange(
                -ci_range * se, ci_range * se + 1e-12, ci_step * se, dtype=np.float64
            )
            + beta_obs
        )
        # Constant p-values in this degenerate case
        if alternative == "two-sided":
            p = (np.abs(beta_perm) >= abs(beta_obs)).mean()
        elif alternative == "right":
            p = (beta_perm >= beta_obs).mean()
        else:  # "left"
            p = (beta_perm <= beta_obs).mean()
        return grid, np.full_like(grid, float(p), dtype=np.float64)

    # Build grid of β₀ values
    grid = (
        np.arange(-ci_range * se, ci_range * se + 1e-12, ci_step * se, dtype=np.float64)
        + beta_obs
    )

    # crit: shape (G,), dist: shape (R, G)
    crit = beta_obs - K_obs * grid
    dist = beta_perm[:, None] - K_perm[:, None] * grid[None, :]

    # Tail comparisons
    if alternative == "two-sided":
        comp = np.abs(dist) >= np.abs(crit)[None, :]
    elif alternative == "right":
        comp = dist >= crit[None, :]
    else:  # "left"
        comp = dist <= crit[None, :]

    pvals = comp.mean(axis=0).astype(np.float64, copy=False)
    return grid, pvals


# ------------------------------------------------------------------ #
# 2) FAST bounds: derived from the band
# ------------------------------------------------------------------ #
def coef_ci_bounds_fast(
    beta_obs: float,
    beta_perm: np.ndarray,
    K_obs: float,
    K_perm: np.ndarray,
    se: float,
    *,
    alpha: float,
    ci_range: float,
    ci_step: float,
    alternative: Alt = "two-sided",
) -> Tuple[float, float]:
    """
    CI bounds for linear models via the vectorised fast band.

    The permutation p-value curve is computed once. The CI consists of the
    outermost β₀ values where p(β₀) ≥ α. For one-sided tests, the CI is open
    on the far side and uses ±∞ where appropriate.

    Parameters
    ----------
    beta_obs : float
        Observed treatment coefficient.
    beta_perm : (R,) array
        Permuted coefficients.
    K_obs : float
        Observed cᵀ T_metric value.
    K_perm : (R,) array
        Permuted cᵀ T_metric values.
    se : float
        Standard error used to size the grid in SE units.
    alpha : float
        Threshold for p(β₀); bounds are the extreme β₀ where p >= alpha.
    ci_range : float
        Half-width of the grid, in SE units.
    ci_step : float
        Spacing of the grid, in SE units.
    alternative : {"two-sided", "left", "right"}
        Which tail to use for the p-value definition.

    Returns
    -------
    (lo, hi) : tuple of float
        Finite bounds when available; ±∞ on the open side for one-sided tests;
        (nan, nan) if no grid points satisfy the threshold.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    grid, pvals = coef_ci_band_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se,
        ci_range=ci_range,
        ci_step=ci_step,
        alternative=alternative,
    )

    inside = pvals >= float(alpha)
    idx = np.nonzero(inside)[0]
    if idx.size == 0:
        return (float("nan"), float("nan"))

    if alternative == "two-sided":
        return (float(grid[idx[0]]), float(grid[idx[-1]]))
    elif alternative == "right":
        return (float(grid[idx[0]]), float("inf"))
    else:  # "left"
        return (float("-inf"), float(grid[idx[-1]]))
