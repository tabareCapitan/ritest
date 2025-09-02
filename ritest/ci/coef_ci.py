"""
ritest.ci.coef_ci
=================

Coefficient CI utilities for permutation inference.

* Fast path:  β̂ = cᵀy models (OLS/WLS) — dot-product only
* Generic:    any model — rerun `ritest()` for each β₀
"""

from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np

Alt = Literal["two-sided", "left", "right"]


# ------------------------------------------------------------------ #
# Empirical p-value calculator (used by all CI methods)
# ------------------------------------------------------------------ #
def _empirical_p(z_obs: float, z_perm: np.ndarray, alternative: Alt) -> float:
    """Permutation p-value for a *scalar* statistic."""
    if alternative == "two-sided":
        return float((np.abs(z_perm) >= abs(z_obs)).mean())
    elif alternative == "right":
        return float((z_perm >= z_obs).mean())
    else:  # "left"
        return float((z_perm <= z_obs).mean())


# ------------------------------------------------------------------ #
# 1. FAST CI bounds (bisection with cᵀy trick)
# ------------------------------------------------------------------ #
def coef_ci_bounds_fast(
    beta_obs: float,
    z_obs: float,
    z_perm: np.ndarray,
    K: float,
    se: float,
    *,
    alpha: float,
    ci_range: float,
    tol: float,
    alternative: Alt = "two-sided",
) -> Tuple[float, float]:
    """
    Compute fast CI bounds for linear models using the cᵀy trick.

    Each p-value evaluation is one dot-product.

    Returns
    -------
    (lower, upper) : CI bounds (float or ±inf for one-sided)
    """

    def _root(side_sign: int) -> float:
        lo = beta_obs
        hi = beta_obs + side_sign * ci_range * se
        if lo > hi:
            lo, hi = hi, lo

        while (hi - lo) > tol * se:
            mid = 0.5 * (lo + hi)
            z_shifted = z_obs - K * mid
            z_perm_shifted = z_perm - K * mid
            p_mid = _empirical_p(z_shifted, z_perm_shifted, alternative)
            if p_mid < alpha:
                hi = mid if side_sign < 0 else hi
                lo = lo if side_sign < 0 else mid
            else:
                hi = hi if side_sign < 0 else mid
                lo = mid if side_sign < 0 else lo
        return lo if side_sign < 0 else hi

    if alternative == "two-sided":
        return (_root(-1), _root(+1))
    elif alternative == "right":
        return (_root(-1), float("inf"))
    else:  # "left"
        return (float("-inf"), _root(+1))


# ------------------------------------------------------------------ #
# 2. FAST CI band: full p(β₀) curve over a grid
# ------------------------------------------------------------------ #
def coef_ci_band_fast(
    beta_obs: float,
    z_obs: float,
    z_perm: np.ndarray,
    K: float,
    se: float,
    *,
    ci_range: float,
    ci_step: float,
    alternative: Alt = "two-sided",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (β₀, p-value) curve for linear models.

    Vectorised dot-product logic. Useful for diagnostic plots.

    Returns
    -------
    grid : np.ndarray
        β values to test.
    pvals : np.ndarray
        Corresponding permutation p-values.
    """
    grid = np.arange(-ci_range * se, ci_range * se + 1e-12, ci_step * se) + beta_obs

    if alternative == "two-sided":
        crit = np.abs(z_obs - K * grid)
        dist = np.abs(z_perm[:, None] - K * grid)
    elif alternative == "right":
        crit = z_obs - K * grid
        dist = z_perm[:, None] - K * grid
    else:  # "left"
        crit = -(z_obs - K * grid)
        dist = -(z_perm[:, None] - K * grid)

    pvals = (dist >= crit).mean(axis=0)
    return grid, pvals


# ------------------------------------------------------------------ #
# 3. GENERIC CI bounds (slow fallback for stat_fn models)
# ------------------------------------------------------------------ #
def coef_ci_bounds_generic(
    beta_obs: float,
    runner: Callable[[float], float],
    *,
    alpha: float,
    ci_range: float,
    ci_step: float,
    se: float,
    alternative: Alt = "two-sided",
) -> Tuple[float, float]:
    """
    Fallback CI via full re-evaluation of stat_fn at each β₀.

    Parameters
    ----------
    runner : callable
        Function `runner(beta0)` that returns permutation p-value.

    Returns
    -------
    (lower, upper) : CI bounds
    """
    grid = np.arange(-ci_range * se, ci_range * se + 1e-12, ci_step * se) + beta_obs
    pvals = np.array([runner(b0) for b0 in grid])

    inside = pvals >= alpha
    idx = np.where(inside)[0]

    if idx.size == 0:
        return (np.nan, np.nan)

    if alternative == "two-sided":
        return (float(grid[idx[0]]), float(grid[idx[-1]]))
    elif alternative == "right":
        return (float(grid[idx[0]]), float("inf"))
    else:  # "left"
        return (float("-inf"), float(grid[idx[-1]]))
