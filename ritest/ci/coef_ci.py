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
# 1) FAST CI band: full p(β₀) curve over a grid (vectorised)
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
    Compute (β₀, p-value) curve for linear models via vectorisation.

    For each candidate β₀ on a grid, we evaluate the permutation p-value using
    the identities:

        β_obs(β0)  = β_obs  − β0 · K_obs
        β_r  (β0)  = β_r    − β0 · K_r

    where K_obs = c_obsᵀ T_obs,metric and K_r = c_rᵀ T_obs,metric.

    Returns
    -------
    grid : np.ndarray
        β values to test (float64).
    pvals : np.ndarray
        Corresponding permutation p-values (float64).
    """
    # --- light sanity checks (public validation happens upstream) ----
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

    # Identifiability: if both K_obs and all K_perm are zero, p(β0) is constant.
    if K_obs == 0.0 and np.all(K_perm == 0.0):
        grid = (
            np.arange(-ci_range * se, ci_range * se + 1e-12, ci_step * se, dtype=np.float64)
            + beta_obs
        )
        # Constant p-values irrespective of β0 (informative for diagnostics).
        if alternative == "two-sided":
            p = (np.abs(beta_perm) >= abs(beta_obs)).mean()
        elif alternative == "right":
            p = (beta_perm >= beta_obs).mean()
        else:
            p = (beta_perm <= beta_obs).mean()
        return grid, np.full_like(grid, float(p), dtype=np.float64)

    # --- build grid and broadcasted arrays -------------------------------
    grid = (
        np.arange(-ci_range * se, ci_range * se + 1e-12, ci_step * se, dtype=np.float64) + beta_obs
    )

    # crit: shape (G,), dist: shape (R,G)
    crit = beta_obs - K_obs * grid  # β_obs(β0)
    dist = beta_perm[:, None] - K_perm[:, None] * grid[None, :]  # β_r(β0)

    # Tail-specific comparisons
    if alternative == "two-sided":
        comp = np.abs(dist) >= np.abs(crit)[None, :]
    elif alternative == "right":
        comp = dist >= crit[None, :]
    else:  # "left"
        comp = dist <= crit[None, :]

    pvals = comp.mean(axis=0).astype(np.float64, copy=False)
    return grid, pvals


# ------------------------------------------------------------------ #
# 2) FAST CI bounds (derived from the band; no brittle bisection)
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
    Compute fast CI bounds for linear models using the cᵀy trick.

    This computes the vectorised band once and returns bounds as the outermost
    grid points with p(β₀) ≥ α. One-sided intervals are open on the far side.

    Returns
    -------
    (lower, upper) : CI bounds (float or ±inf for one-sided)
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


# ------------------------------------------------------------------ #
# 3) GENERIC CI bounds (slow fallback for stat_fn models)
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
    if not (se > 0.0 and ci_range > 0.0 and ci_step > 0.0):
        raise ValueError("se, ci_range, and ci_step must be positive")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    grid = (
        np.arange(-ci_range * se, ci_range * se + 1e-12, ci_step * se, dtype=np.float64) + beta_obs
    )
    pvals = np.array([runner(b0) for b0 in grid], dtype=np.float64)

    inside = pvals >= float(alpha)
    idx = np.where(inside)[0]

    if idx.size == 0:
        return (float("nan"), float("nan"))

    if alternative == "two-sided":
        return (float(grid[idx[0]]), float(grid[idx[-1]]))
    elif alternative == "right":
        return (float(grid[idx[0]]), float("inf"))
    else:  # "left"
        return (float("-inf"), float(grid[idx[-1]]))
