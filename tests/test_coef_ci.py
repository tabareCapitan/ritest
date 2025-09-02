# tests/test_coef_ci.py
"""
Unit tests for ritest.ci.coef_ci

Run with:
    pytest -q -s tests/test_coef_ci.py
"""

from __future__ import annotations

import math

import numpy as np

from ritest.ci.coef_ci import (
    coef_ci_band_fast,
    coef_ci_bounds_fast,
    coef_ci_bounds_generic,
)

# ------------------------------------------------------------------ #
# Common fixtures / helpers
# ------------------------------------------------------------------ #
SEED = 42
rng = np.random.default_rng(SEED)

beta_obs = 0.5
z_obs = 1.0  # observed test statistic
z_perm = rng.normal(0, 1, size=10_000)  # permuted stats
K, se = 1.0, 0.1


# ------------------------------------------------------------------ #
# 1. coef_ci_bounds_fast – two-sided sanity
# ------------------------------------------------------------------ #
def test_bounds_fast_contains_beta():
    print("\n[Section 1] bounds_fast two-sided sanity")
    lo, hi = coef_ci_bounds_fast(beta_obs, z_obs, z_perm, K, se, alpha=0.05, ci_range=5.0, tol=1e-4)
    print(f"  CI = ({lo:.4f}, {hi:.4f})  β̂ = {beta_obs}")
    assert lo <= beta_obs <= hi
    assert hi > lo


# ------------------------------------------------------------------ #
# 2. bounds widen with α
# ------------------------------------------------------------------ #
def test_bounds_fast_alpha_effect():
    print("\n[Section 2] bounds widen when α increases")
    lo1, hi1 = coef_ci_bounds_fast(
        beta_obs, z_obs, z_perm, K, se, alpha=0.05, ci_range=5.0, tol=1e-4
    )
    lo2, hi2 = coef_ci_bounds_fast(
        beta_obs, z_obs, z_perm, K, se, alpha=0.10, ci_range=5.0, tol=1e-4
    )
    print(f"  α=0.05 → ({lo1:.4f}, {hi1:.4f})")
    print(f"  α=0.10 → ({lo2:.4f}, {hi2:.4f})")
    assert lo2 <= lo1 and hi2 >= hi1


# ------------------------------------------------------------------ #
# 3. One-sided returns ±∞
# ------------------------------------------------------------------ #
def test_bounds_fast_one_sided():
    print("\n[Section 3] one-sided ±∞ logic")
    lo_r, hi_r = coef_ci_bounds_fast(
        beta_obs, z_obs, z_perm, K, se, alpha=0.05, ci_range=5.0, tol=1e-4, alternative="right"
    )
    lo_l, hi_l = coef_ci_bounds_fast(
        beta_obs, z_obs, z_perm, K, se, alpha=0.05, ci_range=5.0, tol=1e-4, alternative="left"
    )
    assert math.isfinite(lo_r) and hi_r == float("inf")
    assert lo_l == float("-inf") and math.isfinite(hi_l)


# ------------------------------------------------------------------ #
# 4. coef_ci_band_fast – structural checks
# ------------------------------------------------------------------ #
def test_band_fast_structure():
    print("\n[Section 4] band_fast structural checks")
    grid, pvals = coef_ci_band_fast(
        beta_obs,
        z_obs,
        z_perm,
        K,
        se,
        ci_range=3.0,
        ci_step=0.2,
        alternative="right",
    )
    print(f"  grid len = {len(grid)}, pvals len = {len(pvals)}")
    assert len(grid) == len(pvals)
    assert np.all(np.isfinite(grid))
    assert np.all(np.isfinite(pvals))


# ------------------------------------------------------------------ #
# 5. coef_ci_bounds_generic – synthetic runner
# ------------------------------------------------------------------ #
def test_bounds_generic_dummy():
    print("\n[Section 5] bounds_generic synthetic runner")
    true_beta, width = 1.0, 0.05

    def runner(b0: float) -> float:
        return 0.2 if abs(b0 - true_beta) < width else 0.01

    lo, hi = coef_ci_bounds_generic(
        beta_obs=true_beta, runner=runner, alpha=0.05, ci_range=0.5, ci_step=0.005, se=0.1
    )
    assert abs(lo - (true_beta - width)) < 0.01
    assert abs(hi - (true_beta + width)) < 0.01


# ------------------------------------------------------------------ #
# 6. bounds_generic returns NaNs when no coverage
# ------------------------------------------------------------------ #
def test_bounds_generic_no_coverage():
    print("\n[Section 6] bounds_generic returns NaNs if p < α everywhere")
    lo, hi = coef_ci_bounds_generic(
        beta_obs=0.0,
        runner=lambda _b: 0.004,  # always tiny p
        alpha=0.05,
        ci_range=1.0,
        ci_step=0.01,
        se=0.1,
    )
    assert math.isnan(lo) and math.isnan(hi)
