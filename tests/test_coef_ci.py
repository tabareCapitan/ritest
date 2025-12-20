import numpy as np
import pytest

from ritest.ci.coef_ci import Alt  # Literal["two-sided","left","right"]
from ritest.ci.coef_ci import coef_ci_band_fast, coef_ci_bounds_fast
from ritest.engine.fast_ols import FastOLS


# ---------------------------------------------------------------------
# Helper: simple linear fixture with known positive effect
# ---------------------------------------------------------------------
def make_linear_fixture(n=300, beta_true=1.5, gamma=0.6, alpha0=0.7, seed=42):
    """
    Return:
      (y, X, treat_idx, beta_obs, se_obs, beta_perm, K_obs, K_perm, beta_true)
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    T = rng.integers(0, 2, size=n).astype(float)  # Bernoulli(0.5)
    eps = rng.normal(scale=1.0, size=n)
    y = alpha0 + beta_true * T + gamma * x + eps

    X = np.column_stack([np.ones(n), x, T])
    treat_idx = 2

    # Observed fit (OLS)
    ols_obs = FastOLS(y, X, treat_idx)
    beta_obs = float(ols_obs.beta_hat)
    se_obs = float(ols_obs.se)
    K_obs = float(ols_obs.K)
    T_obs_metric = ols_obs.t_metric

    # Permutations
    R = 1200  # stable bands, still fast
    rng = np.random.default_rng(seed + 1)
    beta_perm = np.empty(R, dtype=float)
    K_perm = np.empty(R, dtype=float)
    for r in range(R):
        T_perm = rng.permutation(T)
        Xr = X.copy()
        Xr[:, treat_idx] = T_perm
        ols_r = FastOLS(y, Xr, treat_idx)
        beta_perm[r] = float(ols_r.beta_hat)
        # Cross-fit shift: K_r = c_rᵀ T_obs,metric
        K_perm[r] = float(ols_r.c_vector @ T_obs_metric)

    return (y, X, treat_idx, beta_obs, se_obs, beta_perm, K_obs, K_perm, beta_true)


# ---------------------------------------------------------------------
# 1) Regression: one-sided bands are NOT flat when K_perm varies
# ---------------------------------------------------------------------
def test_one_sided_band_not_flat():
    (_y, _X, _treat_idx, beta_obs, se_obs, beta_perm, K_obs, K_perm, _beta_true) = (
        make_linear_fixture()
    )

    grid, pvals = coef_ci_band_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        ci_range=3.0,
        ci_step=0.02,
        alternative="right",
    )
    assert grid.shape == pvals.shape
    assert not np.allclose(pvals, pvals[0])  # regression against scalar-K bug


# ---------------------------------------------------------------------
# 2) Alpha monotonicity + "makes sense" check
# ---------------------------------------------------------------------
def test_alpha_monotonicity_and_makes_sense():
    (_y, _X, _treat_idx, beta_obs, se_obs, beta_perm, K_obs, K_perm, _beta_true) = (
        make_linear_fixture()
    )

    ci_range = 3.0
    ci_step = 0.02
    alt: Alt = "two-sided"

    # Two-sided: increasing alpha SHRINKS the acceptance region
    lo05, hi05 = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=ci_range,
        ci_step=ci_step,
        alternative=alt,
    )
    lo10, hi10 = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.10,
        ci_range=ci_range,
        ci_step=ci_step,
        alternative=alt,
    )
    assert lo10 >= lo05 - 1e-12
    assert hi10 <= hi05 + 1e-12

    # Also: β̂ typically lies inside the two-sided CI at α=0.05
    if np.isfinite(lo05) and np.isfinite(hi05):
        assert lo05 <= beta_obs <= hi05

    # One-sided with positive effect:
    lo_r, hi_r = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=ci_range,
        ci_step=ci_step,
        alternative="right",
    )
    lo_l, hi_l = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=ci_range,
        ci_step=ci_step,
        alternative="left",
    )
    assert np.isfinite(lo_r) and np.isinf(hi_r)  # right-sided: finite, +inf
    assert np.isinf(lo_l) and np.isfinite(hi_l)  # left-sided: -inf, finite


# ---------------------------------------------------------------------
# 3) Grid resolution sensitivity
# ---------------------------------------------------------------------
def test_grid_resolution_sensitivity():
    (_y, _X, _treat_idx, beta_obs, se_obs, beta_perm, K_obs, K_perm, _beta_true) = (
        make_linear_fixture()
    )

    ci_range = 3.0
    alpha = 0.05
    step_coarse, step_fine = 0.05, 0.01
    alt: Alt = "two-sided"

    lo_c, hi_c = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=alpha,
        ci_range=ci_range,
        ci_step=step_coarse,
        alternative=alt,
    )
    lo_f, hi_f = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=alpha,
        ci_range=ci_range,
        ci_step=step_fine,
        alternative=alt,
    )
    tol = step_coarse * se_obs + 1e-12
    assert abs(lo_c - lo_f) <= tol
    assert abs(hi_c - hi_f) <= tol


# ---------------------------------------------------------------------
# 4) Identifiability edge: K_obs == 0 and K_perm == 0 ⇒ constant band
# ---------------------------------------------------------------------
def test_identifiability_constant_band():
    (_y, _X, _treat_idx, beta_obs, se_obs, beta_perm, _K_obs, _K_perm, _beta_true) = (
        make_linear_fixture()
    )

    K_obs = 0.0
    K_perm = np.zeros_like(beta_perm)
    alt: Alt = "two-sided"

    grid, pvals = coef_ci_band_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        ci_range=3.0,
        ci_step=0.02,
        alternative=alt,
    )
    assert np.allclose(pvals, pvals[0])

    lo, hi = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=3.0,
        ci_step=0.02,
        alternative=alt,
    )
    if pvals[0] >= 0.05:
        # Entire grid is inside: bounds span grid extremes
        assert lo == pytest.approx(grid[0]) and hi == pytest.approx(grid[-1])
    else:
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------
# 5) Empty acceptance region → (nan, nan)
# ---------------------------------------------------------------------
def test_empty_acceptance_region():
    (_y, _X, _treat_idx, _beta_obs, se_obs, beta_perm, _K_obs, _K_perm, _beta_true) = (
        make_linear_fixture()
    )

    # Force p=0 everywhere by killing K and moving beta_obs way out
    K_obs = 0.0
    K_perm = np.zeros_like(beta_perm)
    beta_obs_big = 1e6
    alt: Alt = "two-sided"

    lo, hi = coef_ci_bounds_fast(
        beta_obs=beta_obs_big,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=3.0,
        ci_step=0.1,
        alternative=alt,
    )
    assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------
# 6) Determinism
# ---------------------------------------------------------------------
def test_determinism_band_and_bounds():
    (_y, _X, _treat_idx, beta_obs, se_obs, beta_perm, K_obs, K_perm, _beta_true) = (
        make_linear_fixture()
    )

    alt: Alt = "two-sided"

    grid1, p1 = coef_ci_band_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        ci_range=3.0,
        ci_step=0.02,
        alternative=alt,
    )
    grid2, p2 = coef_ci_band_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        ci_range=3.0,
        ci_step=0.02,
        alternative=alt,
    )
    assert np.array_equal(grid1, grid2)
    assert np.array_equal(p1, p2)

    b1 = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=3.0,
        ci_step=0.02,
        alternative=alt,
    )
    b2 = coef_ci_bounds_fast(
        beta_obs=beta_obs,
        beta_perm=beta_perm,
        K_obs=K_obs,
        K_perm=K_perm,
        se=se_obs,
        alpha=0.05,
        ci_range=3.0,
        ci_step=0.02,
        alternative=alt,
    )
    assert b1 == b2


# ---------------------------------------------------------------------
# 7) Guards / invalid inputs
# ---------------------------------------------------------------------
def test_invalid_inputs_raise():
    beta_obs = 0.0
    se_obs = 1.0
    beta_perm = np.array([0.1, -0.2, 0.3])
    K_obs = 0.5
    K_perm_bad = np.array([0.2, 0.1])  # wrong shape
    alt: Alt = "two-sided"

    with pytest.raises(ValueError):
        coef_ci_band_fast(
            beta_obs=beta_obs,
            beta_perm=beta_perm,
            K_obs=K_obs,
            K_perm=K_perm_bad,
            se=se_obs,
            ci_range=3.0,
            ci_step=0.02,
            alternative=alt,
        )

    K_perm = np.array([0.2, 0.1, 0.0])
    with pytest.raises(ValueError):
        coef_ci_bounds_fast(
            beta_obs=beta_obs,
            beta_perm=beta_perm,
            K_obs=float("nan"),  # non-finite
            K_perm=K_perm,
            se=se_obs,
            alpha=0.05,
            ci_range=3.0,
            ci_step=0.02,
            alternative=alt,
        )

    with pytest.raises(ValueError):
        coef_ci_bounds_fast(
            beta_obs=beta_obs,
            beta_perm=beta_perm,
            K_obs=K_obs,
            K_perm=K_perm,
            se=-1.0,
            alpha=0.05,
            ci_range=3.0,
            ci_step=0.02,
            alternative=alt,
        )

    with pytest.raises(ValueError):
        coef_ci_bounds_fast(
            beta_obs=beta_obs,
            beta_perm=beta_perm,
            K_obs=K_obs,
            K_perm=K_perm,
            se=se_obs,
            alpha=1.5,  # bad alpha
            ci_range=3.0,
            ci_step=0.02,
            alternative=alt,
        )
