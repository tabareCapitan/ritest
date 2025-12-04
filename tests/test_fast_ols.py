# tests/test_fast_ols.py
#
# Verbose, end-to-end tests for ritest.engine.fast_ols
# ----------------------------------------------------
# What this suite checks:
#  - OLS/HC1 vs statsmodels (beta & SE)
#  - WLS/HC1 vs statsmodels (beta & SE)
#  - CRV1 vs statsmodels (beta, SE, FULL VCOV)
#  - Weighted + Clustered CRV1 vs statsmodels
#  - Correct K (cᵀ T_) under OLS and WLS
#  - c_perm_vector behavior (OLS vs WLS)
#  - permuted_stats wrapper correctness (2-D & 1-D inputs)
#  - Shape/dtype guards in permuted_stats
#  - Engine guards and friendly errors:
#       * n <= k
#       * treat_idx out of bounds
#       * cluster length mismatch
#       * clusters < 2 for CRV1
#       * negative or zero weights rejected
#       * Cholesky error (rank deficiency) surfaces with clear message
#  - NEW (minimal): t_metric metric check (OLS/WLS) and K == cᵀ t_metric

import time

import numpy as np
import pytest
import statsmodels.api as sm

from ritest.engine.fast_ols import NUMBA_OK, FastOLS, fast_permuted_stats

print(f"[fast_ols] Permutation backend: {'Numba (parallel)' if NUMBA_OK else 'NumPy fallback'}")

# Global tolerance for numeric parity vs statsmodels
RTOL = 1e-3


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def design(n=4000, seed=123, binary_t=False):
    """Build a simple design with intercept, covariate x, and treatment T."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if binary_t:
        T = rng.binomial(1, 0.5, size=n).astype(float)
    else:
        T = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    Z = np.column_stack([X, T])  # intercept, x, T
    beta_true = 1.75
    y = X @ np.array([0.5, -0.25]) + beta_true * T + rng.normal(scale=1.0, size=n)
    return y, Z, 2  # treat_idx=2


def clusters_for(n, g=20):
    """Even-ish cluster labels."""
    base = np.repeat(np.arange(g), n // g)
    pad = np.arange(n - base.size) if base.size < n else np.array([], dtype=int)
    return np.concatenate([base, pad])


# ---------------------------------------------------------------------
# Parity with statsmodels (HC1/CRV1)
# ---------------------------------------------------------------------
def test_ols_hc1_matches_statsmodels_verbose():
    print("\n=== [OLS / White HC1] Parity vs statsmodels ===")
    y, Z, t_idx = design(n=5000, seed=1, binary_t=True)

    t1 = time.time()
    sm_fit = sm.OLS(y, Z).fit(cov_type="HC1")
    t2 = time.time()
    fo = FastOLS(y, Z, treat_idx=t_idx)
    t3 = time.time()

    print(
        f"statsmodels: beta={sm_fit.params[t_idx]:.6f}, SE={sm_fit.bse[t_idx]:.6f}, time={t2-t1:.4f}s"
    )
    print(f"FastOLS:     beta={fo.coef():.6f},     SE={fo.se_robust():.6f},  time={t3-t2:.4f}s")

    assert np.allclose(sm_fit.params[t_idx], fo.coef(), rtol=RTOL)
    assert np.allclose(sm_fit.bse[t_idx], fo.se_robust(), rtol=RTOL)


def test_wls_hc1_matches_statsmodels_verbose():
    print("\n=== [WLS / Weighted HC1] Parity vs statsmodels ===")
    y, Z, t_idx = design(n=5000, seed=2, binary_t=False)
    rng = np.random.default_rng(2)
    w = rng.uniform(0.4, 2.2, size=y.size)

    # statsmodels WLS via manual sqrt(w) transform (matches our algebra)
    yw = y * np.sqrt(w)
    Zw = Z * np.sqrt(w)[:, None]

    t1 = time.time()
    sm_fit = sm.OLS(yw, Zw).fit(cov_type="HC1")
    t2 = time.time()
    fo = FastOLS(y, Z, treat_idx=t_idx, weights=w)
    t3 = time.time()

    print(
        f"statsmodels: beta={sm_fit.params[t_idx]:.6f}, SE={sm_fit.bse[t_idx]:.6f}, time={t2-t1:.4f}s"
    )
    print(f"FastOLS:     beta={fo.coef():.6f},     SE={fo.se_robust():.6f},  time={t3-t2:.4f}s")

    assert np.allclose(sm_fit.params[t_idx], fo.coef(), rtol=RTOL)
    assert np.allclose(sm_fit.bse[t_idx], fo.se_robust(), rtol=RTOL)


def test_crv1_matches_statsmodels_full_vcov_verbose():
    print("\n=== [CRV1 / Clustered] Parity vs statsmodels (beta, SE, FULL VCOV) ===")
    y, Z, t_idx = design(n=6000, seed=3, binary_t=True)
    clusters = clusters_for(n=y.size, g=30)

    t1 = time.time()
    sm_fit = sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    t2 = time.time()
    fo = FastOLS(y, Z, treat_idx=t_idx, cluster=clusters)
    t3 = time.time()

    vc_sm = (
        sm_fit.cov_params().values
        if hasattr(sm_fit.cov_params(), "values")
        else sm_fit.cov_params()
    )
    max_abs = np.abs(vc_sm - fo.vcov).max()

    print(
        f"statsmodels: beta={sm_fit.params[t_idx]:.6f}, SE={sm_fit.bse[t_idx]:.6f}, time={t2-t1:.4f}s"
    )
    print(f"FastOLS:     beta={fo.coef():.6f},     SE={fo.se_robust():.6f},  time={t3-t2:.4f}s")
    print(f"Max |Δ| (VCOV): {max_abs:.2e}")

    assert np.allclose(sm_fit.params[t_idx], fo.coef(), rtol=RTOL)
    assert np.allclose(sm_fit.bse[t_idx], fo.se_robust(), rtol=RTOL)
    assert np.allclose(vc_sm, fo.vcov, rtol=RTOL)


def test_weighted_clustered_crv1_matches_statsmodels_verbose():
    print("\n=== [CRV1 / Weighted + Clustered] Parity vs statsmodels ===")
    y, Z, t_idx = design(n=6000, seed=4, binary_t=False)
    clusters = clusters_for(n=y.size, g=24)
    rng = np.random.default_rng(4)
    w = rng.uniform(0.6, 1.8, size=y.size)

    # statsmodels WLS via sqrt(w) transform + clustered VCOV
    yw = y * np.sqrt(w)
    Zw = Z * np.sqrt(w)[:, None]

    t1 = time.time()
    sm_fit = sm.OLS(yw, Zw).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    t2 = time.time()
    fo = FastOLS(y, Z, treat_idx=t_idx, weights=w, cluster=clusters)
    t3 = time.time()

    vc_sm = (
        sm_fit.cov_params().values
        if hasattr(sm_fit.cov_params(), "values")
        else sm_fit.cov_params()
    )
    max_abs = np.abs(vc_sm - fo.vcov).max()

    print(
        f"statsmodels: beta={sm_fit.params[t_idx]:.6f}, SE={sm_fit.bse[t_idx]:.6f}, time={t2-t1:.4f}s"
    )
    print(f"FastOLS:     beta={fo.coef():.6f},     SE={fo.se_robust():.6f},  time={t3-t2:.4f}s")
    print(f"Max |Δ| (VCOV): {max_abs:.2e}")

    assert np.allclose(sm_fit.params[t_idx], fo.coef(), rtol=RTOL)
    assert np.allclose(sm_fit.bse[t_idx], fo.se_robust(), rtol=RTOL)
    assert np.allclose(vc_sm, fo.vcov, rtol=RTOL)


# ---------------------------------------------------------------------
# K, t_metric, c_perm_vector, and permutation API behavior
# ---------------------------------------------------------------------
def test_K_and_c_perm_vector_behavior_verbose():
    print("\n=== [K and c_perm_vector] Correct metric under OLS and WLS ===")
    n = 1200
    rng = np.random.default_rng(5)
    x = rng.normal(size=n)
    T = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, T])
    y = 0.3 + 0.7 * x + 1.9 * T + rng.normal(size=n)

    # OLS
    ols = FastOLS(y, X, treat_idx=2)
    K_ols_expected = ols.c_vector @ X[:, 2]
    print(f"OLS:   K={ols.K:.6f}, expected cᵀT={K_ols_expected:.6f}")
    assert np.isclose(ols.K, K_ols_expected, rtol=RTOL)
    assert np.allclose(ols.c_perm_vector, ols.c_vector, rtol=0.0)

    # NEW: t_metric should equal X[:,2] under OLS and give K via cᵀ t_metric
    assert np.allclose(ols.t_metric, X[:, 2])
    assert np.isclose(ols.K, float(ols.c_vector @ ols.t_metric), rtol=RTOL)

    # WLS
    w = rng.uniform(0.5, 2.0, size=n)
    wls = FastOLS(y, X, treat_idx=2, weights=w)
    Xw = X * np.sqrt(w)[:, None]
    K_wls_expected = wls.c_vector @ Xw[:, 2]
    print(f"WLS:   K={wls.K:.6f}, expected cᵀT_w={K_wls_expected:.6f}")
    assert np.isclose(wls.K, K_wls_expected, rtol=RTOL)
    assert np.allclose(wls.c_perm_vector, wls.c_vector * np.sqrt(w), rtol=0.0)

    # NEW: t_metric should equal weighted treatment column under WLS
    assert np.allclose(wls.t_metric, Xw[:, 2])
    assert np.isclose(wls.K, float(wls.c_vector @ wls.t_metric), rtol=RTOL)


def test_permuted_stats_wrapper_and_kernel_verbose():
    print("\n=== [Permutation API] Wrapper vs kernel; 2-D and 1-D inputs ===")
    n, R = 1500, 4000
    rng = np.random.default_rng(6)
    x = rng.normal(size=n)
    T = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, T])
    y = X @ np.array([0.2, 0.5, 1.2]) + rng.normal(size=n)

    fo = FastOLS(y, X, treat_idx=2)

    # Build permuted outcomes (2-D)
    Y = rng.normal(
        size=(R, n)
    )  # generic permuted-outcome matrix (already in correct metric for OLS)
    # Wrapper vs kernel
    z_wrap = fo.permuted_stats(Y)
    z_kernel = fast_permuted_stats(fo.c_perm_vector, Y)
    assert np.allclose(z_wrap, z_kernel, rtol=0.0, atol=1e-12)
    assert z_wrap.shape == (R,)

    # 1-D vector path (single permutation)
    y1 = rng.normal(size=n)
    z1 = fo.permuted_stats(y1)
    assert z1.shape == (1,)
    assert np.isclose(z1[0], float(fo.c_perm_vector @ y1))


# ---------------------------------------------------------------------
# Shape/dtype guards in permuted_stats
# ---------------------------------------------------------------------
def test_permuted_stats_shape_dtype_guards():
    n = 100
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    T = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, T])
    y = X @ np.array([0.3, -0.4, 0.9]) + rng.normal(size=n)
    fo = FastOLS(y, X, treat_idx=2)

    with pytest.raises(ValueError):
        fo.permuted_stats(np.random.normal(size=(10, n + 1)))  # wrong width

    with pytest.raises(ValueError):
        fo.permuted_stats(np.random.normal(size=(10, n, 2)))  # 3-D array

    with pytest.raises(ValueError):
        fo.permuted_stats(np.random.normal(size=(n + 1,)))  # wrong length


# ---------------------------------------------------------------------
# Engine guards and friendly errors
# ---------------------------------------------------------------------
def test_n_le_k_raises():
    n = 3
    y = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), np.arange(n), np.arange(n)])  # n == k == 3
    with pytest.raises(ValueError, match="need n > k"):
        FastOLS(y, X, treat_idx=2)


def test_treat_idx_out_of_bounds_raises():
    n = 50
    rng = np.random.default_rng(9)
    y = rng.normal(size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    with pytest.raises(IndexError, match="treat_idx out of bounds"):
        FastOLS(y, X, treat_idx=10)


def test_cluster_length_mismatch_raises():
    # Build a valid dataset
    y, Z, t_idx = design(n=200, seed=10, binary_t=True)
    # Correct clusters (length n)
    clusters = clusters_for(n=y.size, g=10)
    # Make them the wrong length
    bad_clusters = clusters[:-1]  # length n-1

    with pytest.raises(ValueError, match="cluster must have the same length as y/X"):
        FastOLS(y, Z, treat_idx=t_idx, cluster=bad_clusters)


def test_too_few_clusters_raises():
    # Build a valid dataset
    y, Z, t_idx = design(n=200, seed=11, binary_t=True)
    # Only one cluster (G=1) -> CRV1 should refuse
    one_cluster = np.zeros(y.size, dtype=int)

    with pytest.raises(ValueError, match="CRV1 requires at least 2 clusters"):
        FastOLS(y, Z, treat_idx=t_idx, cluster=one_cluster)
