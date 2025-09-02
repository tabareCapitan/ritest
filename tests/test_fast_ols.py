import time

import numpy as np
import statsmodels.api as sm

from ritest.engine.fast_ols import NUMBA_OK, FastOLS, fast_permuted_stats

print(f"Permutation backend: {'Numba (parallel)' if NUMBA_OK else 'NumPy fallback'}")


# ============================================================
# Tolerance setting for all tests (tweak if needed)
# ============================================================
RTOL = 1e-3


def test_fastols_all_variants():
    print("\n=== FastOLS full test suite (n = 10000) ===")
    np.random.seed(42)
    n = 10000
    clusters = np.repeat(np.arange(n // 10), 10)
    weights = np.random.uniform(0.5, 2.0, size=n)

    X = np.column_stack([np.ones(n), np.random.normal(size=n)])
    T = np.random.binomial(1, 0.5, size=n)
    Z = np.column_stack([X, T])
    beta_true = 2.0
    y = X @ np.array([1.0, -0.5]) + T * beta_true + np.random.normal(size=n)

    # --- HC1 (no weights, no cluster)
    print("\n==== OLS (White HC1) ====")
    t1 = time.time()
    sm1 = sm.OLS(y, Z).fit(cov_type="HC1")
    t2 = time.time()
    ols1 = FastOLS(y, Z, treat_idx=2)
    t3 = time.time()

    print(f"Statsmodels: β = {sm1.params[-1]:.6f}  SE = {sm1.bse[-1]:.6f}  time = {t2 - t1:.4f}s")
    print(f"FastOLS:     β = {ols1.coef():.6f}  SE = {ols1.se_robust():.6f}  time = {t3 - t2:.4f}s")
    assert np.allclose(sm1.params[-1], ols1.coef(), rtol=RTOL)
    assert np.allclose(sm1.bse[-1], ols1.se_robust(), rtol=RTOL)

    # --- WLS
    print("\n==== WLS (weighted HC1) ====")
    yw = y * np.sqrt(weights)
    Zw = Z * np.sqrt(weights)[:, None]
    t1 = time.time()
    sm2 = sm.OLS(yw, Zw).fit(cov_type="HC1")
    t2 = time.time()
    ols2 = FastOLS(y, Z, treat_idx=2, weights=weights)
    t3 = time.time()

    print(f"Statsmodels: β = {sm2.params[-1]:.6f}  SE = {sm2.bse[-1]:.6f}  time = {t2 - t1:.4f}s")
    print(f"FastOLS:     β = {ols2.coef():.6f}  SE = {ols2.se_robust():.6f}  time = {t3 - t2:.4f}s")
    assert np.allclose(sm2.params[-1], ols2.coef(), rtol=RTOL)
    assert np.allclose(sm2.bse[-1], ols2.se_robust(), rtol=RTOL)

    # --- Clustered (CRV1)
    print("\n==== Clustered (CRV1) ====")
    t1 = time.time()
    sm3 = sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    t2 = time.time()
    ols3 = FastOLS(y, Z, treat_idx=2, cluster=clusters)
    t3 = time.time()

    print(f"Statsmodels: β = {sm3.params[-1]:.6f}  SE = {sm3.bse[-1]:.6f}  time = {t2 - t1:.4f}s")
    print(f"FastOLS:     β = {ols3.coef():.6f}  SE = {ols3.se_robust():.6f}  time = {t3 - t2:.4f}s")
    assert np.allclose(sm3.params[-1], ols3.coef(), rtol=RTOL)
    assert np.allclose(sm3.bse[-1], ols3.se_robust(), rtol=RTOL)

    # --- Sandwich matrix comparison
    print("\n==== Full Sandwich Matrix (CRV1) ====")
    vcov_sm = sm3.cov_params().values if hasattr(sm3.cov_params(), "values") else sm3.cov_params()
    max_diff = np.abs(vcov_sm - ols3.vcov).max()
    print(f"Max abs diff vs statsmodels: {max_diff:.2e}")
    assert np.allclose(vcov_sm, ols3.vcov, rtol=RTOL)

    # --- Permuted dot-product benchmark
    print("\n==== Permuted Dot-Product Benchmark ====")
    c = ols3.c_vector
    Y_perm = y[None, :] + np.random.normal(scale=1.0, size=(10_000, n))
    t1 = time.time()
    z = fast_permuted_stats(c, Y_perm)
    t2 = time.time()
    print(f"10k × cᵀ y_perm → {t2 - t1:.4f}s (avg: {(t2 - t1)/10_000:.6f}s per eval)")
    assert np.allclose(z[0], c @ Y_perm[0], rtol=RTOL)
