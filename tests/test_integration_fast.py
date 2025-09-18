# tests/test_integration_fast.py
import numpy as np
import pandas as pd
import pytest

from ritest import RitestResult, ritest


def _toy_df(n=80, seed=123):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    T = rng.integers(0, 2, size=n)  # ~50/50 treatment
    y = 1.5 * T + 0.3 * X1 - 0.2 * X2 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2})


def test_fast_basic_smoke_and_determinism():
    df = _toy_df(n=120, seed=2024)

    # Minimal fast-OLS call; no coef CI artifacts (ci_mode="none")
    res1 = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=600,
        seed=42,
        alpha=0.05,
        ci_method="clopper-pearson",
        ci_mode="none",
        n_jobs=1,
    )

    res2 = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=600,
        seed=42,
        alpha=0.05,
        ci_method="clopper-pearson",
        ci_mode="none",
        n_jobs=4,
    )

    # Types and basic structure
    assert isinstance(res1, RitestResult)
    assert isinstance(res2, RitestResult)

    # Basic invariants
    assert res1.reps == 600
    assert res1.alternative == "two-sided"
    assert 0.0 <= res1.pval <= 1.0
    assert isinstance(res1.pval_ci, tuple) and len(res1.pval_ci) == 2
    lo, hi = res1.pval_ci
    assert 0.0 <= lo <= hi <= 1.0

    # No coefficient CI artifacts in ci_mode="none"
    assert res1.coef_ci_bounds is None
    assert res1.coef_ci_band is None

    # plot() must raise without a band
    with pytest.raises(ValueError):
        _ = res1.plot()

    # Determinism across threads (same seed/inputs)
    assert res1.obs_stat == pytest.approx(res2.obs_stat, rel=0, abs=0)
    assert res1.pval == pytest.approx(res2.pval, rel=0, abs=0)
    assert res1.c == res2.c
    assert res1.pval_ci == pytest.approx(res2.pval_ci, rel=0, abs=0)

    # Summary should return a string (and not explode)
    s = res1.summary(print_out=False)
    assert isinstance(s, str) and len(s) > 0
