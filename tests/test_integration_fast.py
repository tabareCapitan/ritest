# tests/test_integration_fast.py
# Comprehensive fast-path (linear model) integration tests for ritest
#
# Run:  pytest -q tests/test_integration_fast.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from ritest import RitestResult, ritest, ritest_config, ritest_reset

# ------------------ data helpers ------------------ #


def _toy_df(n: int = 200, beta: float = 1.2, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    T = rng.integers(0, 2, size=n)
    y = beta * T + 0.5 * X1 - 0.25 * X2 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2})


def _df_with_strata(n_per: int = 50, beta: float = 0.8, seed: int = 7) -> pd.DataFrame:
    # two strata; randomize within strata
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    strata = np.repeat([0, 1], n_per)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    # balanced T within each stratum
    half = n_per // 2
    T = np.concatenate([np.tile([0, 1], half), np.tile([0, 1], half)])
    rng.shuffle(T[:n_per])
    rng.shuffle(T[n_per:])
    y = beta * T + 0.3 * X1 - 0.2 * X2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2, "S": strata})


def _df_with_clusters(
    n_clusters: int = 20, m_per: int = 10, beta: float = 1.0, seed: int = 99
) -> pd.DataFrame:
    # cluster-constant T (required for cluster permutation invariants)
    rng = np.random.default_rng(seed)
    G = n_clusters
    m = m_per
    n = G * m
    C = np.repeat(np.arange(G), m)
    T_g = rng.integers(0, 2, size=G)
    T = T_g[C]
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    y = beta * T + 0.4 * X1 - 0.4 * X2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2, "C": C})


def _df_cluster_within_strata(
    clusters_per_stratum: int = 6,
    m_per: int = 5,
    repeats_per_stratum: int = 2,
    beta: float = 0.9,
    seed: int = 321,
) -> pd.DataFrame:
    """
    Build clusters nested in two strata; each cluster has m_per units; we repeat the set of clusters
    per stratum 'repeats_per_stratum' times for a larger sample. T is cluster-constant.
    """
    rng = np.random.default_rng(seed)
    # total clusters = 2 strata * clusters_per_stratum
    total_clusters = 2 * clusters_per_stratum
    # cluster ids by stratum, then repeat
    base_ids = []
    for s in [0, 1]:
        start = s * clusters_per_stratum
        base_ids.append(np.arange(start, start + clusters_per_stratum, dtype=int))
    base_ids = np.concatenate(base_ids)
    C = np.repeat(base_ids, m_per * repeats_per_stratum)
    n = C.size
    # cluster-constant T
    T_g = rng.integers(0, 2, size=total_clusters)
    T = T_g[C]
    # strata label from cluster id
    S = (C // clusters_per_stratum).astype(int)
    # covariates and outcome
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    y = beta * T + 0.2 * X1 - 0.3 * X2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2, "C": C, "S": S})


# ------------------ 1) smoke + threads determinism ------------------ #


def test_fast_basic_smoke_and_determinism_threads():
    df = _toy_df(n=180, seed=2024)

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

    assert isinstance(res1, RitestResult) and isinstance(res2, RitestResult)
    assert res1.reps == 600
    assert res1.alternative == "two-sided"
    assert 0.0 <= res1.pval <= 1.0
    assert isinstance(res1.pval_ci, tuple) and len(res1.pval_ci) == 2
    lo, hi = res1.pval_ci
    assert 0.0 <= lo <= hi <= 1.0
    assert res1.coef_ci_bounds is None and res1.coef_ci_band is None

    with pytest.raises(ValueError):
        _ = res1.plot()

    # same inputs + seed ⇒ identical numerics
    assert res1.obs_stat == pytest.approx(res2.obs_stat, rel=0, abs=0)
    assert res1.pval == pytest.approx(res2.pval, rel=0, abs=0)
    assert res1.c == res2.c
    assert res1.pval_ci == pytest.approx(res2.pval_ci, rel=0, abs=0)

    s = res1.summary(print_out=False)
    assert isinstance(s, str) and len(s) > 0


# ------------------ 2) chunking vs eager parity (all modes) ------------------ #


@pytest.mark.parametrize("mode", ["none", "bounds", "band"])
def test_chunking_vs_eager_identical(mode: Literal["none", "bounds", "band"]):
    df = _toy_df(n=220, seed=2025)

    # likely eager for these sizes
    with ritest_config({"perm_chunk_bytes": 2 * 1024 * 1024}):
        eager = ritest(
            df=df,
            permute_var="T",
            formula="y ~ T + X1 + X2",
            stat="T",
            reps=500,
            seed=7,
            alpha=0.05,
            ci_method="clopper-pearson",
            ci_mode=mode,
            n_jobs=2,
        )

    # force chunking
    with ritest_config({"perm_chunk_bytes": 32 * 1024}):
        chunked = ritest(
            df=df,
            permute_var="T",
            formula="y ~ T + X1 + X2",
            stat="T",
            reps=500,
            seed=7,
            alpha=0.05,
            ci_method="clopper-pearson",
            ci_mode=mode,
            n_jobs=2,
        )

    assert eager.obs_stat == pytest.approx(chunked.obs_stat, rel=0, abs=0)
    assert eager.pval == pytest.approx(chunked.pval, rel=0, abs=0)
    assert eager.c == chunked.c

    if mode == "none":
        assert eager.coef_ci_bounds is None and eager.coef_ci_band is None
    elif mode == "bounds":
        assert isinstance(eager.coef_ci_bounds, tuple) and eager.coef_ci_band is None
    else:
        assert isinstance(eager.coef_ci_bounds, tuple) and isinstance(
            eager.coef_ci_band, tuple
        )


# ------------------ 3) tail alternatives monotonicity ------------------ #


def test_tail_alternatives_monotonicity():
    df = _toy_df(n=240, beta=1.0, seed=11)

    p_right = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=500,
        seed=101,
        ci_mode="none",
        alternative="right",
    ).pval
    p_two = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=500,
        seed=101,
        ci_mode="none",
        alternative="two-sided",
    ).pval
    p_left = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=500,
        seed=101,
        ci_mode="none",
        alternative="left",
    ).pval

    assert p_right <= p_two + 1e-12
    assert p_two <= p_left + 1e-12


# ------------------ 4) ci mode gating + shapes ------------------ #


def test_ci_bounds_and_band_gating_and_shapes():
    df = _toy_df(n=200, beta=0.7, seed=5)

    r_none = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=400,
        seed=0,
        ci_mode="none",
    )
    assert r_none.coef_ci_bounds is None and r_none.coef_ci_band is None

    r_bounds = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=400,
        seed=0,
        ci_mode="bounds",
    )
    assert isinstance(r_bounds.coef_ci_bounds, tuple) and r_bounds.coef_ci_band is None
    lo, hi = r_bounds.coef_ci_bounds
    assert (np.isfinite(lo) and np.isfinite(hi)) or (np.isnan(lo) and np.isnan(hi))

    r_band = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=400,
        seed=0,
        ci_mode="band",
    )
    assert isinstance(r_band.coef_ci_bounds, tuple) and isinstance(
        r_band.coef_ci_band, tuple
    )
    beta_grid, pvals = r_band.coef_ci_band
    assert isinstance(beta_grid, np.ndarray) and isinstance(pvals, np.ndarray)
    assert beta_grid.ndim == 1 and pvals.ndim == 1 and beta_grid.size == pvals.size
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


# ------------------ 5) strata / 6) cluster / 7) both ------------------ #


def test_stratified_permutations():
    df = _df_with_strata(n_per=60, beta=0.6, seed=13)
    res = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        strata="S",
        reps=400,
        seed=77,
        ci_mode="none",
    )
    assert res.stratified is True and res.clustered is False
    assert 0.0 <= res.pval <= 1.0


def test_cluster_permutations_cluster_constant_T():
    df = _df_with_clusters(n_clusters=10, m_per=8, beta=0.9, seed=21)
    g = df.groupby("C")["T"].nunique()
    assert (g == 1).all()

    res = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        cluster="C",
        reps=400,
        seed=19,
        ci_mode="none",
    )
    assert res.clustered is True and res.stratified is False
    assert 0.0 <= res.pval <= 1.0


def test_cluster_within_strata_permutations():
    df = _df_cluster_within_strata(
        clusters_per_stratum=6, m_per=5, repeats_per_stratum=2, beta=0.8, seed=111
    )
    g = df.groupby(["S", "C"])["T"].nunique()
    assert (g == 1).all()

    res = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        cluster="C",
        strata="S",
        reps=400,
        seed=55,
        ci_mode="none",
    )
    assert res.clustered is True and res.stratified is True
    assert 0.0 <= res.pval <= 1.0


# ------------------ 8) weights smoke ------------------ #


def test_weighted_vs_unweighted_smoke():
    df = _toy_df(n=180, beta=0.9, seed=222)
    rng = np.random.default_rng(9)
    w = 0.5 + rng.random(df.shape[0])

    r_unw = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=450,
        seed=3,
        ci_mode="none",
    )
    r_w = ritest(
        df=df.assign(w=w),
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        weights="w",
        reps=450,
        seed=3,
        ci_mode="none",
    )
    assert 0.0 <= r_unw.pval <= 1.0
    assert 0.0 <= r_w.pval <= 1.0


# ------------------ 9) prebuilt permutations path ------------------ #


def test_prebuilt_permutations_override_reps_and_match_seeded():
    df = _toy_df(n=140, seed=5)
    reps_mat = 420
    rng = np.random.default_rng(2025)
    # rows are permuted *labels* for T (shape: reps x n)
    base_T = df["T"].to_numpy()
    perms = np.vstack([rng.permutation(base_T) for _ in range(reps_mat)])

    r_mat = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        ci_mode="none",
        reps=999,
        seed=999,
        permutations=perms,  # reps/seed ignored
    )
    assert r_mat.reps == reps_mat
    # guard before accessing .shape (perm_stats is Optional[np.ndarray] in typing)
    assert r_mat.perm_stats is not None
    assert r_mat.perm_stats.shape == (reps_mat,)

    r_mat2 = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        ci_mode="none",
        reps=1,
        seed=1,
        permutations=perms,
    )
    assert r_mat.obs_stat == pytest.approx(r_mat2.obs_stat, rel=0, abs=0)
    assert r_mat.pval == pytest.approx(r_mat2.pval, rel=0, abs=0)
    assert r_mat.c == r_mat2.c


# ------------------ 10) config reset safety ------------------ #


def test_config_context_manager_resets_on_exit():
    df = _toy_df(n=120, seed=8)
    ritest_reset()  # ensure defaults known

    with ritest_config({"perm_chunk_bytes": 1024, "n_jobs": 1}):
        a = ritest(
            df=df,
            permute_var="T",
            formula="y ~ T + X1 + X2",
            stat="T",
            reps=200,
            seed=4,
            ci_mode="none",
        )

    # after exit, run again (internal chunking decision may differ)
    b = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        reps=200,
        seed=4,
        ci_mode="none",
    )
    assert a.pval == pytest.approx(b.pval, rel=0, abs=0)
