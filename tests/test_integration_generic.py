# tests/test_integration_generic.py
# Generic-path integration tests for ritest using a statsmodels-based stat_fn
#
# Run:  pytest -q tests/test_integration_generic.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from ritest import RitestResult, ritest, ritest_config, ritest_reset

# Skip the whole module if statsmodels is not available
sm = pytest.importorskip("statsmodels.api")


# ------------------ data helpers (same shapes as fast path) ------------------ #


def _toy_df(n: int = 200, beta: float = 1.2, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    T = rng.integers(0, 2, size=n)
    y = beta * T + 0.5 * X1 - 0.25 * X2 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2})


def _df_with_strata(n_per: int = 50, beta: float = 0.8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    strata = np.repeat([0, 1], n_per)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    half = n_per // 2
    T = np.concatenate([np.tile([0, 1], half), np.tile([0, 1], half)])
    rng.shuffle(T[:n_per])
    rng.shuffle(T[n_per:])
    y = beta * T + 0.3 * X1 - 0.2 * X2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2, "S": strata})


def _df_with_clusters(
    n_clusters: int = 20, m_per: int = 10, beta: float = 1.0, seed: int = 99
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    G = n_clusters
    n = G * m_per
    C = np.repeat(np.arange(G), m_per)
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
    rng = np.random.default_rng(seed)
    total_clusters = 2 * clusters_per_stratum
    base_ids = []
    for s in [0, 1]:
        start = s * clusters_per_stratum
        base_ids.append(np.arange(start, start + clusters_per_stratum, dtype=int))
    base_ids = np.concatenate(base_ids)
    C = np.repeat(base_ids, m_per * repeats_per_stratum)
    n = C.size
    T_g = rng.integers(0, 2, size=total_clusters)
    T = T_g[C]
    S = (C // clusters_per_stratum).astype(int)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    y = beta * T + 0.2 * X1 - 0.3 * X2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2, "C": C, "S": S})


# ------------------ generic stat functions (statsmodels) ------------------ #


def _t_stat_T(df_view: pd.DataFrame) -> float:
    """
    Return the t-stat for coefficient on T from OLS: y ~ T + X1 + X2
    """
    y = df_view["y"].to_numpy()
    X = df_view[["T", "X1", "X2"]].to_numpy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X)
    res = model.fit()
    # columns are: const, T, X1, X2  -> index 1 is T
    return float(res.tvalues[1])


def _t_stat_T_wls(df_view: pd.DataFrame) -> float:
    """
    Return the t-stat for coefficient on T from WLS with analytic weights in column 'w'.
    """
    y = df_view["y"].to_numpy()
    X = df_view[["T", "X1", "X2"]].to_numpy()
    w = df_view["w"].to_numpy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.WLS(y, X, weights=w)
    res = model.fit()
    return float(res.tvalues[1])


# ------------------ 1) smoke + threads determinism ------------------ #


def test_generic_basic_smoke_and_determinism_threads():
    df = _toy_df(n=160, seed=2024)

    res1 = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
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
        stat_fn=_t_stat_T,
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
    # generic path: no coef CIs unless explicitly grid+coef_ci_generic
    assert res1.coef_ci_bounds is None and res1.coef_ci_band is None

    with pytest.raises(ValueError):
        _ = res1.plot()

    assert res1.obs_stat == pytest.approx(res2.obs_stat, rel=0, abs=0)
    assert res1.pval == pytest.approx(res2.pval, rel=0, abs=0)
    assert res1.c == res2.c
    assert res1.pval_ci == pytest.approx(res2.pval_ci, rel=0, abs=0)

    s = res1.summary(print_out=False)
    assert isinstance(s, str) and len(s) > 0


# ------------------ 2) chunking vs eager parity (all modes) ------------------ #


@pytest.mark.parametrize("mode", ["none", "bounds", "grid"])
def test_generic_chunking_vs_eager_identical(mode: Literal["none", "bounds", "grid"]):
    df = _toy_df(n=200, seed=2025)

    # likely eager for these sizes
    with ritest_config({"perm_chunk_bytes": 2 * 1024 * 1024}):
        eager = ritest(
            df=df,
            permute_var="T",
            stat_fn=_t_stat_T,
            reps=500,
            seed=7,
            alpha=0.05,
            ci_method="clopper-pearson",
            ci_mode=mode,
            n_jobs=2,
            # for generic path, only GRID supports band and only with coef_ci_generic=True
            coef_ci_generic=(mode == "grid"),
        )

    # force chunking
    with ritest_config({"perm_chunk_bytes": 32 * 1024}):
        chunked = ritest(
            df=df,
            permute_var="T",
            stat_fn=_t_stat_T,
            reps=500,
            seed=7,
            alpha=0.05,
            ci_method="clopper-pearson",
            ci_mode=mode,
            n_jobs=2,
            coef_ci_generic=(mode == "grid"),
        )

    assert eager.obs_stat == pytest.approx(chunked.obs_stat, rel=0, abs=0)
    assert eager.pval == pytest.approx(chunked.pval, rel=0, abs=0)
    assert eager.c == chunked.c

    # generic gating rules
    if mode == "none":
        assert eager.coef_ci_bounds is None and eager.coef_ci_band is None
    elif mode == "bounds":
        # bounds not supported for generic path
        assert eager.coef_ci_bounds is None and eager.coef_ci_band is None
    else:  # grid
        assert eager.coef_ci_bounds is None
        if eager.coef_ci_band is not None:
            beta_grid, pvals = eager.coef_ci_band
            assert isinstance(beta_grid, np.ndarray) and isinstance(pvals, np.ndarray)
            assert beta_grid.ndim == 1 and pvals.ndim == 1 and beta_grid.size == pvals.size
            assert np.all((pvals >= 0.0) & (pvals <= 1.0))


# ------------------ 3) tail alternatives monotonicity ------------------ #


def test_generic_tail_alternatives_monotonicity():
    df = _toy_df(n=220, beta=1.0, seed=11)

    p_right = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=500,
        seed=101,
        ci_mode="none",
        alternative="right",
    ).pval
    p_two = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=500,
        seed=101,
        ci_mode="none",
        alternative="two-sided",
    ).pval
    p_left = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=500,
        seed=101,
        ci_mode="none",
        alternative="left",
    ).pval

    assert p_right <= p_two + 1e-12
    assert p_two <= p_left + 1e-12


# ------------------ 4) ci mode gating + shapes (generic) ------------------ #


def test_generic_ci_modes_gating_and_shapes():
    df = _toy_df(n=180, beta=0.7, seed=5)

    r_none = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=400,
        seed=0,
        ci_mode="none",
    )
    assert r_none.coef_ci_bounds is None and r_none.coef_ci_band is None

    r_bounds = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=400,
        seed=0,
        ci_mode="bounds",
    )
    # bounds not supported on generic
    assert r_bounds.coef_ci_bounds is None and r_bounds.coef_ci_band is None

    r_grid_no = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=400,
        seed=0,
        ci_mode="grid",  # coef_ci_generic defaults False
    )
    assert r_grid_no.coef_ci_bounds is None and r_grid_no.coef_ci_band is None

    r_grid_yes = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=400,
        seed=0,
        ci_mode="grid",
        coef_ci_generic=True,
    )
    assert r_grid_yes.coef_ci_bounds is None and r_grid_yes.coef_ci_band is not None
    beta_grid, pvals = r_grid_yes.coef_ci_band
    assert isinstance(beta_grid, np.ndarray) and isinstance(pvals, np.ndarray)
    assert beta_grid.ndim == 1 and pvals.ndim == 1 and beta_grid.size == pvals.size
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


# ------------------ 5/6/7) strata / cluster / both ------------------ #


def test_generic_stratified_permutations():
    df = _df_with_strata(n_per=60, beta=0.6, seed=13)
    res = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        strata="S",
        reps=400,
        seed=77,
        ci_mode="none",
    )
    assert res.stratified is True and res.clustered is False
    assert 0.0 <= res.pval <= 1.0


def test_generic_cluster_permutations_cluster_constant_T():
    df = _df_with_clusters(n_clusters=10, m_per=8, beta=0.9, seed=21)
    g = df.groupby("C")["T"].nunique()
    assert (g == 1).all()

    res = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        cluster="C",
        reps=400,
        seed=19,
        ci_mode="none",
    )
    assert res.clustered is True and res.stratified is False
    assert 0.0 <= res.pval <= 1.0


def test_generic_cluster_within_strata_permutations():
    df = _df_cluster_within_strata(
        clusters_per_stratum=6, m_per=5, repeats_per_stratum=2, beta=0.8, seed=111
    )
    g = df.groupby(["S", "C"])["T"].nunique()
    assert (g == 1).all()

    res = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        cluster="C",
        strata="S",
        reps=400,
        seed=55,
        ci_mode="none",
    )
    assert res.clustered is True and res.stratified is True
    assert 0.0 <= res.pval <= 1.0


# ------------------ 6) weights smoke (WLS) ------------------ #


def test_generic_weighted_vs_unweighted_smoke():
    df = _toy_df(n=160, beta=0.9, seed=222)
    rng = np.random.default_rng(9)
    w = 0.5 + rng.random(df.shape[0])

    r_unw = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=450,
        seed=3,
        ci_mode="none",
    )
    r_w = ritest(
        df=df.assign(w=w),
        permute_var="T",
        stat_fn=_t_stat_T_wls,
        reps=450,
        seed=3,
        ci_mode="none",
    )
    assert 0.0 <= r_unw.pval <= 1.0
    assert 0.0 <= r_w.pval <= 1.0


# ------------------ 7) prebuilt permutations path ------------------ #


def test_generic_prebuilt_permutations_override_reps_and_match_seeded():
    df = _toy_df(n=120, seed=5)
    reps_mat = 360
    rng = np.random.default_rng(2025)
    base_T = df["T"].to_numpy()
    perms = np.vstack([rng.permutation(base_T) for _ in range(reps_mat)])

    r_mat = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        ci_mode="none",
        reps=999,
        seed=999,
        permutations=perms,  # reps/seed ignored
    )
    assert r_mat.reps == reps_mat
    assert r_mat.perm_stats is not None
    assert r_mat.perm_stats.shape == (reps_mat,)

    r_mat2 = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        ci_mode="none",
        reps=1,
        seed=1,
        permutations=perms,
    )
    assert r_mat.obs_stat == pytest.approx(r_mat2.obs_stat, rel=0, abs=0)
    assert r_mat.pval == pytest.approx(r_mat2.pval, rel=0, abs=0)
    assert r_mat.c == r_mat2.c


# ------------------ 8) config reset safety ------------------ #


def test_generic_config_context_manager_resets_on_exit():
    df = _toy_df(n=100, seed=8)
    ritest_reset()

    with ritest_config({"perm_chunk_bytes": 1024, "n_jobs": 1}):
        a = ritest(
            df=df,
            permute_var="T",
            stat_fn=_t_stat_T,
            reps=200,
            seed=4,
            ci_mode="none",
        )

    b = ritest(
        df=df,
        permute_var="T",
        stat_fn=_t_stat_T,
        reps=200,
        seed=4,
        ci_mode="none",
    )
    assert a.pval == pytest.approx(b.pval, rel=0, abs=0)
