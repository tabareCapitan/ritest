# tests/test_run.py
from typing import cast

import numpy as np
import pandas as pd

from ritest.config import ritest_config
from ritest.run import ritest


# -----------------------------
# helpers
# -----------------------------
def make_linear_df(n=240, seed=123, beta=0.6):
    """Synthetic linear DGP (row-level T)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    T = rng.integers(0, 2, size=n)
    y = beta * T + 0.35 * x1 - 0.2 * x2 + rng.normal(scale=1.0, size=n)
    w = np.exp(rng.normal(scale=0.2, size=n))  # positive analytic weights
    clusters = (np.arange(n) // 12).astype(int)  # ~20 clusters
    strata = np.where(rng.random(n) < 0.5, "A", "B")  # 2 strata
    df = pd.DataFrame(
        {"y": y, "T": T, "x1": x1, "x2": x2, "w": w, "clu": clusters, "str": strata}
    )
    return df


def make_linear_df_cluster_T(n=240, seed=321, beta=0.6):
    """Synthetic linear DGP with cluster-level treatment (T constant within cluster)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    clusters = (np.arange(n) // 12).astype(int)  # ~20 clusters
    G = clusters.max() + 1
    T_g = rng.integers(0, 2, size=G)  # one T per cluster
    T = T_g[clusters]  # broadcast to rows
    y = beta * T + 0.35 * x1 - 0.2 * x2 + rng.normal(scale=1.0, size=n)
    w = np.exp(rng.normal(scale=0.2, size=n))
    strata = np.where(rng.random(n) < 0.5, "A", "B")
    df = pd.DataFrame(
        {"y": y, "T": T, "x1": x1, "x2": x2, "w": w, "clu": clusters, "str": strata}
    )
    return df


def diff_in_means(d: pd.DataFrame) -> float:
    """Generic stat function: difference in means y|T=1 minus y|T=0."""
    g = d.groupby("T")["y"].mean()
    return float(g.get(1, 0.0) - g.get(0, 0.0))


# -----------------------------
# tests: fast-linear path
# -----------------------------
def test_formula_ci_modes_none_bounds_grid():
    df = make_linear_df()

    # ci_mode = none  → no coef-CI at all
    with ritest_config({"reps": 399, "seed": 7, "ci_mode": "none", "n_jobs": 1}):
        r_none = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")
    assert r_none.coef_ci_bounds is None
    assert r_none.coef_ci_band is None
    assert isinstance(r_none.pval, float)
    assert 0.0 <= r_none.pval <= 1.0

    # ci_mode = bounds  → just (lo, hi)
    with ritest_config({"reps": 399, "seed": 7, "ci_mode": "bounds", "n_jobs": 1}):
        r_bounds = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")
    assert (
        isinstance(r_bounds.coef_ci_bounds, tuple) and len(r_bounds.coef_ci_bounds) == 2
    )
    lo, hi = r_bounds.coef_ci_bounds
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert r_bounds.coef_ci_band is None
    assert r_bounds.band_valid_linear is True

    # ci_mode = grid  → band (and bounds)
    with ritest_config({"reps": 399, "seed": 7, "ci_mode": "grid", "n_jobs": 1}):
        r_grid = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")
    assert r_grid.coef_ci_band is not None
    grid, pvals = r_grid.coef_ci_band
    assert grid.ndim == 1 and pvals.ndim == 1 and grid.size == pvals.size
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))
    assert r_grid.band_valid_linear is True


def test_ci_mode_none_matches_stats_without_coef_ci_work():
    df = make_linear_df()
    base = {"reps": 211, "seed": 17, "n_jobs": 1}

    with ritest_config({**base, "ci_mode": "none"}):
        r_none = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")
    with ritest_config({**base, "ci_mode": "bounds"}):
        r_bounds = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")

    # identical permutation outcomes
    assert np.isclose(r_none.obs_stat, r_bounds.obs_stat)
    assert np.isclose(r_none.pval, r_bounds.pval)
    assert r_none.pval_ci == r_bounds.pval_ci
    assert r_none.reps == r_bounds.reps == base["reps"]
    assert r_none.c == r_bounds.c
    assert r_none.perm_stats is not None and r_bounds.perm_stats is not None
    assert np.array_equal(r_none.perm_stats, r_bounds.perm_stats)

    # ci_mode none should skip coef CI artifacts entirely
    assert r_none.coef_ci_bounds is None
    assert r_none.coef_ci_band is None
    assert r_none.band_valid_linear is False

    # bounds mode still produces bounds but no band
    assert r_bounds.coef_ci_bounds is not None
    assert r_bounds.coef_ci_band is None
    assert r_bounds.band_valid_linear is True


def test_formula_parallel_equals_serial():
    df = make_linear_df()
    base = {"reps": 257, "seed": 99, "ci_mode": "none"}

    with ritest_config({**base, "n_jobs": 1}):
        r1 = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")
    with ritest_config({**base, "n_jobs": 2}):
        r2 = ritest(df, permute_var="T", formula="y ~ T + x1 + x2", stat="T")

    # Deterministic permutations => identical perm_stats and pval
    assert r1.reps == r2.reps == base["reps"]
    assert r1.perm_stats is not None and r2.perm_stats is not None
    ps1 = cast(np.ndarray, r1.perm_stats)
    ps2 = cast(np.ndarray, r2.perm_stats)
    assert np.array_equal(ps1, ps2)
    assert r1.pval == r2.pval


def test_formula_flags_and_settings():
    # Use cluster-level treatment so cluster-wise permutations are valid
    df = make_linear_df_cluster_T()
    cfg = {"reps": 149, "seed": 11, "ci_mode": "none", "n_jobs": 1}
    with ritest_config(cfg):
        r = ritest(
            df,
            permute_var="T",
            formula="y ~ T + x1 + x2",
            stat="T",
            cluster="clu",
            strata="str",
            weights="w",
        )
    # flags
    assert r.clustered is True
    assert r.stratified is True
    assert r.weights is True
    # settings snapshot
    s = r.settings
    for key in [
        "alpha",
        "seed",
        "reps",
        "ci_method",
        "ci_mode",
        "ci_range",
        "ci_step",
        "alternative",
        "n_jobs",
    ]:
        assert key in s
    assert s["seed"] == cfg["seed"]
    assert s["reps"] == cfg["reps"]
    assert s["ci_mode"] == cfg["ci_mode"]
    assert isinstance(s.get("runtime_sec", 0.0), float)


# -----------------------------
# tests: generic stat_fn path
# -----------------------------
def test_generic_gating_skips_coef_ci():
    df = make_linear_df()

    # Coefficient CIs are not available for stat_fn path, regardless of ci_mode
    with ritest_config({"reps": 199, "seed": 5, "ci_mode": "bounds", "n_jobs": 2}):
        r_bounds = ritest(df, permute_var="T", stat_fn=diff_in_means)
    assert r_bounds.coef_ci_bounds is None
    assert r_bounds.coef_ci_band is None

    with ritest_config({"reps": 199, "seed": 5, "ci_mode": "grid", "n_jobs": 2}):
        r_grid = ritest(df, permute_var="T", stat_fn=diff_in_means)
    assert r_grid.coef_ci_bounds is None
    assert r_grid.coef_ci_band is None
    assert r_grid.band_valid_linear is False  # generic path


def test_generic_p_and_ci_and_shapes():
    df = make_linear_df()
    with ritest_config(
        {
            "reps": 299,
            "seed": 13,
            "ci_method": "clopper-pearson",
            "ci_mode": "none",
            "n_jobs": 1,
        }
    ):
        r = ritest(df, permute_var="T", stat_fn=diff_in_means)
    # p-value & CI sanity
    assert 0.0 <= r.pval <= 1.0
    lo, hi = r.pval_ci
    assert 0.0 <= lo <= hi <= 1.0
    # perm_stats shape/dtype
    assert r.perm_stats is not None
    ps = cast(np.ndarray, r.perm_stats)
    assert ps.shape == (r.reps,)
    assert ps.dtype == np.float64


# -----------------------------
# tests: negative / validation surfaced via run()
# -----------------------------
def test_run_rejects_both_formula_and_stat_fn():
    df = make_linear_df()
    with ritest_config({"reps": 99, "seed": 3, "ci_mode": "none"}):
        try:
            ritest(
                df,
                permute_var="T",
                formula="y ~ T + x1",
                stat="T",
                stat_fn=diff_in_means,
            )
            assert (
                False
            ), "Expected ValueError when both formula and stat_fn are supplied"
        except ValueError:
            pass


def test_run_requires_stat_in_formula_mode():
    df = make_linear_df()
    with ritest_config({"reps": 99, "seed": 3, "ci_mode": "none"}):
        try:
            ritest(df, permute_var="T", formula="y ~ T + x1")  # missing stat=
            assert False, "Expected ValueError when 'stat' missing in formula mode"
        except ValueError:
            pass
