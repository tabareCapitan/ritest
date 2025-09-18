# tests/test_validation.py
"""
Comprehensive unit tests for ritest.validation.validate_inputs.

Run with:
    pytest -q tests/test_validation.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from ritest.validation import validate_inputs


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def base_df(rng) -> pd.DataFrame:
    """Deterministic minimal DataFrame with outcome, covariate, treatment."""
    n = 60
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "T": rng.integers(0, 2, size=n),  # integer 0/1
        }
    )


# ------------------------------------------------------------------ #
# 1) Happy paths
# ------------------------------------------------------------------ #
def test_formula_minimal_happy(base_df):
    v = validate_inputs(
        base_df,
        formula="y ~ x + T",
        stat="T",
        permute_var="T",
    )
    assert v.y.shape == (len(base_df),)
    assert v.X.shape[0] == len(base_df)
    assert v.X.shape[1] == 3  # intercept + x + T
    assert v.treat_idx == 2
    assert v.has_cluster is False
    assert v.has_strata is False
    assert v.stat_fn is None
    assert v.T.dtype == np.float64  # coerced to float


def test_formula_with_cluster_strata_weights_happy(base_df, rng):
    df = base_df.copy()
    df["cluster"] = np.repeat(np.arange(12), len(df) // 12).astype(int)
    # Ensure exact length (pad last group if needed)
    if df["cluster"].shape[0] < len(df):
        df.loc[df.index[-1], "cluster"] = 11
    df["strata"] = rng.integers(0, 4, size=len(df))
    df["w"] = rng.uniform(0.2, 3.0, size=len(df))

    v = validate_inputs(
        df,
        formula="y ~ x + T",
        stat="T",
        permute_var="T",
        cluster="cluster",
        strata="strata",
        weights="w",
        alternative="right",
    )
    assert v.has_cluster and v.has_strata
    assert v.weights is not None and v.weights.dtype == np.float64 and np.all(v.weights > 0)
    # cluster/strata must be dense int codes
    assert v.cluster is not None and v.cluster.dtype == np.int64
    assert v.strata is not None and v.strata.dtype == np.int64
    assert v.alternative == "right"


def test_formula_boolean_treatment_column(base_df):
    df = base_df.copy()
    # permute_var is boolean (allowed), RHS regressor is numeric so Patsy names it "Tb"
    df["Tb_bool"] = df["T"].astype(bool)
    df["Tb"] = df["Tb_bool"].astype(int)

    v = validate_inputs(
        df,
        formula="y ~ x + Tb",  # RHS numeric column -> exact name "Tb" in design
        stat="Tb",
        permute_var="Tb_bool",  # boolean permute_var is accepted and coerced to float
    )
    assert v.treat_idx == 2
    assert v.T.dtype == np.float64
    assert set(np.unique(v.T)).issubset({0.0, 1.0})
    # Because permute_var isn't in the formula, we should also get the gentle warning
    assert any("does not appear in the formula" in w for w in v.warnings)


def test_stat_fn_minimal_happy(base_df):
    def mean_x(d: pd.DataFrame) -> float:
        return float(d["x"].mean())

    v = validate_inputs(
        base_df,
        stat_fn=mean_x,
        permute_var="T",
    )
    assert v.stat_fn is mean_x
    assert v.T.shape == (len(base_df),)
    assert v.warmup_time >= 0.0
    assert v.y.size == 0 and v.X.size == 0  # unused arrays in stat_fn mode


# ------------------------------------------------------------------ #
# 2) Alignment after Patsy drops rows (NA in formula vars)
# ------------------------------------------------------------------ #
def test_alignment_kept_rows_across_all_vectors(base_df):
    df = base_df.copy()
    # Inject NA in x for the first 7 rows -> Patsy will drop them
    drop_mask = df.index[:7]
    df.loc[drop_mask, "x"] = np.nan

    # Also provide cluster/strata/weights; they must be subset to the same rows
    df["cluster"] = np.repeat(np.arange(6), len(df) // 6 + 1)[: len(df)]
    df["strata"] = np.repeat(np.arange(3), len(df) // 3 + 1)[: len(df)]
    df["w"] = 1.0

    v = validate_inputs(
        df,
        formula="y ~ x + T",
        stat="T",
        permute_var="T",
        cluster="cluster",
        strata="strata",
        weights="w",
    )

    kept = df["x"].notna().to_numpy()
    assert v.y.shape[0] == kept.sum()
    assert v.X.shape[0] == kept.sum()
    assert v.T.shape[0] == kept.sum()
    assert v.cluster is not None and v.cluster.shape[0] == kept.sum()
    assert v.strata is not None and v.strata.shape[0] == kept.sum()
    assert v.weights is not None and v.weights.shape[0] == kept.sum()

    # Check T equals the subset of original T
    assert np.allclose(v.T, df.loc[df["x"].notna(), "T"].to_numpy(dtype=float))


def test_perm_var_single_value_in_analysis_sample_raises(base_df):
    df = base_df.copy()
    # Make Patsy drop all rows with T==1 by setting x=NA there.
    df.loc[df["T"] == 1, "x"] = np.nan
    with pytest.raises(ValueError, match="at least 2 distinct values in the analysis sample"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
        )


def test_cluster_requires_two_groups_in_analysis_sample(base_df):
    df = base_df.copy()
    # Two clusters globally…
    df["cluster"] = (df.index % 2 == 0).astype(int)
    # …but remove all odd rows via NA in x, leaving only one cluster in analysis
    df.loc[df.index % 2 == 1, "x"] = np.nan

    with pytest.raises(ValueError, match="at least 2 groups in the analysis sample"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            cluster="cluster",
        )


def test_cluster_codes_are_dense_after_subsetting(base_df):
    df = base_df.copy()
    # Use non-dense labels {10, 20}; ensure both survive subsetting
    df["cluster"] = np.where(df.index % 2 == 0, 10, 20)
    # Drop a few rows unrelated to cluster labels
    df.loc[df.index[:5], "x"] = np.nan

    v = validate_inputs(
        df,
        formula="y ~ x + T",
        stat="T",
        permute_var="T",
        cluster="cluster",
    )
    assert v.cluster is not None
    # Codes should be {0,1} not {10,20}
    assert set(np.unique(v.cluster)).issubset({0, 1})
    assert len(np.unique(v.cluster)) == 2


# ------------------------------------------------------------------ #
# 3) Failure paths (guards)
# ------------------------------------------------------------------ #
def test_both_formula_and_stat_fn_raise(base_df):
    with pytest.raises(ValueError, match="either `formula`"):
        validate_inputs(
            base_df,
            formula="y ~ x",
            stat="x",
            stat_fn=lambda d: 0.0,
            permute_var="T",
        )


def test_missing_permute_var_raises(base_df):
    with pytest.raises(ValueError, match="permute_var 'Z' not in dataframe"):
        validate_inputs(
            base_df,
            formula="y ~ x",
            stat="x",
            permute_var="Z",
        )


def test_perm_var_has_na_raises(base_df):
    df = base_df.copy()
    df.loc[0, "T"] = np.nan
    with pytest.raises(ValueError, match="contains missing"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
        )


def test_perm_var_needs_two_unique_values_raises(base_df):
    df = base_df.copy()
    df["T"] = 1  # constant
    with pytest.raises(ValueError, match="at least 2 distinct values"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
        )


def test_alternative_bad_string_raises(base_df):
    with pytest.raises(ValueError, match="alternative must be"):
        validate_inputs(
            base_df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            alternative="bad",
        )


@pytest.mark.parametrize("bad_alpha", [-1.0, 0.0, 1.0, 2.0])
def test_alpha_out_of_bounds_raises(base_df, bad_alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        validate_inputs(
            base_df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            alpha=bad_alpha,
        )


@pytest.mark.parametrize("bad_method", ["beta", "CP", "normal-ish"])
def test_ci_method_invalid_raises(base_df, bad_method):
    with pytest.raises(ValueError, match="ci_method must be"):
        validate_inputs(
            base_df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            ci_method=bad_method,
        )


@pytest.mark.parametrize(
    "kw",
    [
        dict(ci_range=-1.0),
        dict(ci_step=0.0),
        dict(ci_tol=-1e-6),
    ],
)
def test_ci_knobs_positive_raises(base_df, kw):
    with pytest.raises(ValueError):
        validate_inputs(
            base_df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            **kw,
        )


def test_weights_negative_raises(base_df):
    df = base_df.copy()
    df["w"] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            weights="w",
        )


def test_weights_non_numeric_raises(base_df):
    df = base_df.copy()
    df["w"] = "a"
    with pytest.raises(ValueError, match="weights must be numeric"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            weights="w",
        )


def test_weights_na_raises(base_df):
    df = base_df.copy()
    df["w"] = 1.0
    df.loc[3, "w"] = np.nan
    with pytest.raises(ValueError, match="weights column has missing"):
        validate_inputs(
            df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            weights="w",
        )


def test_formula_missing_stat_column_raises(base_df):
    with pytest.raises(ValueError, match="not found among RHS"):
        validate_inputs(
            base_df,
            formula="y ~ x + T",
            stat="Z",
            permute_var="T",
        )


# ------------------------------------------------------------------ #
# 4) Warning paths
# ------------------------------------------------------------------ #
def test_warning_when_permute_var_not_in_formula(base_df):
    # permute_var 'T' exists in DF but is not in the formula
    v = validate_inputs(
        base_df,
        formula="y ~ x",
        stat="x",
        permute_var="T",
    )
    assert any("does not appear in the formula" in w for w in v.warnings)


def test_stat_fn_slow_warning(base_df):
    def slow_fn(d: pd.DataFrame) -> float:
        time.sleep(1.15)
        return float(d["x"].mean())

    v = validate_inputs(
        base_df,
        stat_fn=slow_fn,
        permute_var="T",
    )
    assert any("may be slow" in w for w in v.warnings)


# ------------------------------------------------------------------ #
# 5) stat_fn branch: errors & metadata
# ------------------------------------------------------------------ #
def test_stat_fn_returns_non_numeric_raises(base_df):
    def bad_fn(_: pd.DataFrame):
        return "not a number"

    with pytest.raises(ValueError, match="numeric scalar"):
        validate_inputs(
            base_df,
            stat_fn=bad_fn,  # type: ignore[arg-type]
            permute_var="T",
        )


def test_stat_fn_raises_is_wrapped(base_df):
    def boom(_: pd.DataFrame) -> float:
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="raised an error"):
        validate_inputs(
            base_df,
            stat_fn=boom,  # type: ignore[arg-type]
            permute_var="T",
        )


def test_stat_fn_keeps_full_dataframe_shapes(base_df, rng):
    df = base_df.copy()
    df["cluster"] = rng.integers(0, 5, size=len(df))
    df["strata"] = rng.integers(0, 3, size=len(df))
    df["w"] = rng.uniform(0.3, 2.0, size=len(df))

    def stat_ok(d: pd.DataFrame) -> float:
        return float(d["y"].mean())

    v = validate_inputs(
        df,
        stat_fn=stat_ok,
        permute_var="T",
        cluster="cluster",
        strata="strata",
        weights="w",
    )
    # In stat_fn mode, arrays are taken from the full df (no Patsy subsetting)
    assert v.T.shape == (len(df),)
    assert v.cluster is not None and v.cluster.shape == (len(df),)
    assert v.strata is not None and v.strata.shape == (len(df),)
    assert v.weights is not None and v.weights.shape == (len(df),)
