# tests/test_validation.py
"""
Comprehensive unit tests for ritest.validation.validate_inputs.

Run with:
    pytest -q
"""

import time

import numpy as np
import pandas as pd
import pytest

from ritest.validation import validate_inputs


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def base_df() -> pd.DataFrame:
    """A minimal DataFrame with outcome, covariate, treatment."""
    n = 50
    return pd.DataFrame(
        {
            "y": np.random.randn(n),
            "x": np.random.randn(n),
            "T": np.random.binomial(1, 0.5, size=n),
        }
    )


# ------------------------------------------------------------------ #
# 1. Happy-path tests
# ------------------------------------------------------------------ #
def test_formula_minimal(base_df):
    """OLS + HC1, no cluster/strata."""
    v = validate_inputs(
        base_df,
        formula="y ~ x + T",
        stat="T",
        permute_var="T",
    )
    print("Happy path: formula – passed")
    assert v.y.shape == (50,)
    assert v.X.shape[1] == 3  # intercept + x + T
    assert v.treat_idx == 2
    assert v.has_cluster is False
    assert v.has_strata is False
    assert v.stat_fn is None


def test_formula_with_cluster_strata_weights(base_df):
    df = base_df.copy()
    df["cluster"] = np.repeat(np.arange(10), 5)
    df["strata"] = np.random.randint(0, 3, size=len(df))
    df["w"] = np.random.uniform(0.5, 2.0, size=len(df))

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
    print("Happy path: formula + cluster/strata/weights – passed")
    assert v.has_cluster and v.has_strata
    assert v.weights is not None
    # cluster & strata must be np.int64 dense codes
    assert v.cluster is not None and v.cluster.dtype == np.int64
    assert v.strata is not None and v.strata.dtype == np.int64

    # alternative string propagated
    assert v.alternative == "right"


def test_stat_fn_minimal(base_df):
    def mean_x(df):
        return df["x"].mean()

    v = validate_inputs(
        base_df,
        stat_fn=mean_x,
        permute_var="T",
    )
    print("Happy path: stat_fn – passed")
    assert v.stat_fn is mean_x
    assert v.T.shape == (50,)
    assert v.warmup_time >= 0.0
    assert v.y.size == 0  # unused arrays for stat_fn
    assert v.X.size == 0


# ------------------------------------------------------------------ #
# 2. Failure-path tests (each must raise ValueError)
# ------------------------------------------------------------------ #
def test_both_formula_and_stat_fn_raise(base_df):
    with pytest.raises(ValueError, match="either `formula`"):
        validate_inputs(
            base_df,
            formula="y ~ x",
            stat="x",
            stat_fn=lambda d: 0,
            permute_var="T",
        )


def test_missing_permute_var_raises(base_df):
    with pytest.raises(ValueError, match="permute_var"):
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


def test_alternative_bad_string_raises(base_df):
    with pytest.raises(ValueError, match="alternative"):
        validate_inputs(
            base_df,
            formula="y ~ x + T",
            stat="T",
            permute_var="T",
            alternative="bad",
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


def test_stat_fn_returns_non_numeric_raises(base_df):
    def bad_fn(df):
        return "not a number"

    with pytest.raises(ValueError, match="numeric scalar"):
        validate_inputs(
            base_df,
            stat_fn=bad_fn,  # type: ignore[arg-type]
            permute_var="T",
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
# 3. Edge-case & warning tests
# ------------------------------------------------------------------ #
def test_stat_fn_slow_warning(base_df):
    def slow_fn(df):
        time.sleep(1.2)  # >1-second sleep
        return df["x"].mean()

    v = validate_inputs(
        base_df,
        stat_fn=slow_fn,
        permute_var="T",
    )
    assert any("slow" in w for w in v.warnings)
    print("Edge path: slow stat_fn warning captured")
