"""
Comprehensive tests for ritest.validation.validate_inputs.

Run with:
    pytest -q -s tests/test_validation.py
"""

import time
import warnings
from typing import List

import numpy as np
import pandas as pd
import pytest

from ritest.validation import (
    ValidatedInputs,
    _binary_int8_from_series,  # type: ignore[attr-defined]
    validate_inputs,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def make_base_df(n: int = 200, seed: int = 123) -> pd.DataFrame:
    """Synthetic DGP with row-level treatment and auxiliary columns."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    T = rng.integers(0, 2, size=n)
    y = 0.7 * T + 0.3 * x1 - 0.2 * x2 + rng.normal(scale=1.0, size=n)

    # positive weights, cluster and strata labels
    w = np.exp(rng.normal(scale=0.2, size=n))
    cluster = (np.arange(n) // 10).astype(int)  # ~20 clusters
    strata = np.where(rng.random(n) < 0.5, "A", "B")

    return pd.DataFrame(
        {
            "y": y,
            "T": T,
            "x1": x1,
            "x2": x2,
            "w": w,
            "cluster": cluster,
            "strata": strata,
        }
    )


def diff_in_means(df: pd.DataFrame) -> float:
    """Generic stat function: E[y|T=1] - E[y|T=0]."""
    g = df.groupby("T")["y"].mean()
    return float(g.get(1, 0.0) - g.get(0, 0.0))


# ------------------------------------------------------------------ #
# 1) Happy-path: formula (linear) and stat_fn (generic)
# ------------------------------------------------------------------ #
def test_formula_path_basic_shapes_and_flags():
    df = make_base_df()

    v = validate_inputs(
        df,
        permute_var="T",
        formula="y ~ T + x1 + x2",
        stat="T",
        cluster="cluster",
        strata="strata",
        weights="w",
    )
    assert isinstance(v, ValidatedInputs)

    # shapes and dtypes
    assert v.y.ndim == 1 and v.y.shape[0] == df.shape[0]
    assert v.X.ndim == 2 and v.X.shape[0] == df.shape[0]
    assert v.T.ndim == 1 and v.T.shape[0] == df.shape[0]
    assert v.T.dtype == np.int8
    assert set(np.unique(v.T)).issubset({0, 1})

    # treatment index corresponds to "T" in RHS design matrix
    assert 0 <= v.treat_idx < v.X.shape[1]

    # design flags + metadata
    assert v.has_cluster is True
    assert v.has_strata is True
    assert v.weights is not None
    assert v.stat_fn is None
    assert isinstance(v.warnings, list)


def test_stat_fn_path_basic_shapes_and_metadata():
    df = make_base_df()

    v = validate_inputs(
        df,
        permute_var="T",
        stat_fn=diff_in_means,
        cluster="cluster",
        strata="strata",
        weights="w",
    )
    # y/X unused in stat_fn path
    assert v.y.shape == (0,)
    assert v.X.shape == (0, 0)
    # T is full length, int8 0/1
    assert v.T.shape[0] == df.shape[0]
    assert v.T.dtype == np.int8
    assert set(np.unique(v.T)).issubset({0, 1})

    # design info carried over
    assert v.cluster is not None
    assert v.strata is not None
    assert v.weights is not None
    assert v.has_cluster is True
    assert v.has_strata is True

    # stat_fn + warmup_time recorded
    assert v.stat_fn is diff_in_means
    assert isinstance(v.warmup_time, float)
    assert v.warmup_time >= 0.0


# ------------------------------------------------------------------ #
# 2) Argument-pattern validation
# ------------------------------------------------------------------ #
def test_cannot_supply_both_formula_and_stat_fn():
    df = make_base_df()
    with pytest.raises(
        ValueError, match="provide either `formula`.*`stat_fn`, not both"
    ):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            stat_fn=diff_in_means,
        )


def test_stat_required_in_formula_mode_and_forbidden_in_stat_fn_mode():
    df = make_base_df()

    # formula path: missing stat
    with pytest.raises(
        ValueError,
        match="`stat` \\(name of treatment column\\) required when using formula",
    ):
        validate_inputs(df, permute_var="T", formula="y ~ T")

    # stat_fn path: stat must be omitted
    with pytest.raises(
        ValueError, match="`stat` should not be supplied when using stat_fn"
    ):
        validate_inputs(df, permute_var="T", stat_fn=diff_in_means, stat="T")


@pytest.mark.parametrize("alt", ["two-sided", "left", "right"])
def test_alternative_valid_values_ok(alt):
    df = make_base_df()
    validate_inputs(df, permute_var="T", formula="y ~ T", stat="T", alternative=alt)


def test_alternative_invalid_raises():
    df = make_base_df()
    with pytest.raises(
        ValueError, match="alternative must be 'two-sided', 'left', or 'right'"
    ):
        validate_inputs(
            df, permute_var="T", formula="y ~ T", stat="T", alternative="twosided"
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_bounds_raises(alpha):
    df = make_base_df()
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        validate_inputs(df, permute_var="T", formula="y ~ T", stat="T", alpha=alpha)


@pytest.mark.parametrize("ci_method", ["cp", "normal"])
def test_ci_method_valid_values_ok(ci_method):
    df = make_base_df()
    validate_inputs(df, permute_var="T", formula="y ~ T", stat="T", ci_method=ci_method)


def test_ci_method_invalid_raises():
    df = make_base_df()
    with pytest.raises(
        ValueError,
        match=r"ci_method must be 'cp' \(Clopper–Pearson\) or 'normal' \(Wald with continuity correction\)",
    ):
        validate_inputs(
            df, permute_var="T", formula="y ~ T", stat="T", ci_method="beta"
        )


@pytest.mark.parametrize("ci_mode", ["none", "bounds", "grid"])
def test_ci_mode_valid_values_ok(ci_mode):
    df = make_base_df()
    validate_inputs(df, permute_var="T", formula="y ~ T", stat="T", ci_mode=ci_mode)


@pytest.mark.parametrize("ci_mode", ["BOUNDs", "band", "invalid"])
def test_ci_mode_invalid_raises(ci_mode):
    df = make_base_df()
    with pytest.raises(ValueError, match="ci_mode must be 'none', 'bounds', or 'grid'"):
        validate_inputs(df, permute_var="T", formula="y ~ T", stat="T", ci_mode=ci_mode)


@pytest.mark.parametrize(
    "key, bad",
    [
        ("ci_range", 0.0),
        ("ci_range", -1.0),
        ("ci_step", 0.0),
        ("ci_step", -0.01),
        ("ci_tol", 0.0),
        ("ci_tol", -1e-6),
    ],
)
def test_ci_numeric_positive_params_validation(key, bad):
    df = make_base_df()
    kwargs = {key: bad}
    with pytest.raises(ValueError, match=f"{key} must be > 0"):
        validate_inputs(df, permute_var="T", formula="y ~ T", stat="T", **kwargs)


# ------------------------------------------------------------------ #
# 3) permute_var column validation and recoding
# ------------------------------------------------------------------ #
def test_missing_permute_var_column_raises():
    df = make_base_df().drop(columns=["T"])
    with pytest.raises(ValueError, match="permute_var 'T' not in dataframe"):
        validate_inputs(df, permute_var="T", formula="y ~ 1", stat="T")


def test_permute_var_must_be_numeric_or_bool_and_non_missing():
    df = make_base_df()
    df["T"] = df["T"].astype(str)  # non-numeric
    with pytest.raises(ValueError, match="permute_var must be numeric/boolean"):
        validate_inputs(df, permute_var="T", formula="y ~ 1", stat="T")

    df = make_base_df()
    df.loc[0, "T"] = np.nan
    with pytest.raises(ValueError, match="permute_var contains missing values"):
        validate_inputs(df, permute_var="T", formula="y ~ 1", stat="T")


def test_permute_var_must_be_binary():
    df = make_base_df()
    df["T"] = np.arange(df.shape[0])  # many distinct values
    with pytest.raises(ValueError, match="permute_var must be binary"):
        validate_inputs(df, permute_var="T", formula="y ~ 1", stat="T")


def test_binary_int8_from_series_recodes_and_warns_for_non_01():
    s = pd.Series([-1, -1, 3, 3])
    warnings_out: List[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        arr, recoded = _binary_int8_from_series(s, name="T", warnings_out=warnings_out)

    assert arr.dtype == np.int8
    assert set(np.unique(arr)) == {0, 1}
    assert recoded is True

    # One explicit warning + captured in warnings_out
    assert len(warnings_out) == 1
    msg = warnings_out[0]
    assert "recoding to {0,1} with 1 mapped to the greater value" in msg
    assert any("recoding to {0,1}" in str(item.message) for item in w)


def test_binary_int8_from_series_zero_one_no_warning():
    s = pd.Series([0, 0, 1, 1])
    warnings_out: List[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        arr, recoded = _binary_int8_from_series(s, name="T", warnings_out=warnings_out)

    assert set(np.unique(arr)) == {0, 1}
    assert recoded is False
    assert warnings_out == []
    assert len(w) == 0


# ------------------------------------------------------------------ #
# 4) cluster, strata, and weights validation
# ------------------------------------------------------------------ #
def test_cluster_strata_and_weights_basic_rules():
    df = make_base_df()
    v = validate_inputs(
        df,
        permute_var="T",
        formula="y ~ T + x1 + x2",
        stat="T",
        cluster="cluster",
        strata="strata",
        weights="w",
    )
    assert v.cluster is not None
    assert v.strata is not None
    assert v.weights is not None
    assert v.has_cluster is True
    assert v.has_strata is True


def test_missing_cluster_column_and_missing_values_raises():
    df = make_base_df().drop(columns=["cluster"])
    with pytest.raises(ValueError, match="column 'cluster' not in dataframe"):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            cluster="cluster",
        )

    df = make_base_df()
    df.loc[0, "cluster"] = np.nan
    with pytest.raises(ValueError, match="column 'cluster' contains missing values"):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            cluster="cluster",
        )


def test_cluster_must_have_at_least_two_groups_global_and_subset():
    df = make_base_df()
    df["cluster"] = 1  # single group
    with pytest.raises(ValueError, match="cluster must have at least 2 groups"):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            cluster="cluster",
        )

    # Now create a situation where Patsy drops a whole cluster, leaving 1 cluster
    df = make_base_df()
    # two clusters overall (so global check passes)
    df["cluster"] = np.where(df.index < df.shape[0] // 2, 0, 1)
    # Drop all rows for cluster==1 from the analysis sample
    df.loc[df["cluster"] == 1, "y"] = np.nan

    with pytest.raises(
        ValueError, match="cluster must have at least 2 groups in the analysis sample"
    ):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T + x1",
            stat="T",
            cluster="cluster",
        )


def test_missing_strata_column_and_missing_values_raises():
    df = make_base_df().drop(columns=["strata"])
    with pytest.raises(ValueError, match="column 'strata' not in dataframe"):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            strata="strata",
        )

    df = make_base_df()
    df.loc[0, "strata"] = np.nan
    with pytest.raises(ValueError, match="column 'strata' contains missing values"):
        validate_inputs(
            df,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            strata="strata",
        )


def test_weights_column_rules():
    df = make_base_df()

    # Missing column
    df_missing = df.drop(columns=["w"])
    with pytest.raises(ValueError, match="weights column 'w' not in dataframe"):
        validate_inputs(
            df_missing,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            weights="w",
        )

    # Non-numeric
    df_bad_type = df.copy()
    df_bad_type["w"] = "a"
    with pytest.raises(ValueError, match="weights must be numeric"):
        validate_inputs(
            df_bad_type,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            weights="w",
        )

    # Missing values
    df_na = df.copy()
    df_na.loc[0, "w"] = np.nan
    with pytest.raises(ValueError, match="weights column has missing values"):
        validate_inputs(
            df_na,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            weights="w",
        )

    # Non-positive
    df_nonpos = df.copy()
    df_nonpos.loc[0, "w"] = 0.0
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        validate_inputs(
            df_nonpos,
            permute_var="T",
            formula="y ~ T",
            stat="T",
            weights="w",
        )


# ------------------------------------------------------------------ #
# 5) Formula-specific behaviour
# ------------------------------------------------------------------ #
def test_invalid_formula_message():
    df = make_base_df()
    with pytest.raises(ValueError, match="Invalid formula:"):
        validate_inputs(df, permute_var="T", formula="y ~ ~ T", stat="T")


def test_stat_must_be_rhs_term_in_formula():
    df = make_base_df()
    with pytest.raises(
        ValueError, match="stat 'not_in_rhs' not found among RHS terms of formula"
    ):
        validate_inputs(df, permute_var="T", formula="y ~ T + x1", stat="not_in_rhs")


def test_perm_var_not_in_formula_triggers_warning_message():
    df = make_base_df()
    v = validate_inputs(df, permute_var="T", formula="y ~ x1 + x2", stat="x1")
    msgs = "\n".join(v.warnings)
    assert "permute_var 'T' does not appear in the formula" in msgs


# ------------------------------------------------------------------ #
# 6) stat_fn-specific behaviour
# ------------------------------------------------------------------ #
def test_stat_fn_error_is_wrapped_with_clear_message():
    df = make_base_df()

    def bad_stat(df_: pd.DataFrame) -> float:  # noqa: ARG001
        raise RuntimeError("boom")

    with pytest.raises(
        ValueError, match="stat_fn raised an error on original data: boom"
    ):
        validate_inputs(df, permute_var="T", stat_fn=bad_stat)


def test_stat_fn_must_return_numeric_scalar():
    df = make_base_df()

    def non_numeric(df_: pd.DataFrame):
        return {"x": 1}

    with pytest.raises(ValueError, match="stat_fn must return a numeric scalar"):
        validate_inputs(
            df,
            permute_var="T",
            stat_fn=non_numeric,  # type: ignore[arg-type]
        )


def test_stat_fn_runtime_warning_when_slow(monkeypatch):
    df = make_base_df()

    # Fake a long runtime by monkeypatching perf_counter
    calls: List[float] = []

    def fake_perf_counter() -> float:
        calls.append(time.time())
        # Make successive calls differ by 2 seconds so dt > 1.0 for sure
        return float(len(calls) * 2)

    monkeypatch.setattr("ritest.validation.time.perf_counter", fake_perf_counter)

    v = validate_inputs(df, permute_var="T", stat_fn=diff_in_means)

    # Warmup time should be positive and we should have recorded a warning
    assert v.warmup_time > 0.0
    msgs = v.warnings
    assert any(
        ("stat_fn took" in m) and ("permutation test may be slow" in m) for m in msgs
    )


def test_generic_ci_warning_added_when_ci_mode_and_coef_ci_generic_false():
    df = make_base_df()
    v = validate_inputs(
        df,
        permute_var="T",
        stat_fn=diff_in_means,
        ci_mode="grid",
        coef_ci_generic=False,
    )
    msgs = "\n".join(v.warnings)
    assert "coef_ci_generic=False with stat_fn" in msgs
