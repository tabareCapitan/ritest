"""
ritest.validation
=================

Centralised input validation for the public ``ritest()`` API.

The *only* job of this module is to:

1. Check that the user supplied **consistent, supported** arguments.
2. Refuse anything unexpected or ambiguous with clear `ValueError`s.
3. Convert pandas objects to NumPy arrays of the *exact* dtypes required
   by the computation engines (FastOLS, shuffle.py, etc.).

No mutation of the original DataFrame is done; recoding (e.g. factorising
cluster/strata) happens on *copies* that are returned downstream.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import patsy

from .config import _canonical_ci_method


# --------------------------------------------------------------------- #
# Public return container
# --------------------------------------------------------------------- #
@dataclass
class ValidatedInputs:
    """Everything downstream modules need, NA-free and correctly typed."""

    y: np.ndarray
    X: np.ndarray
    T: np.ndarray
    treat_idx: int
    permute_var: str

    cluster: Optional[np.ndarray] = None
    strata: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    # user-defined statistic
    stat_fn: Optional[Callable[[pd.DataFrame], float]] = None

    alternative: str = "two-sided"
    alpha: float = 0.05
    ci_method: str = "clopper-pearson"
    ci_mode: str = "bounds"
    ci_range: float = 3.0
    ci_step: float = 0.005
    ci_tol: float = 1e-4

    # misc metadata for run.py / summary
    has_cluster: bool = False
    has_strata: bool = False
    warmup_time: float = 0.0  # time to run stat_fn once (if any)
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _require(cond: object, msg: str) -> None:
    """Raise ValueError with message if `cond` is falsy (bool-cast)."""
    if not bool(cond):
        raise ValueError(f"Validation error: {msg}")


def _is_numeric_series(s: pd.Series) -> bool:
    # Accept ints, floats, and booleans (0/1); everything else is rejected.
    t = s.dtype
    return (
        pd.api.types.is_integer_dtype(t)
        or pd.api.types.is_float_dtype(t)
        or pd.api.types.is_bool_dtype(t)
    )


def _factorize_series_no_na(s: pd.Series) -> np.ndarray:
    """Factorize a no-NA Series to dense int64 codes, sorted for stability."""
    codes, _ = pd.factorize(s, sort=True)
    return codes.astype(np.int64)


def _binary_int8_from_series(
    s: pd.Series, *, name: str, warnings_out: List[str]
) -> Tuple[np.ndarray, bool]:
    """
    Coerce a numeric/boolean Series with *exactly two* distinct values into
    a dense int8 array of 0/1 labels, mapping the *greater* value to 1.
    Returns (arr, recoded_flag).
    """
    # No NA allowed here; callers ensure that already.
    vals = s.to_numpy(copy=False)
    # Compute exact distinct values present (stable, sorted)
    uniq = np.unique(vals)
    if uniq.size != 2:
        raise ValueError(
            f"permute_var '{name}' must be binary (exactly 2 distinct values), "
            f"found {uniq.size} distinct values"
        )

    lo, hi = uniq[0], uniq[1]

    # If exactly {0,1} (int or float), we avoid a user warning; still return int8 for compactness.
    is_exact_01 = (lo == 0 and hi == 1) or (
        lo == 0.0 and hi == 1.0
    )  # ints/bools  # floats

    # Map the *greater* of the two values to 1, the lesser to 0
    out = (vals == hi).astype(np.int8, copy=False)

    # Warn if not already {0,1} (including booleans or unusual coding like {-1,1}, {2,7}, etc.)
    if not is_exact_01:
        msg = (
            f"permute_var '{name}' has non-{{0,1}} values [{lo!r}, {hi!r}]; "
            "recoding to {0,1} with 1 mapped to the greater value."
        )
        warnings_out.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return out, (not is_exact_01)


# --------------------------------------------------------------------- #
# Main public validator
# --------------------------------------------------------------------- #
def validate_inputs(
    df: pd.DataFrame,
    *,
    permute_var: str,
    # Linear-model path
    formula: str | None = None,
    stat: str | None = None,
    # User-supplied statistic path
    stat_fn: Callable[[pd.DataFrame], float] | None = None,
    # Optional columns
    cluster: str | None = None,
    strata: str | None = None,
    weights: str | None = None,
    # Test direction & config knobs (already defaulted upstream)
    alternative: str = "two-sided",
    alpha: float = 0.05,
    ci_method: str = "clopper-pearson",
    ci_mode: str = "bounds",
    ci_range: float = 3.0,
    ci_step: float = 0.005,
    ci_tol: float = 1e-4,
) -> ValidatedInputs:
    """
    Validate *all* user arguments and return clean NumPy arrays.

    Raises
    ------
    ValueError
        If any rule is violated.
    """

    # ------------------------------------------------------------------ #
    # 0. basic argument structure
    # ------------------------------------------------------------------ #
    has_formula = formula is not None
    has_stat_fn = stat_fn is not None
    _require(
        has_formula ^ has_stat_fn,
        "provide either `formula` (+ `stat`) **or** `stat_fn`, not both",
    )

    if has_formula:
        _require(
            stat is not None,
            "`stat` (name of treatment column) required when using formula",
        )
    else:
        _require(stat is None, "`stat` should not be supplied when using stat_fn")

    _require(
        alternative in {"two-sided", "left", "right"},
        "alternative must be 'two-sided', 'left', or 'right'",
    )

    # CI parameters used by p-value CIs and coefficient CIs
    _require(0.0 < float(alpha) < 1.0, "alpha must be in (0, 1)")
    try:
        ci_method_canon = _canonical_ci_method(ci_method)
    except ValueError as exc:
        raise ValueError(f"Validation error: {exc}")
    _require(
        ci_mode in {"none", "bounds", "grid"},
        "ci_mode must be 'none', 'bounds', or 'grid'",
    )
    _require(ci_range > 0.0, "ci_range must be > 0")
    _require(ci_step > 0.0, "ci_step must be > 0")
    _require(ci_tol > 0.0, "ci_tol must be > 0")

    # ------------------------------------------------------------------ #
    # 1. verify permute_var column (global)
    # ------------------------------------------------------------------ #
    _require(permute_var in df.columns, f"permute_var '{permute_var}' not in dataframe")

    pseries = cast(pd.Series, df[permute_var])
    _require(_is_numeric_series(pseries), "permute_var must be numeric/boolean")
    _require(~pseries.isna().any(), "permute_var contains missing values")

    # HARD RULE: treatment must be binary.
    _require(
        pseries.nunique(dropna=True) == 2,
        "permute_var must be binary (exactly 2 distinct values)",
    )

    # ------------------------------------------------------------------ #
    # 2. preflight: columns existence + no NA (global)
    #    (Codes/arrays will be *rebuilt on subset* for the formula path.)
    # ------------------------------------------------------------------ #
    warnings_list: List[str] = []

    # cluster / strata
    cluster_codes_full: Optional[np.ndarray] = None
    strata_codes_full: Optional[np.ndarray] = None
    if cluster is not None:
        _require(cluster in df.columns, f"column '{cluster}' not in dataframe")
        cser = cast(pd.Series, df[cluster])
        _require(~cser.isna().any(), f"column '{cluster}' contains missing values")
        cluster_codes_full = _factorize_series_no_na(cser)
        _require(
            np.unique(cluster_codes_full).size >= 2,
            "cluster must have at least 2 groups",
        )

    if strata is not None:
        _require(strata in df.columns, f"column '{strata}' not in dataframe")
        sser = cast(pd.Series, df[strata])
        _require(~sser.isna().any(), f"column '{strata}' contains missing values")
        strata_codes_full = _factorize_series_no_na(sser)

    # weights
    weights_full: Optional[np.ndarray] = None
    if weights is not None:
        _require(weights in df.columns, f"weights column '{weights}' not in dataframe")
        wser = cast(pd.Series, df[weights])
        _require(_is_numeric_series(wser), "weights must be numeric")
        _require(~wser.isna().any(), "weights column has missing values")
        _require((wser > 0).all(), "weights must be strictly positive")
        weights_full = np.asarray(wser, dtype=float)

    # Optional UX hint: permute_var token in formula
    if has_formula and formula is not None:
        token_pat = re.compile(rf"\b{re.escape(permute_var)}\b")
        if token_pat.search(formula) is None:
            warnings_list.append(
                f"permute_var '{permute_var}' does not appear in the formula; "
                "you will permute a column not explicitly used in the model."
            )

    # Heads-up: coefficient CIs are unavailable for stat_fn path
    if has_stat_fn and ci_mode != "none":
        warnings_list.append(
            "Coefficient CIs are only available for the linear formula/stat path; "
            "ci_mode will be ignored when using stat_fn."
        )

    # ------------------------------------------------------------------ #
    # 3. choose linear-model vs stat_fn path
    # ------------------------------------------------------------------ #
    if has_formula:
        # ---------- 3a. build design matrices (Patsy drops NA rows) --- #
        try:
            dmats = getattr(patsy, "dmatrices")
            y_mat, X_mat = dmats(formula, data=df, return_type="dataframe")
        except Exception as e:
            # Catch broad here to normalize Patsy errors to ValueError with context.
            raise ValueError(f"Invalid formula: {e}")

        # Subset index actually used by Patsy
        idx = X_mat.index

        # Convert y/X
        y = np.asarray(y_mat.iloc[:, 0], dtype=float)  # (n,)
        X = np.asarray(X_mat, dtype=float)  # (n,k)

        # Subset *everything else* to the same rows to keep alignment
        T_ser = cast(pd.Series, df.loc[idx, permute_var])

        # Re-check basic invariants on the subset
        _require(~np.isnan(y).any(), "outcome contains NA")
        _require(~np.isnan(X).any(), "design matrix contains NA")
        _require(
            ~T_ser.isna().any(), "permute_var contains NA after subsetting by formula"
        )
        _require(
            T_ser.nunique(dropna=True) == 2,
            "permute_var must be binary (exactly 2 distinct values) in the analysis sample",
        )

        # Coerce to binary int8 0/1, with 1 mapped to the greater value
        T_sub, recoded = _binary_int8_from_series(
            T_ser, name=permute_var, warnings_out=warnings_list
        )

        # Redo factorization on the subset for clean dense codes
        cluster_codes = (
            _factorize_series_no_na(cast(pd.Series, df.loc[idx, cluster]))
            if cluster is not None
            else None
        )
        strata_codes = (
            _factorize_series_no_na(cast(pd.Series, df.loc[idx, strata]))
            if strata is not None
            else None
        )
        weights_arr = (
            np.asarray(cast(pd.Series, df.loc[idx, weights]), dtype=float)
            if weights is not None
            else None
        )

        if cluster_codes is not None:
            _require(
                np.unique(cluster_codes).size >= 2,
                "cluster must have at least 2 groups in the analysis sample",
            )

        # treatment idx (stat must be one of the RHS column names)
        _require(
            stat in X_mat.columns, f"stat '{stat}' not found among RHS terms of formula"
        )
        treat_idx = list(X_mat.columns).index(stat)

        return ValidatedInputs(
            y=y,
            X=X,
            T=T_sub,  # int8 0/1 for memory-compact permutations
            treat_idx=treat_idx,
            permute_var=permute_var,
            cluster=cluster_codes,
            strata=strata_codes,
            weights=weights_arr,
            stat_fn=None,
            alternative=alternative,
            alpha=alpha,
            ci_method=ci_method_canon,
            ci_mode=ci_mode,
            ci_range=ci_range,
            ci_step=ci_step,
            ci_tol=ci_tol,
            has_cluster=cluster_codes is not None,
            has_strata=strata_codes is not None,
            warmup_time=0.0,
            warnings=warnings_list,
        )

    # ---------- 3b. stat_fn path ------------------------------------- #
    else:
        # warm-up call to ensure scalar return + measure runtime
        t0 = time.perf_counter()
        try:
            stat0 = stat_fn(df)  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(f"stat_fn raised an error on original data: {e}")
        dt = time.perf_counter() - t0

        _require(
            isinstance(stat0, (int, float, np.floating)),
            "stat_fn must return a numeric scalar",
        )

        if dt > 1.0:
            warnings_list.append(
                f"stat_fn took {dt:.2f}s; permutation test may be slow."
            )

        # Coerce full-sample permute_var to binary int8
        T_ser = cast(pd.Series, df[permute_var])
        _require(~T_ser.isna().any(), "permute_var contains missing values")
        _require(
            T_ser.nunique(dropna=True) == 2,
            "permute_var must be binary (exactly 2 distinct values)",
        )
        T_full, recoded = _binary_int8_from_series(
            T_ser, name=permute_var, warnings_out=warnings_list
        )

        return ValidatedInputs(
            y=np.empty(0),  # unused in stat_fn mode
            X=np.empty((0, 0)),
            T=T_full,  # int8 0/1 for compact permutations
            treat_idx=-1,
            permute_var=permute_var,
            cluster=cluster_codes_full,
            strata=strata_codes_full,
            weights=weights_full,
            stat_fn=stat_fn,
            alternative=alternative,
            alpha=alpha,
            ci_method=ci_method,
            ci_mode=ci_mode,
            ci_range=ci_range,
            ci_step=ci_step,
            ci_tol=ci_tol,
            has_cluster=cluster_codes_full is not None,
            has_strata=strata_codes_full is not None,
            warmup_time=dt,
            warnings=warnings_list,
        )
