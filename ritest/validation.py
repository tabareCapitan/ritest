"""
ritest.validation
=================

Centralised input validation for the public ``ritest()`` API.

The *only* job of this module is to:

1. Check that the user supplied **consistent, supported** arguments.
2. Refuse anything unexpected or ambiguous with clear `ValueError`s.
3. Convert pandas objects → NumPy arrays of the *exact* dtypes required
   by the computation engines (FastOLS, shuffle.py, etc.).

No mutation of the original DataFrame is done; recoding (e.g. factorising
cluster/strata) happens on *copies* that are returned downstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from patsy import PatsyError, dmatrices


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
    ci_method: str = "cp"
    ci_mode: str = "bounds"
    ci_range: float = 3.0
    ci_step: float = 0.005
    ci_tol: float = 1e-4
    coef_ci_generic: bool = False

    # misc metadata for run.py / summary
    has_cluster: bool = False
    has_strata: bool = False
    warmup_time: float = 0.0  # time to run stat_fn once (if any)
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _require(cond: bool, msg: str) -> None:
    """Tiny helper for readability."""
    if not cond:
        raise ValueError(f"Validation error: {msg}")


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)


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
    ci_method: str = "cp",
    ci_mode: str = "bounds",
    ci_range: float = 3.0,
    ci_step: float = 0.005,
    ci_tol: float = 1e-4,
    coef_ci_generic: bool = False,
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
        has_formula ^ has_stat_fn, "provide either `formula` (+ `stat`) **or** `stat_fn`, not both"
    )

    if has_formula:
        _require(stat is not None, "`stat` (name of treatment column) required when using formula")
    else:
        _require(stat is None, "`stat` should not be supplied when using stat_fn")

    _require(
        alternative in {"two-sided", "left", "right"},
        "alternative must be 'two-sided', 'left', or 'right'",
    )

    # ------------------------------------------------------------------ #
    # 1. verify permute_var column
    # ------------------------------------------------------------------ #
    _require(permute_var in df.columns, f"permute_var '{permute_var}' not in dataframe")

    pseries = df[permute_var]
    _require(_is_numeric_series(pseries), "permute_var must be numeric")
    _require(~pseries.isna().any(), "permute_var contains missing values")

    # ------------------------------------------------------------------ #
    # 2. handle cluster / strata (factorise to dense int codes)
    # ------------------------------------------------------------------ #
    warnings: list[str] = []

    def _factorize(name: Optional[str]) -> Optional[np.ndarray]:
        if name is None:
            return None
        _require(name in df.columns, f"column '{name}' not in dataframe")
        col = df[name]
        _require(~col.isna().any(), f"column '{name}' contains missing values")
        codes, _ = pd.factorize(col, sort=True)
        return codes.astype(np.int64)

    cluster_codes = _factorize(cluster)
    strata_codes = _factorize(strata)

    # ------------------------------------------------------------------ #
    # 3. weights
    # ------------------------------------------------------------------ #
    weights_arr: np.ndarray | None = None
    if weights is not None:
        _require(weights in df.columns, f"weights column '{weights}' not in dataframe")
        wser = df[weights]
        _require(_is_numeric_series(wser), "weights must be numeric")
        _require(~wser.isna().any(), "weights column has missing values")
        _require((wser > 0).all(), "weights must be strictly positive")
        weights_arr = wser.to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 4. choose linear-model vs stat_fn path
    # ------------------------------------------------------------------ #
    if has_formula:
        # ---------- 4a. build design matrices ------------------------- #
        try:
            y_mat, X_mat = dmatrices(formula, data=df, return_type="dataframe")
        except PatsyError as e:
            raise ValueError(f"Invalid formula: {e}")

        y = y_mat.iloc[:, 0].to_numpy(dtype=float)  # (n,)
        X = X_mat.to_numpy(dtype=float)  # (n,k)
        T = df[permute_var].to_numpy(dtype=float)

        # treatment idx
        _require(stat in X_mat.columns, f"stat '{stat}' not found among RHS terms of formula")
        treat_idx = list(X_mat.columns).index(stat)

        # no missing in y/X
        _require(~np.isnan(y).any(), "outcome contains NA")
        _require(~np.isnan(X).any(), "design matrix contains NA")

        vinputs = ValidatedInputs(
            y=y,
            X=X,
            T=T,
            treat_idx=treat_idx,
            permute_var=permute_var,
            cluster=cluster_codes,
            strata=strata_codes,
            weights=weights_arr,
            stat_fn=None,
            alternative=alternative,
            alpha=alpha,
            ci_method=ci_method,
            ci_mode=ci_mode,
            ci_range=ci_range,
            ci_step=ci_step,
            ci_tol=ci_tol,
            coef_ci_generic=coef_ci_generic,
            has_cluster=cluster_codes is not None,
            has_strata=strata_codes is not None,
            warmup_time=0.0,
            warnings=warnings,
        )
        return vinputs

    # ---------- 4b. stat_fn path ------------------------------------- #
    else:
        # warm-up call to ensure scalar return + measure runtime
        t0 = time.perf_counter()
        try:
            stat0 = stat_fn(df)  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(f"stat_fn raised an error on original data: {e}")
        dt = time.perf_counter() - t0

        _require(
            isinstance(stat0, (int, float, np.floating)), "stat_fn must return a numeric scalar"
        )

        if dt > 1.0:
            warnings.append(f"stat_fn took {dt:.2f}s; permutation test may be slow.")

        vinputs = ValidatedInputs(
            y=np.empty(0),  # unused in stat_fn mode
            X=np.empty((0, 0)),
            T=df[permute_var].to_numpy(dtype=float),
            treat_idx=-1,
            permute_var=permute_var,
            cluster=cluster_codes,
            strata=strata_codes,
            weights=weights_arr,
            stat_fn=stat_fn,
            alternative=alternative,
            alpha=alpha,
            ci_method=ci_method,
            ci_mode=ci_mode,
            ci_range=ci_range,
            ci_step=ci_step,
            ci_tol=ci_tol,
            coef_ci_generic=coef_ci_generic,
            has_cluster=cluster_codes is not None,
            has_strata=strata_codes is not None,
            warmup_time=dt,
            warnings=warnings,
        )
        return vinputs
