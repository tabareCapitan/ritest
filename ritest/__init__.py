"""Public interface for the ritest package."""

from typing import Callable, Optional, Union, Literal, cast
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

from .core import permutation_test
from .ci import compute_p_value_ci
from .config import DEFAULTS


def ritest_set(overrides: dict):
    """
    Update global default settings.

    Args:
        overrides: dict of key-value pairs to override in DEFAULTS.
    
    Raises:
        ValueError if any keys in overrides are not recognized config keys.
    """
    invalid_keys = [k for k in overrides if k not in DEFAULTS]
    if invalid_keys:
        raise ValueError(f"Invalid config key(s): {invalid_keys}. Allowed keys: {list(DEFAULTS.keys())}")
    DEFAULTS.update(overrides)


def make_ols_stat_fn(formula: str, stat: str):
    """
    Construct a statistic function that extracts a coefficient from an OLS model.

    Args:
        formula: A patsy-style model formula.
        stat: Name of the coefficient to extract.

    Returns:
        A function that takes a DataFrame and returns the target coefficient.
    """
    def fn(df):
        model = smf.ols(formula, data=df)
        result = model.fit()
        if stat not in result.params.index:
            raise KeyError(f"{stat} not found in model coefficients.")
        return result.params[stat]
    return fn


def ritest(
    df: pd.DataFrame,
    permute_var: str,
    *,
    formula: Optional[str] = None,
    stat: Optional[str] = None,
    stat_fn: Optional[Callable[[pd.DataFrame], float]] = None,
    cluster: Optional[Union[str, list]] = None,
    strata: Optional[Union[str, list]] = None,
    alternative: Literal["two-sided", "left", "right"] = "two-sided",
    coef_ci: bool = False
) -> dict:
    """
    Run randomization inference on a test statistic.

    Args:
        df: DataFrame with input data.
        permute_var: Column to permute during the test (typically treatment).
        formula: Model formula (e.g., 'y ~ treat'); required if not using stat_fn.
        stat: Name of coefficient in model to test; required if not using stat_fn.
        stat_fn: Optional function(df) -> float that computes the test statistic.
        cluster: Optional column or list of columns used to define clusters.
        strata: Optional column or list of columns to constrain permutation within.
        alternative: 'two-sided', 'left', or 'right'.
        coef_ci: Whether to compute a confidence interval for the coefficient (not yet implemented).

    Returns:
        A dictionary with:
            - 'stat': observed test statistic
            - 'pval': permutation p-value
            - 'pval_ci': confidence interval for p-value
            - 'pval_se': standard error of p-value
            - 'null': array of null test statistics
            - 'coef_ci': placeholder for coefficient CI (if implemented)
    """

    def validate_inputs():
        # Exclusive use of (formula + stat) OR stat_fn
        if (formula or stat) and stat_fn:
            raise ValueError("Provide either (formula and stat) OR stat_fn, not both.")
        if not stat_fn and not (formula and stat):
            raise ValueError("Must provide either stat_fn or both formula and stat.")
        if stat_fn and stat:
            raise ValueError("Do not provide `stat` when using `stat_fn`.")

        # permute_var must be numeric
        if not pd.api.types.is_numeric_dtype(df[permute_var]):
            raise ValueError(
                f"Column '{permute_var}' must be numeric (e.g., 0/1), not type {df[permute_var].dtype}."
            )

        # If using formula, validate both LHS and RHS variables
        if formula:
            try:
                ymat, xmat = dmatrices(formula, data=df, return_type="dataframe")
            except Exception as e:
                raise ValueError(f"Error parsing formula: {e}")

            # LHS validation
            yvar_name = ymat.columns[0]
            if not pd.api.types.is_numeric_dtype(ymat[yvar_name]):
                raise ValueError(f"The dependent variable '{yvar_name}' must be numeric, got {ymat[yvar_name].dtype}.")

            # RHS validation
            non_numeric_cols = [
                col for col in xmat.columns
                if not pd.api.types.is_numeric_dtype(xmat[col])
            ]
            if non_numeric_cols:
                raise ValueError(f"The following RHS variables are not numeric: {non_numeric_cols}")

    # Run validations first
    validate_inputs()

    config = DEFAULTS.copy()

    # Construct stat function
    if stat_fn:
        use_stat_fn = stat_fn
    else:
        use_stat_fn = make_ols_stat_fn(
            cast(str, formula),
            cast(str, stat)
        )

    # Run permutation test
    result = permutation_test(
        df=df,
        stat_fn=use_stat_fn,
        permute_var=permute_var,
        cluster=cluster,
        strata=strata,
        alternative=alternative,
        reps=config["reps"],
        seed=config["seed"],
        n_jobs=config["n_jobs"]
    )

    # Compute p-value CI
    count = int(result["pval"] * config["reps"])
    result["pval_ci"], result["pval_se"] = compute_p_value_ci(
        count, config["reps"], config["alpha"], config["ci_method"]
    )
    result["count"] = count
    result["reps"] = config["reps"]
    result["coef_ci"] = None  # Placeholder for future coefficient CI

    return result
