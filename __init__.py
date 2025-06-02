"""Public interface for the ritest package."""

from .core import permutation_test
from .ci import compute_p_value_ci
from .config import DEFAULTS
import statsmodels.formula.api as smf


# PENDING: VALIDATE USER INPUTS
def ritest_set(overrides: dict):
    """
    Update global default settings.

    Args:
        overrides: dict of key-value pairs to override in DEFAULTS.
    """
    DEFAULTS.update(overrides)


def make_ols_stat_fn(formula: str, stat: str):
    """
    Returns a statistic function from a formula and coefficient name.

    Args:
        formula: A patsy-style formula.
        stat: The coefficient to extract from the fitted model.

    Returns:
        A function that takes a DataFrame and returns the coefficient.
    """
    def fn(df):
        model = smf.ols(formula, data=df).fit()
        if stat not in model.params.index:
            raise KeyError(f"{stat} not found in model coefficients.")
        return model.params[stat]
    return fn


def ritest(
    df,
    formula,
    stat,
    permute_var,
    alternative="two-sided",
    coef_ci=False
):
    """
    Run randomization inference on a linear model coefficient.

    Args:
        df: DataFrame with input data.
        formula: patsy-style model formula.
        stat: Name of coefficient to test.
        permute_var: Column to permute.
        alternative: 'two-sided', 'left', or 'right'.
        coef_ci: If True, will compute coefficient confidence interval.

    Returns:
        Dictionary with keys:
            - 'stat': observed coefficient
            - 'pval': permutation p-value
            - 'pval_ci': CI for p-value
            - 'pval_se': SE of p-value
            - 'null': null distribution
            - 'coef_ci': CI for coefficient (if computed)
    """
    config = DEFAULTS.copy()
    stat_fn = make_ols_stat_fn(formula, stat)

    result = permutation_test(
        df=df,
        stat_fn=stat_fn,
        permute_var=permute_var,
        alternative=alternative,
        reps=config["reps"],
        seed=config["seed"],
        n_jobs=config["n_jobs"]
    )

    # p-value CI
    count = int(result["pval"] * config["reps"])
    result["pval_ci"], result["pval_se"] = compute_p_value_ci(
        count, config["reps"], config["ci_method"], config["alpha"]
    )

    result["coef_ci"] = None

    if coef_ci:
        # coef CI logic will be added later
        pass

    return result

    
    return result