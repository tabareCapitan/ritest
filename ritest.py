from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def fit_observed_model(df: pd.DataFrame, formula: str, stat: str):
    """
    Fit the OLS model and extract the observed coefficient.
    """
    model = smf.ols(formula, data=df).fit()
    if stat not in model.params.index:
        raise KeyError(f"Coefficient '{stat}' not found in model.")
    obs_coef = model.params[stat]
    return model, obs_coef


def permute_distribution(
    df: pd.DataFrame,
    formula: str,
    stat: str,
    permute_var: str,
    seeds: np.ndarray
) -> np.ndarray:
    """
    Generate null distribution of the statistic by permuting the assignment variable.
    """
    def _one(seed_i):
        rs = np.random.RandomState(int(seed_i))
        df2 = df.copy()
        df2[permute_var] = rs.permutation(df2[permute_var].to_numpy())
        mod = smf.ols(formula, data=df2).fit()
        return mod.params[stat]

    return np.array(Parallel(n_jobs=-1)(delayed(_one)(s) for s in seeds))


def compute_p_value(
    dist: np.ndarray,
    obs_coef: float,
    alternative: str
) -> float:
    """
    Compute permutation p-value under specified alternative.
    """
    if alternative == "two-sided":
        return float(np.mean(np.abs(dist) >= abs(obs_coef)))
    elif alternative == "left":
        return float(np.mean(dist <= obs_coef))
    else:
        return float(np.mean(dist >= obs_coef))


def compute_p_value_ci(
    count: int,
    reps: int,
    method: str = "cp",
    alpha: float = 0.05
) -> tuple:
    """
    Compute 95% confidence interval for p-value via Clopper-Pearson or normal approximation.
    """
    p = count / reps
    se = np.sqrt(p * (1 - p) / reps)
    if method == "normal":
        lo = max(0.0, p - 1.96 * se)
        hi = min(1.0, p + 1.96 * se)
    else:
        lo, hi = proportion_confint(count, reps, alpha=alpha, method='beta')
    return (lo, hi), se


def invert_confidence_interval(
    df: pd.DataFrame,
    formula: str,
    stat: str,
    permute_var: str,
    obs_coef: float,
    seeds: np.ndarray,
    alternative: str,
    ci_range: float,
    ci_step: float,
    alpha: float = 0.05
) -> tuple:
    """
    Invert the permutation test to obtain confidence interval for the coefficient.
    """
    # Compute robust SE for grid boundaries
    model = smf.ols(formula, data=df).fit()
    # robust_se = model.get_robustcov_results(cov_type='HC1').bse[stat] DEBUG
    robust = model.get_robustcov_results(cov_type='HC1')
    robust_se = dict(zip(model.params.index, robust.bse))[stat]
    grid = np.arange(
        obs_coef - ci_range * robust_se,
        obs_coef + ci_range * robust_se + ci_step,
        ci_step
    )
    profile = {}

    def p_for_beta(beta0):
        def _one_shift(seed_i):
            rs = np.random.RandomState(int(seed_i))
            df2 = df.copy()
            response = formula.split('~')[0].strip()
            # permute_var is categorical, which is correct but won't work here
            df2[response] = df2[response] - beta0 * df2[permute_var].astype(float) 
            df2[permute_var] = rs.permutation(df2[permute_var].to_numpy())
            mod = smf.ols(formula, data=df2).fit()
            coef = mod.params[stat]
            if alternative == "two-sided":
                return abs(coef) >= abs(beta0)
            elif alternative == "left":
                return coef <= beta0
            else:
                return coef >= beta0
        hits = list(Parallel(n_jobs=-1)(delayed(_one_shift)(s) for s in seeds))
        return np.mean(hits)

    for b in grid:
        profile[b] = p_for_beta(b)

    # Determine CI endpoints
    accepted = [b for b, p in profile.items() if p > alpha]
    if alternative == "two-sided":
        low = min(accepted) if accepted else np.nan
        high = max(accepted) if accepted else np.nan
    elif alternative == "left":
        low, high = -np.inf, max(accepted) if accepted else np.nan
    else:
        low, high = min(accepted) if accepted else np.nan, np.inf

    return (low, high), profile


def plot_profile(profile: dict, obs_coef: float, ci_bounds: tuple, alpha: float = 0.05) -> Figure:
    """
    Plot p-value profile versus hypothesized coefficient values.
    """
    low, high = ci_bounds
    fig, ax = plt.subplots()
    ax.plot(list(profile.keys()), list(profile.values()), label='p(β₀)')
    ax.axhline(alpha, linestyle='--', label='α')
    if np.isfinite(low):
        ax.axvline(low, linestyle=':', label='CI lower')
    if np.isfinite(high):
        ax.axvline(high, linestyle=':', label='CI upper')
    ax.axvline(obs_coef, color='r', label='Observed β')
    ax.set_xlabel('β₀ hypothesis')
    ax.set_ylabel('Permutation p-value')
    ax.legend()
    return fig


def ritest(
    df: pd.DataFrame,
    formula: str,
    stat: str,
    permute_var: str,
    reps: int = 1000,
    seed: int = 23,
    alternative: str = "two-sided",
    ci: Optional[str] = None,
    plot: bool = False,
    ci_range: float = 3.0,
    ci_step: float = 0.005,
    ci_method: str = "cp"
) -> dict:
    """
    Orchestrates randomization inference for a coefficient, returning p-values,
    confidence intervals, and optional profile plots.
    """
    # Validate inputs
    alt_opts = {"two-sided", "left", "right"}
    if alternative not in alt_opts:
        raise ValueError("alternative must be 'two-sided', 'left', or 'right'")
    if ci not in (None, "CI"):
        raise ValueError("ci must be None or 'CI'")
    if plot and ci != "CI":
        raise ValueError("plot=True only valid when ci='CI'")

    # Initialize RNG and seeds
    seed = int(seed)
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=reps)

    # Fit observed model and extract coefficient
    model, obs_coef = fit_observed_model(df, formula, stat)

    # Generate permutation distribution
    dist = permute_distribution(df, formula, stat, permute_var, seeds)

    # Compute p-value
    pval = compute_p_value(dist, obs_coef, alternative)
    count = int(pval * reps)

    # Compute CI on p-value
    (lo_p, hi_p), se_p = compute_p_value_ci(count, reps, ci_method)

    results = {
        'estimate': obs_coef,
        'pval': pval,
        'se_p': se_p,
        'ci_p': (lo_p, hi_p),
        'ci_coef': None,
        'ci_profile': None,
        'fig': None
    }

    # Optional coefficient CI inversion
    if ci == 'CI':
        ci_coef, profile = invert_confidence_interval(
            df, formula, stat, permute_var,
            obs_coef, seeds, alternative,
            ci_range, ci_step
        )
        results['ci_coef'] = ci_coef
        results['ci_profile'] = profile

        # Optional plotting
        if plot:
            fig = plot_profile(profile, obs_coef, ci_coef)
            results['fig'] = fig

    return results