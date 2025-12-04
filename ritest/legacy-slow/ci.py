"""Compute confidence intervals for permutation p-values."""

import numpy as np


def compute_p_value_ci(count: int, reps: int, alpha: float, method: str = "cp") -> tuple:
    """
    Compute confidence interval for a permutation-based p-value.

    Args:
        count: Number of permutations at least as extreme as observed.
        reps: Total number of permutations.
        alpha: Confidence level (default is 0.05 → 95% CI).
        method: 'cp' = Clopper-Pearson, 'normal' = Wald approx.

    Returns:
        (lo, hi): confidence interval
        se: standard error of the p-value
    """
    p = count / reps
    se = np.sqrt(p * (1 - p) / reps)
    if method == "normal":
        from scipy.stats import norm

        z = norm.ppf(1 - alpha / 2)
        lo = max(0.0, p - z * se)
        hi = min(1.0, p + z * se)
    else:
        from statsmodels.stats.proportion import proportion_confint

        lo, hi = proportion_confint(count, reps, alpha=alpha, method="beta")

    return (lo, hi), se
