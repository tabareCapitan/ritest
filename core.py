"""Core randomization inference logic."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Callable, Literal


def permutation_test(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    permute_var: str,
    seed: int,
    n_jobs: int,
    reps: int,
    alternative: str = "two-sided"
) -> dict:
    """
    Compute a randomization-inference p-value by permuting a treatment variable.

    Args:
        df: Original dataset.
        stat_fn: A function that takes a DataFrame and returns a scalar test statistic.
        permute_var: Column in df to permute (typically a treatment indicator).
        seed: Random seed for reproducibility.
        n_jobs: Number of CPU cores to use (-1 = all available).
        reps: Number of permutations to generate.
        alternative: Direction of test ('two-sided', 'left', or 'right').

    Returns:
        A dictionary with:
            - 'stat': observed statistic
            - 'pval': permutation-based p-value
            - 'null': array of permuted statistics
            - 'alternative': test direction
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, int(1e9), size=reps)

    observed = stat_fn(df)

    def _one(seed_i):
        # Permute treatment column using a seed-specific permutation
        rs = np.random.RandomState(int(seed_i))
        df2 = df.copy()
        df2[permute_var] = rs.permutation(df2[permute_var].to_numpy())
        return stat_fn(df2)

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds))

    if alternative == "two-sided":
        pval = np.mean(np.abs(null) >= abs(observed))
    elif alternative == "left":
        pval = np.mean(np.array(null) <= observed)
    elif alternative == "right":
        pval = np.mean(np.array(null) >= observed)
    else:
        raise ValueError("invalid alternative")

    return {
        "stat": observed,
        "pval": pval,
        "null": np.array(null),
        "alternative": alternative,
    }
