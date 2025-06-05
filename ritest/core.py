"""Core randomization inference logic."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Callable, Literal, Union
from numpy.typing import NDArray


def permutation_test(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    permute_var: str,
    seed: int,
    n_jobs: int,
    reps: int,
    alternative: Literal["two-sided", "left", "right"] = "two-sided",
    strata: Union[str, list, None] = None,
    cluster: Union[str, list, None] = None
) -> dict:
    """
    Compute a permutation-based p-value by permuting a treatment variable.

    Args:
        df: Original dataset.
        stat_fn: Function that computes the test statistic from a DataFrame.
        permute_var: Column to permute (typically a treatment indicator).
        seed: Random seed for reproducibility.
        n_jobs: Number of CPU cores to use (-1 = all available).
        reps: Number of permutations.
        alternative: Direction of test: 'two-sided', 'left', or 'right'.
        strata: Optional column or list of columns to permute within strata.
        cluster: Optional column or list of columns to permute at the group level.

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
        rs = np.random.RandomState(int(seed_i))
        df2 = df.copy()

        # Helper: ensure argument is a list
        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            return list(x)

        cluster_cols = _to_list(cluster)
        strata_cols = _to_list(strata)

        if cluster_cols:
            # Cluster permutation: permute clusters as a unit (possibly within strata)
            if strata_cols:
                # Both strata and cluster: permute clusters within each stratum
                grouped = df.groupby(strata_cols)
                for stratum_keys, stratum_df in grouped:
                    stratum_idx = stratum_df.index
                    clusters_in_stratum = stratum_df[cluster_cols].drop_duplicates()
                    cluster_labels = clusters_in_stratum.apply(lambda row: tuple(row), axis=1)
                    permuted_labels = rs.permutation(cluster_labels)
                    cluster_map = dict(zip(cluster_labels, permuted_labels))
                    for clus_row, orig_label in zip(clusters_in_stratum.values, cluster_labels):
                        mask = np.ones(len(df2), dtype=bool)
                        for col, val in zip(strata_cols, stratum_keys if isinstance(stratum_keys, tuple) else (stratum_keys,)):
                            mask &= (df2[col] == val)
                        for c_col, c_val in zip(cluster_cols, clus_row):
                            mask &= (df2[c_col] == c_val)
                        # donor mask for permuted cluster in same stratum
                        donor_mask = np.ones(len(df), dtype=bool)
                        for col, val in zip(strata_cols, stratum_keys if isinstance(stratum_keys, tuple) else (stratum_keys,)):
                            donor_mask &= (df[col] == val)
                        permuted_cluster_vals = cluster_map[orig_label]
                        for c_col, c_val in zip(cluster_cols, permuted_cluster_vals):
                            donor_mask &= (df[c_col] == c_val)
                        donor_series = df.loc[donor_mask, permute_var]
                        donor_val = donor_series.iloc[0] if not donor_series.empty else np.nan
                        df2.loc[mask, permute_var] = donor_val
            else:
                # No strata, permute all clusters globally
                clusters = df[cluster_cols].drop_duplicates()
                cluster_labels = clusters.apply(lambda row: tuple(row), axis=1)
                permuted_labels = rs.permutation(cluster_labels)
                cluster_map = dict(zip(cluster_labels, permuted_labels))
                for clus_row, orig_label in zip(clusters.values, cluster_labels):
                    mask = np.ones(len(df2), dtype=bool)
                    for c_col, c_val in zip(cluster_cols, clus_row):
                        mask &= (df2[c_col] == c_val)
                    # donor mask for permuted cluster
                    donor_mask = np.ones(len(df), dtype=bool)
                    for c_col, c_val in zip(cluster_cols, cluster_map[orig_label]):
                        donor_mask &= (df[c_col] == c_val)
                    donor_series = df.loc[donor_mask, permute_var]
                    donor_val = donor_series.iloc[0] if not donor_series.empty else np.nan
                    df2.loc[mask, permute_var] = donor_val
        elif strata_cols:
            # Only strata: permute within strata at the row level
            df2[permute_var] = df.groupby(strata_cols)[permute_var].transform(
                lambda x: rs.permutation(x.to_numpy())
            )
        else:
            # No cluster, no strata: permute entire column
            df2[permute_var] = rs.permutation(df[permute_var].to_numpy())

        return stat_fn(df2)

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds))

    if alternative == "two-sided":
        pval = np.mean(np.abs(null) >= abs(observed))
    elif alternative == "left":
        pval = np.mean(null <= observed)
    elif alternative == "right":
        pval = np.mean(null >= observed)
    else:
        raise ValueError("invalid alternative: must be 'two-sided', 'left', or 'right'")

    return {
        "stat": observed,
        "pval": pval,
        "null": null,
        "alternative": alternative,
    }
