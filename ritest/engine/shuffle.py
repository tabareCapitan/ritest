"""
ritest.engine.shuffle
=====================

Fast permutation helpers with optional Numba acceleration.

Design
------
* Four flavours: plain, strata, cluster, cluster-within-strata.
* Validation & dtype checking happens upstream.
* If Numba is available, heavy loops are JIT-compiled; otherwise we
  fall back to vectorised NumPy versions. Either path returns the same result.

Notes on assumptions
--------------------
* Cluster modes assume `a` is cluster-constant (one value per cluster).
  This is asserted inside `permute_assignment` for safety.
* The stratified Numba path previously assumed strata were contiguous; this is
  FIXED by scattering via explicit indices, so strata may be non-contiguous.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from numba import njit, prange

    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

    def njit(*args, **kwargs):  # noqa: D401
        def wrapper(fn):
            return fn

        return wrapper

    prange = range  # Safe fallback


# ------------------------------------------------------------------ #
# Plain shuffle – NumPy only (fast enough, no loop to JIT)
# ------------------------------------------------------------------ #


def _permute_plain(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(a)


# ------------------------------------------------------------------ #
# Stratified shuffle
# ------------------------------------------------------------------ #


def _np_perm_strata(
    a: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = a.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = rng.permutation(out[idx])
    return out


if NUMBA_OK:

    @njit(parallel=True)
    def _nb_scatter_by_index(out, idx, vals):
        """
        Scatter `vals` into `out` at positions `idx`. Supports non-contiguous strata.

        Requirements:
        - `idx` must contain each target position at most once (unique indices).
        - `idx.size == vals.size`.
        """
        for i in prange(idx.size):
            out[idx[i]] = vals[i]


# ------------------------------------------------------------------ #
# Cluster shuffle (no strata)
# ------------------------------------------------------------------ #


def _np_perm_cluster(
    a: np.ndarray,
    cluster: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorised O(n + G) implementation using return_inverse.
    """
    clusters, first_idx, inv = np.unique(cluster, return_index=True, return_inverse=True)
    cluster_vals = a[first_idx]  # one value per cluster
    perm_vals = cluster_vals[rng.permutation(clusters.size)]
    out = perm_vals[inv]  # broadcast permuted cluster values to members
    return out


if NUMBA_OK:

    @njit(parallel=True)
    def _nb_broadcast_cluster(out, cluster, perm_map):
        for i in prange(cluster.size):
            out[i] = perm_map[cluster[i]]


# ------------------------------------------------------------------ #
# Cluster-within-strata shuffle  (NumPy implementation = reference)
# ------------------------------------------------------------------ #


def _np_perm_cluster_strata(
    a: np.ndarray,
    cluster: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorised O(n + sum_s G_s) implementation: per stratum, permute cluster-level
    values and broadcast via return_inverse.
    """
    out = np.empty_like(a)
    for s in np.unique(strata):
        mask = strata == s
        # Unique clusters within this stratum, and inverse map back to rows in the stratum
        clust_s = cluster[mask]
        clust_ids, first_idx, inv = np.unique(clust_s, return_index=True, return_inverse=True)
        vals = a[mask][first_idx]  # one representative value per (stratum, cluster)
        perm_vals = vals[rng.permutation(clust_ids.size)]
        out[mask] = perm_vals[inv]
    return out


# ------------------------------------------------------------------ #
# Small helpers (guards)
# ------------------------------------------------------------------ #


def _assert_cluster_constant(a: np.ndarray, cluster: np.ndarray) -> None:
    """
    Require that `a` is constant within each cluster.

    Vectorised check:
    - Map each row to its cluster's first value (via return_inverse),
      then compare to `a`.
    """
    clusters, first_idx, inv = np.unique(cluster, return_index=True, return_inverse=True)
    first_vals = a[first_idx]  # one representative per cluster
    expected = first_vals[inv]
    if not np.all(a == expected):
        raise ValueError(
            "permute_assignment: `a` must be constant within each cluster "
            "(cluster-wise permutation assumes a cluster-level value)."
        )


def _can_use_numba_cluster(cluster: np.ndarray) -> bool:
    """
    Decide if the Numba cluster path (perm_map of length max_label+1) is safe.

    Conditions:
    - integer dtype
    - non-negative labels
    - label span not wildly larger than number of unique clusters
      (avoid allocating huge sparse maps). Threshold: span <= 4 * n_unique.
    """
    if not np.issubdtype(cluster.dtype, np.integer):
        return False
    cmin = int(np.min(cluster))
    if cmin < 0:
        return False
    cmax = int(np.max(cluster))
    n_unique = np.unique(cluster).size
    span = cmax + 1  # since cmin >= 0
    return span <= 4 * n_unique


def _check_lengths(
    a: np.ndarray, cluster: Optional[np.ndarray], strata: Optional[np.ndarray]
) -> None:
    n = a.size
    if cluster is not None and cluster.size != n:
        raise ValueError("permute_assignment: `cluster` length must match `a`.")
    if strata is not None and strata.size != n:
        raise ValueError("permute_assignment: `strata` length must match `a`.")


# ------------------------------------------------------------------ #
# Dispatcher
# ------------------------------------------------------------------ #


def permute_assignment(
    a: np.ndarray,
    *,
    cluster: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Return a permuted copy of `a` according to the requested scheme.

    Modes:
    - Plain: global permutation of `a`.
    - Stratified: permute within each stratum (counts preserved per stratum).
      (Numba path supports non-contiguous strata via index scatter.)
    - Cluster: permute cluster-level values and broadcast to members.
      (Requires `a` to be cluster-constant.)
    - Cluster within strata: cluster-wise permutation separately inside each stratum.
    """
    if rng is None:
        rng = np.random.default_rng()

    _check_lengths(a, cluster, strata)

    # 1. Plain -------------------------------------------------------- #
    if cluster is None and strata is None:
        return _permute_plain(a, rng)

    # 2. Stratified only --------------------------------------------- #
    if cluster is None and strata is not None:
        if NUMBA_OK:
            # Build concatenated indices and permuted values per stratum.
            # This supports NON-CONTIGUOUS strata robustly.
            idx_list = []
            vals_list = []
            for s in np.unique(strata):
                idx = np.where(strata == s)[0]
                idx_list.append(idx)
                vals_list.append(rng.permutation(a[idx]))
            idx_all = np.concatenate(idx_list) if idx_list else np.empty(0, dtype=np.int64)
            vals_all = np.concatenate(vals_list) if vals_list else np.empty(0, dtype=a.dtype)
            out = np.empty_like(a)
            _nb_scatter_by_index(out, idx_all, vals_all)
            return out
        return _np_perm_strata(a, strata, rng)

    # 3. Cluster only ------------------------------------------------- #
    if cluster is not None and strata is None:
        # Consistency requirement for cluster modes
        _assert_cluster_constant(a, cluster)

        if NUMBA_OK and _can_use_numba_cluster(cluster):
            clusters, first_idx = np.unique(cluster, return_index=True)
            cluster_vals = a[first_idx]
            perm_vals = cluster_vals[rng.permutation(len(cluster_vals))]
            # Dense non-negative labels guaranteed by _can_use_numba_cluster
            perm_map = np.empty(int(cluster.max()) + 1, dtype=a.dtype)
            perm_map[clusters] = perm_vals
            out = np.empty_like(a)
            _nb_broadcast_cluster(out, cluster, perm_map)
            return out
        # Fallback (label-agnostic, now vectorised)
        return _np_perm_cluster(a, cluster, rng)

    # 4. Cluster *within* strata  ➜ always NumPy for correctness
    if cluster is not None and strata is not None:
        # Consistency requirement for cluster modes
        _assert_cluster_constant(a, cluster)
        # NOTE: A previous Numba impl had OOB risk with non-dense labels per stratum.
        # We keep the reference NumPy version for clarity and safety (now vectorised).
        return _np_perm_cluster_strata(a, cluster, strata, rng)

    # Should never reach here
    raise ValueError("Unhandled permutation mode.")


# ------------------------------------------------------------------ #
# Generate matrix of permuted assignments  (called from run.py) ---- #
# ------------------------------------------------------------------ #


def generate_permuted_matrix(
    a: np.ndarray,
    reps: int,
    *,
    cluster: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Return a (reps, n) matrix of independently permuted copies of `a`.

    Parameters
    ----------
    a : ndarray
        Treatment assignment vector to permute (length n).
    reps : int
        Number of permutations to generate.
    cluster : ndarray, optional
        Cluster labels for cluster-wise permutation.
    strata : ndarray, optional
        Stratum labels for stratified permutation.
    rng : np.random.Generator, optional
        Random number generator (reused across draws).

    Returns
    -------
    perms : (reps, n) ndarray
        Each row is an independently permuted version of `a`.

    Notes
    -----
    - If `cluster` or `strata` are passed, permutations are constrained accordingly.
    - A light uniqueness diagnostic is run for clustered modes only, using a
      compressed view at cluster (or stratum×cluster) representatives and optional
      downsampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    if reps < 0:
        raise ValueError("generate_permuted_matrix: `reps` must be non-negative.")

    perms = np.empty((reps, a.size), dtype=a.dtype)
    for r in range(reps):
        perms[r] = permute_assignment(
            a,
            cluster=cluster,
            strata=strata,
            rng=rng,
        )

    # Heuristic uniqueness check (only meaningful in clustered modes)
    if cluster is not None and reps >= 2:
        import warnings

        # Build a compact representation to check uniqueness cheaply:
        # - if stratified clustering: use first index of each (stratum, cluster) pair
        # - else: use first index of each cluster
        if strata is not None:
            pairs = np.stack((strata, cluster), axis=1)
            _, first_idx = np.unique(pairs, axis=0, return_index=True)
        else:
            _, first_idx = np.unique(cluster, return_index=True)

        rows = perms[:, first_idx]  # (reps, G') compact representation

        # Optional downsampling to cap cost for huge reps
        if reps > 5000:
            step = max(reps // 5000, 1)
            rows = rows[::step]

        n_unique = np.unique(rows, axis=0).shape[0]
        denom = rows.shape[0]
        ratio = n_unique / denom if denom else 1.0
        if ratio < 0.9:
            warnings.warn(
                f"Low uniqueness in cluster permutations: {n_unique}/{denom} "
                f"unique ({ratio:.1%}). Power may be limited."
            )

    return perms


__all__ = ["permute_assignment", "generate_permuted_matrix"]
