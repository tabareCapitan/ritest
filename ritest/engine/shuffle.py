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

    @njit
    def _nb_perm_strata(a, strata, perm_vals, slice_starts, slice_ends):
        out = np.empty_like(a)
        k = 0
        for i in range(slice_starts.size):
            lo, hi = slice_starts[i], slice_ends[i]
            out[lo:hi] = perm_vals[k : k + (hi - lo)]
            k += hi - lo
        return out


# ------------------------------------------------------------------ #
# Cluster shuffle (no strata)
# ------------------------------------------------------------------ #


def _np_perm_cluster(
    a: np.ndarray,
    cluster: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty_like(a)
    clusters, first_idx = np.unique(cluster, return_index=True)
    cluster_vals = a[first_idx]
    perm_vals = cluster_vals[rng.permutation(len(cluster_vals))]
    for c, v in zip(clusters, perm_vals):
        out[cluster == c] = v
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
    out = np.empty_like(a)
    for s in np.unique(strata):
        mask = strata == s
        clust_s, first_idx = np.unique(cluster[mask], return_index=True)
        vals = a[mask][first_idx]
        perm_vals = vals[rng.permutation(len(vals))]
        for c, v in zip(clust_s, perm_vals):
            out[(cluster == c) & mask] = v
    return out


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
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Plain -------------------------------------------------------- #
    if cluster is None and strata is None:
        return _permute_plain(a, rng)

    # 2. Stratified only --------------------------------------------- #
    if cluster is None and strata is not None:
        if NUMBA_OK:
            perm_vals, slice_starts, slice_ends = [], [], []
            for s in np.unique(strata):
                idx = np.where(strata == s)[0]
                slice_starts.append(idx[0])
                slice_ends.append(idx[-1] + 1)
                perm_vals.extend(rng.permutation(a[idx]))
            return _nb_perm_strata(
                a,
                strata,
                np.asarray(perm_vals),
                np.asarray(slice_starts),
                np.asarray(slice_ends),
            )
        return _np_perm_strata(a, strata, rng)

    # 3. Cluster only ------------------------------------------------- #
    if cluster is not None and strata is None:
        if NUMBA_OK:
            clusters, first_idx = np.unique(cluster, return_index=True)
            cluster_vals = a[first_idx]
            perm_vals = cluster_vals[rng.permutation(len(cluster_vals))]
            perm_map = np.empty(cluster.max() + 1, dtype=a.dtype)
            perm_map[clusters] = perm_vals
            out = np.empty_like(a)
            _nb_broadcast_cluster(out, cluster, perm_map)
            return out
        return _np_perm_cluster(a, cluster, rng)

    # 4. Cluster *within* strata  ➜ always NumPy for correctness
    if cluster is not None and strata is not None:
        # NOTE: The previous Numba implementation had an out-of-bounds bug
        # when cluster labels were non-dense inside each stratum.  Until it
        # is re-written safely, we rely on the reference NumPy version.
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
    - If <90% of rows are unique, a warning is issued (e.g., too few clusters).
    """
    if rng is None:
        rng = np.random.default_rng()

    perms = np.empty((reps, a.size), dtype=a.dtype)
    for r in range(reps):
        perms[r] = permute_assignment(
            a,
            cluster=cluster,
            strata=strata,
            rng=rng,
        )

    # Heuristic warning if uniqueness is low
    n_unique = np.unique(perms, axis=0).shape[0]
    if n_unique < 0.9 * reps:
        import warnings

        warnings.warn(
            f"Only {n_unique}/{reps} unique permutations generated "
            f"({n_unique/reps:.1%}). Power may be limited."
        )

    return perms


__all__ = ["permute_assignment", "generate_permuted_matrix"]
