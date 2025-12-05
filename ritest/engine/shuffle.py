"""
Permutation helpers with optional Numba acceleration.

Implements plain, stratified, clustered, and cluster-within-strata
rearrangements for assignment vectors. Upstream validation handles
types and missing-value rules; this module focuses on fast and
deterministic reshuffling.

Design
------
- Four modes: plain, strata, cluster, and cluster-within-strata.
- Numba paths are used when available; otherwise NumPy fallbacks are used.
- All implementations return identical results across Numba/NumPy variants.
- Strata and cluster labels need not be contiguous; all paths support
  arbitrary ordering.

Assumptions
-----------
- Cluster-based permutations require that `a` is constant within clusters.
  This is checked once using a vectorised consistency test.
- Stratified Numba path supports non-contiguous strata using explicit index
  scatter operations.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

try:
    from numba import njit, prange

    NUMBA_OK = True
except ImportError:
    # Safe fallbacks that preserve call signatures when Numba is absent
    NUMBA_OK = False

    def njit(*args, **kwargs):  # noqa: D401
        def wrapper(fn):
            return fn

        return wrapper

    prange = range


# ------------------------------------------------------------------ #
# Plain shuffle
# ------------------------------------------------------------------ #


def _permute_plain(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global permutation of `a` (no constraints)."""
    return rng.permutation(a)


# ------------------------------------------------------------------ #
# Stratified shuffle
# ------------------------------------------------------------------ #


def _np_perm_strata(
    a: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """NumPy implementation: permute `a` independently within each stratum."""
    out = a.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = rng.permutation(out[idx])
    return out


if NUMBA_OK:

    @njit(parallel=True)
    def _nb_scatter_by_index(out, idx, vals):
        """
        Scatter values into selected positions.

        Parameters
        ----------
        out : array
            Destination array.
        idx : array
            Target positions (unique and same length as `vals`).
        vals : array
            Values to place at `idx`.

        Notes
        -----
        Used for stratified shuffling with non-contiguous strata.
        """
        for i in prange(idx.size):
            out[idx[i]] = vals[i]


# ------------------------------------------------------------------ #
# Cluster shuffle
# ------------------------------------------------------------------ #


def _np_perm_cluster(
    a: np.ndarray,
    cluster: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorised cluster-level permutation.

    One representative value per cluster is permuted, then broadcast back
    to members using inverse indexing.
    """
    clusters, first_idx, inv = np.unique(cluster, return_index=True, return_inverse=True)
    cluster_vals = a[first_idx]
    perm_vals = cluster_vals[rng.permutation(clusters.size)]
    out = perm_vals[inv]
    return out


if NUMBA_OK:

    @njit(parallel=True)
    def _nb_broadcast_cluster(out, cluster, perm_map):
        """Broadcast permuted cluster-level values to all rows."""
        for i in prange(cluster.size):
            out[i] = perm_map[cluster[i]]


# ------------------------------------------------------------------ #
# Cluster-within-strata shuffle
# ------------------------------------------------------------------ #


def _np_perm_cluster_strata(
    a: np.ndarray,
    cluster: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Cluster permutation applied separately within each stratum.

    For each stratum:
    - Identify clusters present,
    - Permute the cluster representatives,
    - Broadcast to the units belonging to that stratum.
    """
    out = np.empty_like(a)
    for s in np.unique(strata):
        mask = strata == s
        clust_s = cluster[mask]
        clust_ids, first_idx, inv = np.unique(clust_s, return_index=True, return_inverse=True)
        vals = a[mask][first_idx]
        perm_vals = vals[rng.permutation(clust_ids.size)]
        out[mask] = perm_vals[inv]
    return out


# ------------------------------------------------------------------ #
# Validation helpers
# ------------------------------------------------------------------ #


def _assert_cluster_constant(a: np.ndarray, cluster: np.ndarray) -> None:
    """
    Require that `a` is constant within each cluster.

    Uses a representative-per-cluster vector and inverse indexing to
    verify that all rows in each cluster share the same value.
    """
    clusters, first_idx, inv = np.unique(cluster, return_index=True, return_inverse=True)
    first_vals = a[first_idx]
    expected = first_vals[inv]
    if not np.all(a == expected):
        raise ValueError(
            "permute_assignment: `a` must be constant within each cluster "
            "(cluster-wise permutation assumes a cluster-level value)."
        )


def _can_use_numba_cluster(cluster: np.ndarray) -> bool:
    """
    Check whether dense-label Numba cluster broadcasting is safe.

    Requirements:
    - Non-negative integer labels.
    - Label span not excessively larger than number of unique clusters,
      enforced via a span <= 4 × n_unique heuristic.
    """
    if not np.issubdtype(cluster.dtype, np.integer):
        return False
    cmin = int(np.min(cluster))
    if cmin < 0:
        return False
    cmax = int(np.max(cluster))
    n_unique = np.unique(cluster).size
    span = cmax + 1
    return span <= 4 * n_unique


def _check_lengths(
    a: np.ndarray, cluster: Optional[np.ndarray], strata: Optional[np.ndarray]
) -> None:
    """Ensure optional cluster/strata vectors match the length of `a`."""
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

    Modes
    -----
    Plain:
        Full permutation of `a`.
    Stratified:
        Permute within each stratum; counts preserved per stratum.
    Cluster:
        Permute cluster-level values and broadcast to members.
        Requires `a` to be constant within clusters.
    Cluster-within-strata:
        Apply cluster-wise permutation separately inside each stratum.
    """
    if rng is None:
        rng = np.random.default_rng()

    _check_lengths(a, cluster, strata)

    # Plain ------------------------------------------------------------- #
    if cluster is None and strata is None:
        return _permute_plain(a, rng)

    # Stratified --------------------------------------------------------- #
    if cluster is None and strata is not None:
        if NUMBA_OK:
            # Build index/value lists for all strata; supports arbitrary ordering.
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

    # Cluster ------------------------------------------------------------ #
    if cluster is not None and strata is None:
        _assert_cluster_constant(a, cluster)

        if NUMBA_OK and _can_use_numba_cluster(cluster):
            clusters, first_idx = np.unique(cluster, return_index=True)
            cluster_vals = a[first_idx]
            perm_vals = cluster_vals[rng.permutation(len(cluster_vals))]
            perm_map = np.empty(int(cluster.max()) + 1, dtype=a.dtype)
            perm_map[clusters] = perm_vals
            out = np.empty_like(a)
            _nb_broadcast_cluster(out, cluster, perm_map)
            return out

        return _np_perm_cluster(a, cluster, rng)

    # Cluster within strata --------------------------------------------- #
    if cluster is not None and strata is not None:
        _assert_cluster_constant(a, cluster)
        return _np_perm_cluster_strata(a, cluster, strata, rng)

    raise ValueError("Unhandled permutation mode.")


# ------------------------------------------------------------------ #
# Full permutation matrix
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
        Base assignment vector.
    reps : int
        Number of permutations.
    cluster, strata : ndarray, optional
        Mode selectors; see `permute_assignment`.
    rng : Generator, optional
        RNG reused across draws for reproducibility.

    Notes
    -----
    - For clustered modes, a lightweight uniqueness heuristic is run to detect
      extreme degeneracy in cluster-wise permutations.
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

    # Cluster uniqueness diagnostic
    if cluster is not None and reps >= 2:
        import warnings

        if strata is not None:
            pairs = np.stack((strata, cluster), axis=1)
            _, first_idx = np.unique(pairs, axis=0, return_index=True)
        else:
            _, first_idx = np.unique(cluster, return_index=True)

        rows = perms[:, first_idx]

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


# ------------------------------------------------------------------ #
# Streaming generator of permutation blocks
# ------------------------------------------------------------------ #


def iter_permuted_matrix(
    a: np.ndarray,
    reps: int,
    *,
    cluster: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    chunk_rows: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Yield successive blocks of permuted copies of `a`.

    Produces blocks of at most `chunk_rows` rows until `reps` total rows
    have been generated. Preserves dtype and uses the same RNG instance for
    deterministic sequences regardless of chunking.
    """
    if rng is None:
        rng = np.random.default_rng()

    if reps < 0:
        raise ValueError("iter_permuted_matrix: `reps` must be non-negative.")
    if chunk_rows is None or chunk_rows >= reps:
        yield generate_permuted_matrix(a, reps, cluster=cluster, strata=strata, rng=rng)
        return

    n = a.size
    produced = 0
    while produced < reps:
        m = min(chunk_rows, reps - produced)
        block = np.empty((m, n), dtype=a.dtype)
        for i in range(m):
            block[i] = permute_assignment(a, cluster=cluster, strata=strata, rng=rng)
        produced += m
        yield block


__all__ = ["permute_assignment", "generate_permuted_matrix", "iter_permuted_matrix"]
