# tests/test_shuffle.py
"""
Comprehensive tests for ritest.engine.shuffle.

Run with:
    pytest -q -s tests/test_shuffle.py
"""

import warnings

import numpy as np
import pytest

from ritest.engine.shuffle import generate_permuted_matrix, permute_assignment

# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #


def multiset_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if two 1d arrays contain the same elements with multiplicity."""
    return np.array_equal(np.sort(a, kind="mergesort"), np.sort(b, kind="mergesort"))


def describe_perm(label: str, x: np.ndarray, out: np.ndarray) -> None:
    """Print a small, human-readable summary of a permutation."""
    print(f"[{label}] n={x.size}, dtype={x.dtype}")
    print("  first 10 original:", x[:10])
    print("  first 10 permuted:", out[:10])
    print("  same multiset?   :", multiset_equal(x, out))
    print()


# ------------------------------------------------------------------ #
# 1) Plain permutations
# ------------------------------------------------------------------ #


def test_plain_permutation_reorders_but_preserves_values():
    x = np.arange(30, dtype=np.int64)
    rng = np.random.default_rng(123)
    out = permute_assignment(x, rng=rng)
    describe_perm("plain", x, out)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert multiset_equal(x, out)
    # It's possible (though extremely unlikely) to get identity by chance; allow fallback
    if np.array_equal(out, x):
        rng2 = np.random.default_rng(124)
        out2 = permute_assignment(x, rng=rng2)
        assert not np.array_equal(out2, x)


def test_plain_permutation_does_not_modify_input():
    x = np.arange(20, dtype=np.int32)
    x_copy = x.copy()
    out = permute_assignment(x, rng=np.random.default_rng(7))
    print("[plain immutability] input unchanged, output shape ok")
    assert np.array_equal(x, x_copy)
    assert out.shape == x.shape


# ------------------------------------------------------------------ #
# 2) Stratified permutations (including non-contiguous strata)
# ------------------------------------------------------------------ #


def test_stratified_preserves_within_stratum_multiset():
    # Interleaved / non-contiguous strata: 0,1,0,1,0,1,...
    n = 24
    x = np.arange(n)  # distinct values to make checks strict
    strata = np.arange(n) % 2
    rng = np.random.default_rng(123)
    out = permute_assignment(x, strata=strata, rng=rng)
    describe_perm("stratified (non-contiguous)", x, out)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # Global multiset preserved
    assert multiset_equal(x, out)
    # Within each stratum, the multiset is preserved and elements never cross strata
    for s in (0, 1):
        idx = np.where(strata == s)[0]
        assert multiset_equal(x[idx], out[idx])


def test_stratified_edge_singleton_strata():
    # Some strata with a single element; permutation there is a no-op but should not error
    x = np.array([10, 11, 12, 13, 14, 15])
    strata = np.array([0, 1, 1, 2, 3, 3])  # stratum 0 and 2 are singletons
    out = permute_assignment(x, strata=strata, rng=np.random.default_rng(9))
    print("[stratified singleton] works and preserves per-stratum values")
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        assert multiset_equal(x[idx], out[idx])


def test_stratified_length_mismatch_raises():
    x = np.arange(10)
    strata = np.ones(9, dtype=int)  # wrong length
    with pytest.raises(ValueError, match="strata.*length.*match"):
        _ = permute_assignment(x, strata=strata, rng=np.random.default_rng(1))


# ------------------------------------------------------------------ #
# 3) Cluster permutations
# ------------------------------------------------------------------ #


def test_cluster_permutation_broadcasts_cluster_values_and_preserves_set():
    # Three clusters, cluster-constant values 5, 7, 9
    cluster = np.repeat([0, 1, 2], repeats=[4, 3, 5])
    a = np.where(cluster == 0, 5, np.where(cluster == 1, 7, 9)).astype(np.int64)
    rng = np.random.default_rng(123)
    out = permute_assignment(a, cluster=cluster, rng=rng)
    describe_perm("cluster", a, out)

    assert out.shape == a.shape
    # Within each cluster, values are constant
    for c in np.unique(cluster):
        vals = out[cluster == c]
        assert np.all(vals == vals[0])
    # The multiset of unique cluster-level values is preserved (permutation)
    orig_cluster_vals = a[np.unique(cluster, return_index=True)[1]]
    new_cluster_vals = out[np.unique(cluster, return_index=True)[1]]
    assert multiset_equal(orig_cluster_vals, new_cluster_vals)


def test_cluster_constancy_violation_raises():
    # Same cluster id but different 'a' values inside → should raise
    cluster = np.array([0, 0, 1, 1], dtype=int)
    a = np.array([10, 11, 20, 20], dtype=int)  # cluster 0 not constant
    with pytest.raises(ValueError, match="must be constant within each cluster"):
        _ = permute_assignment(a, cluster=cluster, rng=np.random.default_rng(99))


def test_cluster_length_mismatch_raises():
    a = np.array([1, 1, 2, 2])
    cluster = np.array([0, 0, 1])  # wrong length
    with pytest.raises(ValueError, match="cluster.*length.*match"):
        _ = permute_assignment(a, cluster=cluster, rng=np.random.default_rng(1))


# ------------------------------------------------------------------ #
# 4) Cluster-within-strata permutations
# ------------------------------------------------------------------ #


def test_cluster_within_strata_invariants_hold():
    # Two strata; in each, clusters have their own constant values
    cluster = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    # cluster values: in stratum 0 -> {10, 20}; in stratum 1 -> {30, 40}
    a = np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=int)
    out = permute_assignment(
        a, cluster=cluster, strata=strata, rng=np.random.default_rng(5)
    )
    describe_perm("cluster-within-strata", a, out)

    assert out.shape == a.shape
    # Within each cluster, constant values
    for c in np.unique(cluster):
        vals = out[cluster == c]
        assert np.all(vals == vals[0])

    # Per stratum, the set of cluster-level representatives is preserved
    for s in np.unique(strata):
        mask = strata == s
        _, first_idx = np.unique(cluster[mask], return_index=True)
        orig = a[mask][first_idx]
        new = out[mask][first_idx]
        assert multiset_equal(orig, new)


def test_cluster_within_strata_length_mismatch_raises():
    a = np.array([10, 10, 20, 20])
    cluster = np.array([0, 0, 1, 1])
    strata = np.array([0, 0, 0])  # wrong length
    with pytest.raises(ValueError, match="strata.*length.*match"):
        _ = permute_assignment(
            a, cluster=cluster, strata=strata, rng=np.random.default_rng(2)
        )


# ------------------------------------------------------------------ #
# 5) Determinism & dtype preservation
# ------------------------------------------------------------------ #


def test_same_seed_same_result_different_seed_diff_result():
    # Determinism with cluster×strata mode: make `a` cluster-constant.
    cl = np.repeat(np.arange(5), 10)  # 5 clusters, size 10 each
    st = np.tile([0, 1], 25)  # alternating strata
    # cluster-level values -> [10, 20, 30, 40, 50], broadcast to members
    a = np.repeat(np.array([10, 20, 30, 40, 50]), 10)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    rng3 = np.random.default_rng(124)

    out1 = permute_assignment(a, cluster=cl, strata=st, rng=rng1)
    out2 = permute_assignment(a, cluster=cl, strata=st, rng=rng2)
    out3 = permute_assignment(a, cluster=cl, strata=st, rng=rng3)

    print("[determinism] same seed => identical; different seed => different")
    assert np.array_equal(out1, out2)
    assert not np.array_equal(out1, out3)


def test_dtype_roundtrip_bool_and_float():
    x_bool = np.arange(20) % 2 == 0
    x_float = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    rng = np.random.default_rng(321)

    out_b = permute_assignment(x_bool, rng=rng)
    out_f = permute_assignment(x_float, strata=np.arange(20) % 3, rng=rng)

    print(f"[dtype] bool -> {out_b.dtype}, float -> {out_f.dtype}")
    assert out_b.dtype == x_bool.dtype
    assert out_f.dtype == x_float.dtype


# ------------------------------------------------------------------ #
# 6) generate_permuted_matrix: shapes, determinism, uniqueness diag
# ------------------------------------------------------------------ #


def test_generate_permuted_matrix_shapes_and_content_plain_and_strata():
    n, reps = 25, 7
    x = np.arange(n)
    st = np.arange(n) % 5
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    perms_plain_1 = generate_permuted_matrix(x, reps, rng=rng1)
    perms_plain_2 = generate_permuted_matrix(x, reps, rng=rng2)
    print("[gpm plain] shape, dtype, determinism checked")

    assert perms_plain_1.shape == (reps, n)
    assert perms_plain_1.dtype == x.dtype
    assert np.array_equal(perms_plain_1, perms_plain_2)

    perms_strata = generate_permuted_matrix(
        x, reps, strata=st, rng=np.random.default_rng(7)
    )
    print("[gpm strata] per-row invariants checked")
    for r in range(reps):
        assert multiset_equal(perms_strata[r, st == 0], x[st == 0])
        assert multiset_equal(perms_strata[r, st == 1], x[st == 1])


def test_generate_permuted_matrix_cluster_invariants_and_determinism():
    # Small n, small G to keep it light
    cluster = np.repeat([0, 1, 2], [4, 3, 5])
    a = np.where(cluster == 0, 5, np.where(cluster == 1, 7, 9))
    reps = 10
    rng1 = np.random.default_rng(2024)
    rng2 = np.random.default_rng(2024)

    perms1 = generate_permuted_matrix(a, reps, cluster=cluster, rng=rng1)
    perms2 = generate_permuted_matrix(a, reps, cluster=cluster, rng=rng2)

    print("[gpm cluster] shape/dtype + determinism + per-row cluster constancy")
    assert np.array_equal(perms1, perms2)
    assert perms1.shape == (reps, a.size)
    assert perms1.dtype == a.dtype

    # Per row, cluster-constancy and cluster-level multiset preserved
    _, first_idx = np.unique(cluster, return_index=True)
    orig_rep = a[first_idx]
    for r in range(reps):
        out = perms1[r]
        for c in np.unique(cluster):
            vals = out[cluster == c]
            assert np.all(vals == vals[0])
        assert multiset_equal(orig_rep, out[first_idx])


def test_generate_permuted_matrix_cluster_within_strata_invariants():
    # Two strata, two clusters in each
    cluster = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    a = np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=int)
    reps = 9

    perms = generate_permuted_matrix(
        a, reps, cluster=cluster, strata=strata, rng=np.random.default_rng(77)
    )
    print("[gpm cluster×strata] per-row invariants checked")

    for r in range(reps):
        out = perms[r]
        # cluster-constancy
        for c in np.unique(cluster):
            vals = out[cluster == c]
            assert np.all(vals == vals[0])
        # per-stratum cluster-representatives equal as multisets
        for s in np.unique(strata):
            mask = strata == s
            _, first_idx = np.unique(cluster[mask], return_index=True)
            assert multiset_equal(a[mask][first_idx], out[mask][first_idx])


def test_generate_permuted_matrix_zero_and_negative_reps():
    a = np.arange(12)
    # Zero reps: valid, returns (0, n)
    perms0 = generate_permuted_matrix(a, 0, rng=np.random.default_rng(1))
    print("[gpm] zero reps returns empty matrix with correct shape")
    assert perms0.shape == (0, a.size)

    # Negative reps: should raise
    with pytest.raises(ValueError, match="reps.*non-negative"):
        _ = generate_permuted_matrix(a, -1, rng=np.random.default_rng(1))


def test_uniqueness_warning_triggers_for_tiny_cluster_space():
    # Only 2 clusters -> only 2 possible permutations; with many reps, uniqueness < 90%
    cluster = np.repeat([0, 1], 5)
    a = np.where(cluster == 0, 0, 1)
    reps = 3000  # large enough to trigger warning; also triggers downsampling path (>5000) if raised
    rng = np.random.default_rng(555)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = generate_permuted_matrix(a, reps, cluster=cluster, rng=rng)

    msgs = [str(rec.message) for rec in w]
    print("[gpm uniqueness] captured warnings:", msgs)
    assert any("Low uniqueness in cluster permutations" in m for m in msgs)


def test_no_uniqueness_warning_for_plain_or_stratified():
    a = np.arange(40)
    st = np.arange(40) % 4
    rng = np.random.default_rng(1234)

    with warnings.catch_warnings(record=True) as w_plain:
        warnings.simplefilter("always")
        _ = generate_permuted_matrix(a, 500, rng=rng)
    with warnings.catch_warnings(record=True) as w_strata:
        warnings.simplefilter("always")
        _ = generate_permuted_matrix(a, 500, strata=st, rng=rng)

    print("[gpm uniqueness] no warnings for plain/stratified modes")
    assert len(w_plain) == 0
    assert len(w_strata) == 0
