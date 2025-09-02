# tests/test_shuffle.py
"""
Comprehensive unit tests for ritest.engine.shuffle.permute_assignment.

Run with:
    pytest -q -s tests/test_shuffle.py
"""

import numpy as np

from ritest.engine.shuffle import permute_assignment

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def assert_same_values(x, y):
    """Check that two arrays contain the same elements, regardless of order."""
    assert sorted(x.tolist()) == sorted(y.tolist())


# ------------------------------------------------------------------ #
# 1. Plain shuffle
# ------------------------------------------------------------------ #


def test_plain_permutation():
    x = np.arange(10)
    rng = np.random.default_rng(123)
    out = permute_assignment(x, rng=rng)
    print("Plain permutation – passed")
    assert out.shape == x.shape
    assert_same_values(x, out)
    assert not np.all(out == x)  # Should actually be shuffled


# ------------------------------------------------------------------ #
# 2. Stratified shuffle
# ------------------------------------------------------------------ #


def test_stratified_shuffle():
    x = np.arange(10)
    strata = np.repeat([0, 1], 5)
    rng = np.random.default_rng(123)
    out = permute_assignment(x, strata=strata, rng=rng)
    print("Stratified permutation – passed")

    assert out.shape == x.shape
    assert_same_values(x, out)
    for s in [0, 1]:
        idx = np.where(strata == s)[0]
        assert_same_values(x[idx], out[idx])


# ------------------------------------------------------------------ #
# 3. Cluster shuffle
# ------------------------------------------------------------------ #


def test_cluster_shuffle():
    x = np.array([1, 1, 2, 2, 3, 3])
    cluster = np.array([0, 0, 1, 1, 2, 2])
    rng = np.random.default_rng(123)
    out = permute_assignment(x, cluster=cluster, rng=rng)
    print("Cluster permutation – passed")

    assert out.shape == x.shape
    assert set(out) == set(x)
    for c in [0, 1, 2]:
        vals = out[cluster == c]
        assert np.all(vals == vals[0])


# ------------------------------------------------------------------ #
# 4. Cluster-within-strata shuffle
# ------------------------------------------------------------------ #


def test_cluster_within_strata_shuffle():
    x = np.array([10, 10, 20, 20, 30, 30])
    cluster = np.array([0, 0, 1, 1, 2, 2])
    strata = np.array([0, 0, 0, 0, 1, 1])
    rng = np.random.default_rng(123)
    out = permute_assignment(x, cluster=cluster, strata=strata, rng=rng)
    print("Cluster-within-strata permutation – passed")

    assert out.shape == x.shape
    assert set(out) == set(x)
    for c in np.unique(cluster):
        vals = out[cluster == c]
        assert np.all(vals == vals[0])


# ------------------------------------------------------------------ #
# 5. Dispatcher: no crashes, valid shapes and values
# ------------------------------------------------------------------ #


def test_dispatcher_variants():
    x = np.arange(12)
    cluster = np.repeat([0, 1, 2], 4)
    strata = np.tile([0, 1], 6)
    rng = np.random.default_rng(42)

    plain = permute_assignment(x, rng=rng)
    strat = permute_assignment(x, strata=strata, rng=rng)
    clust = permute_assignment(x, cluster=cluster, rng=rng)
    both = permute_assignment(x, cluster=cluster, strata=strata, rng=rng)
    print("Dispatcher test – all variants passed")

    for result in [plain, strat, clust, both]:
        assert result.shape == x.shape
        assert set(result) <= set(x)


# ------------------------------------------------------------------ #
# 6. Fallback to NumPy when Numba is not available
# ------------------------------------------------------------------ #


def test_fallback_without_numba(monkeypatch):
    import ritest.engine.shuffle as shuffle

    monkeypatch.setattr(shuffle, "NUMBA_OK", False)

    x = np.array([10, 10, 20, 20, 30, 30])
    cluster = np.array([0, 0, 1, 1, 2, 2])
    strata = np.array([0, 0, 0, 0, 1, 1])
    rng = np.random.default_rng(999)
    out = shuffle.permute_assignment(x, cluster=cluster, strata=strata, rng=rng)
    print("Fallback to NumPy permutation – passed")

    assert out.shape == x.shape
    assert set(out) == set(x)
    for c in np.unique(cluster):
        vals = out[cluster == c]
        assert np.all(vals == vals[0])
