# tests/test_config.py
# Comprehensive tests for ritest.config
#
# Covers:
# - Shape & types of DEFAULTS
# - ritest_get (full copy vs single key) and immutability
# - ritest_set validation (all-or-nothing semantics, error paths)
# - n_jobs constraints (-1 or >=1)
# - alpha in (0,1), reps > 0
# - ci_method / ci_mode allowed sets
# - ci_range / ci_step / ci_tol positivity
# - NEW: perm_chunk_bytes / perm_chunk_min_rows shape, types, validation
# - ritest_reset (all and subset) and identity preservation
# - ritest_config context manager (normal and exception paths, nesting)
# - Unknown keys and non-mapping overrides

import pytest

from ritest.config import DEFAULTS, ritest_config, ritest_get, ritest_reset, ritest_set


@pytest.fixture(autouse=True)
def clean_config_state():
    """Ensure each test starts and ends from import-time defaults."""
    ritest_reset()
    yield
    ritest_reset()


def test_defaults_shape_and_types():
    d = ritest_get()
    # Expected keys (explicit to detect accidental additions/removals)
    expected_keys = {
        "reps",
        "seed",
        "alpha",
        "ci_method",
        "ci_mode",
        "ci_range",
        "ci_step",
        "ci_tol",
        "n_jobs",
        # NEW (chunking)
        "perm_chunk_bytes",
        "perm_chunk_min_rows",
    }
    assert set(d.keys()) == expected_keys

    # Types / ranges
    assert isinstance(d["reps"], int) and d["reps"] > 0
    assert isinstance(d["seed"], int)
    assert isinstance(d["alpha"], float) and 0 < d["alpha"] < 1
    assert d["ci_method"] in {"clopper-pearson", "normal"}
    assert d["ci_mode"] in {"bounds", "band", "none"}
    assert isinstance(d["ci_range"], float) and d["ci_range"] > 0
    assert isinstance(d["ci_step"], float) and d["ci_step"] > 0
    assert isinstance(d["ci_tol"], float) and d["ci_tol"] > 0
    assert isinstance(d["n_jobs"], int)

    # NEW: chunking keys
    assert isinstance(d["perm_chunk_bytes"], int) and d["perm_chunk_bytes"] > 0
    assert isinstance(d["perm_chunk_min_rows"], int) and d["perm_chunk_min_rows"] >= 1


def test_get_returns_copy_and_single_key_access():
    before = ritest_get()
    # Mutating the returned dict must not affect DEFAULTS
    before["alpha"] = 0.123
    now = ritest_get()
    assert now["alpha"] != 0.123

    # Single-key access
    assert isinstance(ritest_get("reps"), int)

    # Unknown key raises KeyError
    with pytest.raises(KeyError):
        ritest_get("not_a_key")


def test_set_valid_updates_and_identity_preserved():
    d_id = id(DEFAULTS)
    ritest_set({"alpha": 0.1, "reps": 2000})
    assert id(DEFAULTS) == d_id  # identity preserved
    got = ritest_get()
    assert got["alpha"] == 0.1
    assert got["reps"] == 2000


def test_ci_method_alias_normalised_to_canonical():
    ritest_set({"ci_method": "cp"})
    assert ritest_get("ci_method") == "clopper-pearson"


def test_set_all_or_nothing_semantics():
    snapshot = ritest_get()
    # This should fail (n_jobs=0 invalid), and nothing should change
    with pytest.raises(ValueError):
        ritest_set({"alpha": 0.1, "n_jobs": 0})
    after = ritest_get()
    assert after == snapshot  # no partial updates applied


def test_set_rejects_non_mapping():
    with pytest.raises(ValueError):
        ritest_set([("alpha", 0.2)])  # not a Mapping


@pytest.mark.parametrize(
    "overrides",
    [
        {"reps": 0},
        {"reps": -5},
        {"reps": 3.14},
        {"reps": "10"},
    ],
)
def test_reps_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": 3.14},
        {"seed": "not-int"},
    ],
)
def test_seed_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha": 0},  # boundary
        {"alpha": 1},  # boundary
        {"alpha": -0.001},
        {"alpha": 1.001},
        {"alpha": "0.1x"},
    ],
)
def test_alpha_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ci_method": "beta"},
        {"ci_method": "pearson"},
        {"ci_method": 123},
    ],
)
def test_ci_method_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ci_mode": "BOUNDs"},
        {"ci_mode": "bands"},
        {"ci_mode": "grid"},
        {"ci_mode": 0},
    ],
)
def test_ci_mode_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


@pytest.mark.parametrize(
    "key, bad_value",
    [
        ("ci_range", 0.0),
        ("ci_range", -1),
        ("ci_range", "x"),
        ("ci_step", 0.0),
        ("ci_step", -0.01),
        ("ci_step", "a"),
        ("ci_tol", 0.0),
        ("ci_tol", -1e-6),
        ("ci_tol", "tiny"),
    ],
)
def test_ci_numeric_positive_validation(key, bad_value):
    with pytest.raises(ValueError):
        ritest_set({key: bad_value})


def test_n_jobs_validation_strict():
    # Allowed
    ritest_set({"n_jobs": -1})
    assert ritest_get("n_jobs") == -1
    ritest_set({"n_jobs": 1})
    assert ritest_get("n_jobs") == 1
    ritest_set({"n_jobs": 8})
    assert ritest_get("n_jobs") == 8

    # Disallowed
    for bad in (0, -2, -99):
        with pytest.raises(ValueError):
            ritest_set({"n_jobs": bad})

    # Non-int
    with pytest.raises(ValueError):
        ritest_set({"n_jobs": 3.14})


# NEW: chunking validation
@pytest.mark.parametrize(
    "overrides",
    [
        {"perm_chunk_bytes": 0},
        {"perm_chunk_bytes": -1},
        {"perm_chunk_bytes": 3.14},
        {"perm_chunk_bytes": "1GiB"},
    ],
)
def test_perm_chunk_bytes_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"perm_chunk_min_rows": 0},
        {"perm_chunk_min_rows": -5},
        {"perm_chunk_min_rows": 2.5},
        {"perm_chunk_min_rows": "64"},
    ],
)
def test_perm_chunk_min_rows_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


def test_unknown_key_rejected_in_set_and_reset():
    with pytest.raises(ValueError):
        ritest_set({"not_a_key": 123})

    with pytest.raises(ValueError):
        ritest_reset(keys=["not_a_key"])


def test_reset_all_restores_import_time_defaults_and_identity():
    d_id = id(DEFAULTS)
    baseline = ritest_get()

    # Change many settings
    ritest_set(
        {
            "alpha": 0.1,
            "reps": 777,
            "ci_method": "normal",
            "ci_mode": "band",
            "ci_range": 4.0,
            "ci_step": 0.01,
            "ci_tol": 1e-5,
            "n_jobs": 1,
            "seed": 999,
            # NEW (chunking)
            "perm_chunk_bytes": 8 * 1024 * 1024,
            "perm_chunk_min_rows": 128,
        }
    )
    # Reset all and check identity preserved
    ritest_reset()
    assert id(DEFAULTS) == d_id

    # Assert we are back to import-time defaults (whatever they are)
    d = ritest_get()
    assert d == baseline


def test_reset_subset_only_affects_selected_keys():
    baseline = ritest_get()

    # Change three keys
    ritest_set({"alpha": 0.1, "reps": 321, "n_jobs": 1})
    # Reset a subset
    ritest_reset(keys=["alpha", "n_jobs"])

    d = ritest_get()
    # Reset keys restored to baseline
    assert d["alpha"] == baseline["alpha"]
    assert d["n_jobs"] == baseline["n_jobs"]
    # Unlisted key remains overridden
    assert d["reps"] == 321


def test_context_manager_temporary_overrides_and_restore():
    base = ritest_get()

    with ritest_config({"alpha": 0.1, "reps": base["reps"] + 1}):
        assert ritest_get("alpha") == 0.1
        assert ritest_get("reps") == base["reps"] + 1

    # Restored to prior state (import-time defaults here due to autouse fixture)
    after = ritest_get()
    assert after == base


def test_context_manager_restores_on_exception():
    base = ritest_get()

    with pytest.raises(RuntimeError):
        with ritest_config({"alpha": base["alpha"] + 0.01}):
            assert ritest_get("alpha") == base["alpha"] + 0.01
            raise RuntimeError("boom")

    # Restored
    assert ritest_get() == base


def test_context_manager_nesting_behavior():
    base = ritest_get()
    assert ritest_get() == base

    with ritest_config({"alpha": base["alpha"] + 0.1}):
        # Outer override applied
        outer = ritest_get()
        assert outer["alpha"] == base["alpha"] + 0.1
        assert outer["reps"] == base["reps"]

        with ritest_config({"reps": base["reps"] + 10}):
            # Inner override augments outer
            inner = ritest_get()
            assert inner["alpha"] == base["alpha"] + 0.1
            assert inner["reps"] == base["reps"] + 10

        # After inner, outer still in effect
        after_inner = ritest_get()
        assert after_inner["alpha"] == base["alpha"] + 0.1
        assert after_inner["reps"] == base["reps"]

    # After outer, fully restored
    assert ritest_get() == base
