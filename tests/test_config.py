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
# - coef_ci_generic bool
# - ritest_reset (all and subset) and identity preservation
# - ritest_config context manager (normal and exception paths, nesting)
# - Unknown keys and non-mapping overrides

import pytest

from ritest.config import DEFAULTS, ritest_config, ritest_get, ritest_reset, ritest_set

# ----------------------------------------------------------------------
# Test utilities / fixtures
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_config_state():
    """Ensure each test starts and ends from import-time defaults.

    We reset at the beginning and at the end of each test so tests are isolated.
    """
    ritest_reset()
    yield
    ritest_reset()


# ----------------------------------------------------------------------
# Basic shape and types
# ----------------------------------------------------------------------


def test_defaults_shape_and_types():
    d = ritest_get()
    # Expected keys (explicit to detect accidental additions/removals)
    expected_keys = {
        "reps",
        "seed",
        "alpha",
        "ci_method",
        "ci_mode",
        "coef_ci_generic",
        "ci_range",
        "ci_step",
        "ci_tol",
        "n_jobs",
    }
    assert set(d.keys()) == expected_keys

    # Types (do not assert exact values here; those are covered below)
    assert isinstance(d["reps"], int) and d["reps"] > 0
    assert isinstance(d["seed"], int)
    assert isinstance(d["alpha"], float) and 0 < d["alpha"] < 1
    assert d["ci_method"] in {"cp", "normal"}
    assert d["ci_mode"] in {"bounds", "grid"}
    assert isinstance(d["coef_ci_generic"], bool)
    assert isinstance(d["ci_range"], float) and d["ci_range"] > 0
    assert isinstance(d["ci_step"], float) and d["ci_step"] > 0
    assert isinstance(d["ci_tol"], float) and d["ci_tol"] > 0
    assert isinstance(d["n_jobs"], int)


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


# ----------------------------------------------------------------------
# ritest_set: valid updates, identity preservation, and all-or-nothing
# ----------------------------------------------------------------------


def test_set_valid_updates_and_identity_preserved():
    d_id = id(DEFAULTS)
    ritest_set({"alpha": 0.1, "reps": 2000})
    assert id(DEFAULTS) == d_id  # identity preserved
    got = ritest_get()
    assert got["alpha"] == 0.1
    assert got["reps"] == 2000


def test_set_all_or_nothing_semantics():
    # Snapshot before attempting a mixed-good/bad update
    snapshot = ritest_get()
    # This should fail (n_jobs=0 invalid), and nothing should change
    with pytest.raises(ValueError):
        ritest_set({"alpha": 0.1, "n_jobs": 0})

    after = ritest_get()
    assert after == snapshot  # no partial updates applied


def test_set_rejects_non_mapping():
    with pytest.raises(ValueError):
        ritest_set([("alpha", 0.2)])  # not a Mapping


# ----------------------------------------------------------------------
# Validation matrix
# ----------------------------------------------------------------------


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
        {"ci_method": "CP"},  # case sensitive
        {"ci_method": "beta"},
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
        {"ci_mode": "band"},
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"coef_ci_generic": "true"},
        {"coef_ci_generic": 1},
        {"coef_ci_generic": None},
    ],
)
def test_coef_ci_generic_bool_validation(overrides):
    with pytest.raises(ValueError):
        ritest_set(overrides)


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


def test_unknown_key_rejected_in_set_and_reset():
    with pytest.raises(ValueError):
        ritest_set({"not_a_key": 123})

    with pytest.raises(ValueError):
        ritest_reset(keys=["not_a_key"])


# ----------------------------------------------------------------------
# Reset behavior (all vs subset)
# ----------------------------------------------------------------------


def test_reset_all_restores_import_time_defaults_and_identity():
    d_id = id(DEFAULTS)
    # Change many settings
    ritest_set(
        {
            "alpha": 0.1,
            "reps": 777,
            "ci_method": "normal",
            "ci_mode": "grid",
            "coef_ci_generic": True,
            "ci_range": 4.0,
            "ci_step": 0.01,
            "ci_tol": 1e-5,
            "n_jobs": 1,
            "seed": 999,
        }
    )
    # Reset all and check identity preserved
    ritest_reset()
    assert id(DEFAULTS) == d_id

    # Assert we are back to import-time defaults
    d = ritest_get()
    assert d["alpha"] == 0.05
    assert d["reps"] == 100
    assert d["ci_method"] == "cp"
    assert d["ci_mode"] == "bounds"
    assert d["coef_ci_generic"] is False
    assert d["ci_range"] == 3.0
    assert d["ci_step"] == 0.005
    assert d["ci_tol"] == 1e-4
    assert d["n_jobs"] == -1
    assert d["seed"] == 23


def test_reset_subset_only_affects_selected_keys():
    # Change three keys
    ritest_set({"alpha": 0.1, "reps": 321, "n_jobs": 1})
    # Reset a subset
    ritest_reset(keys=["alpha", "n_jobs"])

    d = ritest_get()
    # Reset keys restored
    assert d["alpha"] == 0.05
    assert d["n_jobs"] == -1
    # Unlisted key remains overridden
    assert d["reps"] == 321


# ----------------------------------------------------------------------
# Context manager behavior
# ----------------------------------------------------------------------


def test_context_manager_temporary_overrides_and_restore():
    base = ritest_get()
    assert base["alpha"] == 0.05
    assert base["reps"] == 100

    with ritest_config({"alpha": 0.1, "reps": 999}):
        assert ritest_get("alpha") == 0.1
        assert ritest_get("reps") == 999

    # Restored to prior state (import-time defaults here due to autouse fixture)
    after = ritest_get()
    assert after["alpha"] == 0.05
    assert after["reps"] == 100


def test_context_manager_restores_on_exception():
    with pytest.raises(RuntimeError):
        with ritest_config({"alpha": 0.2}):
            assert ritest_get("alpha") == 0.2
            raise RuntimeError("boom")

    # Restored
    assert ritest_get("alpha") == 0.05


def test_context_manager_nesting_behavior():
    assert ritest_get("alpha") == 0.05
    assert ritest_get("reps") == 100

    with ritest_config({"alpha": 0.2}):
        # Outer override applied
        assert ritest_get("alpha") == 0.2
        assert ritest_get("reps") == 100

        with ritest_config({"reps": 10}):
            # Inner override augments outer
            assert ritest_get("alpha") == 0.2
            assert ritest_get("reps") == 10

        # After inner, outer still in effect
        assert ritest_get("alpha") == 0.2
        assert ritest_get("reps") == 100

    # After outer, fully restored
    assert ritest_get("alpha") == 0.05
    assert ritest_get("reps") == 100
