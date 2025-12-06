"""Configuration handling for ritest.

Defines the DEFAULTS dict that stores all global configuration values,
and the public interface for safely reading/updating them.

Public API:
- DEFAULTS                : live dict with current global config (do not mutate directly)
- ritest_set(overrides)   : validate and update selected keys (in-place)
- ritest_get(key=None)    : read a single value or a (shallow) copy of all config
- ritest_reset(keys=None) : restore all or selected keys to import-time defaults
- ritest_config(overrides): context manager for temporary overrides (auto-reset)

Notes:
- DEFAULTS is a live dictionary used internally throughout the package.
- Prefer ritest_set / ritest_reset / ritest_config over mutating DEFAULTS directly.
- Mutations are applied in-place (identity of DEFAULTS is preserved).
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = [
    "DEFAULTS",
    "ritest_set",
    "ritest_get",
    "ritest_reset",
    "ritest_config",
]

# ---------------------------------------------------------------------
# Global config used by all internal modules
# ---------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    # --- Main permutation settings ---
    "reps": 1000,  # Number of permutations
    "seed": 23,  # Random seed (default: for MJ fans)
    "alpha": 0.05,  # Significance level for p-value and CI
    # --- CI for permutation p-value ---
    "ci_method": "cp",  # 'cp' = Clopper–Pearson, 'normal' = Wald
    # --- Coefficient-CI controls ---
    # 'none'   : do not compute coefficient CI (neither bounds nor grid)
    # 'bounds' : compute only the 2-point coefficient CI
    # 'grid'   : compute the full (β, p(β)) band (and bounds)
    "ci_mode": "bounds",
    # Applies ONLY when using a generic stat function (`stat_fn`).
    # - If True  and ci_mode in {'bounds','grid'}: compute generic coef CI
    #   (grid allowed but potentially slow; a runtime warning may be shown).
    # - If False and using stat_fn: skip coefficient CI even if ci_mode != 'none'.
    # Ignored for fast-linear (formula) models, where fast CIs are always available
    # unless ci_mode == 'none'.
    "coef_ci_generic": False,
    # Search/grid sizing for coefficient CI (interpreted in SE units)
    "ci_range": 3.0,  # search half-range in SE units
    "ci_step": 0.005,  # grid step in SE units (used when ci_mode == 'grid')
    "ci_tol": 1e-4,  # bisection tolerance (as *SE* fractions)
    # --- Parallelism ---
    "n_jobs": -1,  # -1 = use all available CPU cores
    # --- Memory / chunking (internal; used by run.py to bound memory) ---
    # Soft budget for the in-RAM permutation block. If the full (reps × n × itemsize)
    # permutation matrix would exceed this budget, run.py will stream permutations
    # in chunks using engine.shuffle.iter_permuted_matrix(...).
    #
    # Default: 256 MiB – conservative on 8–16 GB laptops, and configurable.
    "perm_chunk_bytes": 256 * 1024 * 1024,
    # Minimum number of permutation rows per chunk to avoid tiny blocks when n is huge.
    # Has effect only when chunking is enabled by the budget above.
    "perm_chunk_min_rows": 64,
}

# Keep a baseline snapshot to enable full resets.
# Use a deepcopy for future-proofing if nested structures are added later.
_BASE_DEFAULTS: Dict[str, Any] = deepcopy(DEFAULTS)

_ALLOWED_CI_METHODS = {"cp", "normal"}
_ALLOWED_CI_MODES = {"none", "bounds", "grid"}


def _validate_pair(key: str, val: Any) -> None:
    """Raise ValueError if (key, val) is invalid."""
    if key not in DEFAULTS:
        raise ValueError(f"Invalid config key: '{key}'")

    if key == "reps":
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"reps must be a positive integer (got {val!r})")

    elif key == "seed":
        if not isinstance(val, int):
            raise ValueError(f"seed must be an integer (got {val!r})")

    elif key == "alpha":
        try:
            ok = 0 < float(val) < 1
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"alpha must be a float strictly between 0 and 1 (got {val!r})")

    elif key == "ci_method":
        if val not in _ALLOWED_CI_METHODS:
            raise ValueError(f"ci_method must be one of {_ALLOWED_CI_METHODS} (got {val!r})")

    elif key == "ci_mode":
        if val not in _ALLOWED_CI_MODES:
            raise ValueError(f"ci_mode must be one of {_ALLOWED_CI_MODES} (got {val!r})")

    elif key in {"ci_range", "ci_step", "ci_tol"}:
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"{key} must be a positive number (got {val!r})")

    elif key == "coef_ci_generic":
        if not isinstance(val, bool):
            raise ValueError("coef_ci_generic must be True/False")

    elif key == "n_jobs":
        if not isinstance(val, int):
            raise ValueError(f"n_jobs must be an integer (got {val!r})")
        if not (val == -1 or val >= 1):
            raise ValueError("n_jobs must be -1 (all cores) or a positive integer >= 1")

    elif key == "perm_chunk_bytes":
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"perm_chunk_bytes must be a positive integer (bytes, got {val!r})")

    elif key == "perm_chunk_min_rows":
        if not isinstance(val, int) or val < 1:
            raise ValueError(f"perm_chunk_min_rows must be an integer >= 1 (got {val!r})")


def ritest_set(overrides: Mapping[str, Any]) -> None:
    """
    Update the global configuration in-place (validated).

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Dict-like with keys in DEFAULTS. Unknown keys are rejected.

    Raises
    ------
    ValueError
        If unknown keys or invalid values are passed.

    Notes
    -----
    - This mutates global state. Prefer using `ritest_config(...)` when you want
      temporary overrides that automatically revert (e.g., per-test or per-run).
    """
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a mapping of {key: value}")

    # Validate first, then apply (all-or-nothing semantics)
    for k, v in overrides.items():
        _validate_pair(k, v)

    # Apply in-place
    DEFAULTS.update(overrides)


def ritest_get(key: Optional[str] = None) -> Any:
    """
    Read configuration values safely.

    Parameters
    ----------
    key : str or None, optional
        If None (default), returns a shallow copy of the entire config.
        If a key is provided, returns the current value for that key.

    Returns
    -------
    Any
        The value for the given key, or a shallow copy of DEFAULTS if key is None.

    Raises
    ------
    KeyError
        If `key` is provided and is not a valid config key.
    """
    if key is None:
        # Shallow copy is fine because current values are scalars.
        return dict(DEFAULTS)
    if key not in DEFAULTS:
        raise KeyError(f"Unknown config key: {key!r}")
    return DEFAULTS[key]


def ritest_reset(keys: Optional[Iterable[str]] = None) -> None:
    """
    Reset configuration to the import-time defaults.

    Parameters
    ----------
    keys : iterable of str or None, optional
        - None (default): reset all keys to baseline values.
        - Iterable: reset only those keys (unknown keys raise ValueError).

    Notes
    -----
    Resets are applied in-place (identity of DEFAULTS is preserved).
    """
    if keys is None:
        DEFAULTS.clear()
        DEFAULTS.update(_BASE_DEFAULTS)
        return

    # Validate keys first
    to_reset = list(keys)
    for k in to_reset:
        if k not in DEFAULTS:
            raise ValueError(f"Unknown config key for reset: {k!r}")

    # Reset only selected keys
    for k in to_reset:
        DEFAULTS[k] = _BASE_DEFAULTS[k]


@contextmanager
def ritest_config(overrides: Mapping[str, Any]):
    """
    Context manager for temporary configuration overrides.

    Example
    -------
    >>> with ritest_config({"alpha": 0.1, "reps": 2000}):
    ...     # run code with temporary config
    ...     pass
    >>> # here config is restored to previous values

    Notes
    -----
    - Validates and applies overrides on entry.
    - Always restores the prior configuration on exit, even if an exception is raised.
    - The restore preserves the identity of DEFAULTS.
    """
    prev = dict(DEFAULTS)  # shallow snapshot of scalar values
    try:
        ritest_set(overrides)
        yield
    finally:
        # Clear and update in-place to preserve object identity.
        DEFAULTS.clear()
        DEFAULTS.update(prev)
