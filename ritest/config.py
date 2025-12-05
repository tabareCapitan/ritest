"""Global configuration for ritest.

Defines a shared DEFAULTS dict and a small public API for inspecting and
temporarily overriding configuration in a safe, validated way.

Public API
----------
DEFAULTS : dict
    Live dictionary with the current configuration (do not mutate directly).
ritest_set(overrides)
    Validate and update selected keys in-place.
ritest_get(key=None)
    Read a single value or a shallow copy of all configuration.
ritest_reset(keys=None)
    Restore all or selected keys to import-time defaults.
ritest_config(overrides)
    Context manager for temporary overrides that automatically revert.
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
    # Main permutation controls
    "reps": 100,  # Number of permutations
    "seed": 23,  # Random seed for permutation draws
    "alpha": 0.05,  # Significance level for p-values and CIs
    # CI for permutation p-value
    "ci_method": "cp",  # 'cp' = Clopper–Pearson, 'normal' = Wald
    # Coefficient-CI mode
    # 'none'   : do not compute a coefficient CI
    # 'bounds' : compute only the 2-point coefficient CI
    # 'grid'   : compute the full (beta, p(beta)) band (and bounds)
    "ci_mode": "bounds",
    # Controls for generic statistic-based CIs when using stat_fn.
    # - If True  and ci_mode in {'bounds','grid'}: compute generic coefficient CI
    #   (grid is allowed but may be slow; a runtime warning may be shown).
    # - If False and using stat_fn: skip coefficient CI even if ci_mode != 'none'.
    # Ignored for fast linear models, where CIs are always available unless
    # ci_mode == 'none'.
    "coef_ci_generic": False,
    # Search/grid sizing for the coefficient CI (interpreted in SE units)
    "ci_range": 3.0,  # Half-range of the search grid in SE units
    "ci_step": 0.005,  # Grid step in SE units (used when ci_mode == 'grid')
    "ci_tol": 1e-4,  # Bisection tolerance as a fraction of one SE
    # Parallelism
    "n_jobs": -1,  # -1 = use all available CPU cores
    # Memory / chunking (used by run.py to bound memory)
    # Soft budget for the in-RAM permutation block. If the full
    # (reps × n × itemsize) permutation matrix would exceed this budget,
    # permutations are streamed in chunks using iter_permuted_matrix(...).
    # Default: 256 MiB, conservative on typical laptops and configurable.
    "perm_chunk_bytes": 256 * 1024 * 1024,
    # Minimum number of permutation rows per chunk to avoid many tiny blocks
    # when n is large. Takes effect only when chunking is enabled by the
    # memory budget above.
    "perm_chunk_min_rows": 64,
}

# Baseline snapshot used to support full or partial resets.
# Deep copy keeps this robust if nested structures are added later.
_BASE_DEFAULTS: Dict[str, Any] = deepcopy(DEFAULTS)

_ALLOWED_CI_METHODS = {"cp", "normal"}
_ALLOWED_CI_MODES = {"none", "bounds", "grid"}


def _validate_pair(key: str, val: Any) -> None:
    """Validate a (key, value) pair for the global configuration.

    Raises
    ------
    ValueError
        If the key is unknown or the value is not acceptable for that key.
    """
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
    """Update the global configuration in place.

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Mapping from config keys to new values. All keys must exist in DEFAULTS.

    Raises
    ------
    ValueError
        If overrides is not a mapping, or contains invalid keys or values.

    Notes
    -----
    This mutates global state. For temporary changes that automatically revert,
    prefer using ``ritest_config(...)``.
    """
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a mapping of {key: value}")

    # Validate first, then apply (all-or-nothing semantics)
    for k, v in overrides.items():
        _validate_pair(k, v)

    # Apply in-place
    DEFAULTS.update(overrides)


def ritest_get(key: Optional[str] = None) -> Any:
    """Read configuration values.

    Parameters
    ----------
    key : str or None, optional
        If None (default), return a shallow copy of the entire config.
        Otherwise, return the value for the requested key.

    Returns
    -------
    Any
        The value for the given key, or a shallow copy of DEFAULTS if key is None.

    Raises
    ------
    KeyError
        If a non-existent key is requested.
    """
    if key is None:
        # Shallow copy is sufficient because configuration values are scalars.
        return dict(DEFAULTS)
    if key not in DEFAULTS:
        raise KeyError(f"Unknown config key: {key!r}")
    return DEFAULTS[key]


def ritest_reset(keys: Optional[Iterable[str]] = None) -> None:
    """Reset configuration to import-time defaults.

    Parameters
    ----------
    keys : iterable of str or None, optional
        If None (default), reset all keys to baseline values.
        If an iterable is given, reset only those keys (all must exist).

    Raises
    ------
    ValueError
        If any requested key does not exist.

    Notes
    -----
    Resets are applied in place; the identity of DEFAULTS is preserved.
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
    """Context manager for temporary configuration overrides.

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Mapping from config keys to new values. All keys must be valid and
        values must pass validation.

    Yields
    ------
    None
        Execution continues inside the managed block with overrides applied.

    Notes
    -----
    The previous configuration is restored on exit, even if an exception is
    raised. The DEFAULTS object itself is preserved (only its contents change).
    """
    prev = dict(DEFAULTS)  # shallow snapshot of scalar values
    try:
        ritest_set(overrides)
        yield
    finally:
        # Clear and update in-place to preserve object identity.
        DEFAULTS.clear()
        DEFAULTS.update(prev)
