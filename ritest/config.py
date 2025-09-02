"""Configuration handling for ritest.

Defines the DEFAULTS dict that stores all global configuration values,
and the public interface `ritest_set()` to safely override them.

Notes:
- Only recognized keys can be overridden.
- Type validation and value checks are enforced.
- DEFAULTS is a live dictionary used internally throughout the package.
"""

# Global config used by all internal modules
DEFAULTS = {
    # --- Main permutation settings ---
    "reps": 100,  # Number of permutations
    "seed": 23,  # Random seed (default: for MJ fans)
    "alpha": 0.05,  # Significance level for p-value and CI
    # --- CI for permutation p-value ---
    "ci_method": "cp",  # 'cp' = Clopper–Pearson, 'normal' = Wald
    # --- Coefficient-CI controls ---
    "ci_mode": "bounds",  # 'bounds'  (fast 2-bound CI)
    # 'grid'    (full (β, p(β)) band – *opt-in*)
    "coef_ci_generic": False,  # if True and model is NOT fast-linear
    # → fall back to slow generic bounds/band
    "ci_range": 3.0,  # search range in SE units for coef CI
    "ci_step": 0.005,  # step for grid (β) in SE units – used when ci_mode=='grid'
    "ci_tol": 1e-4,  # bisection tol (as *SE* fractions)
    # --- Parallelism ---
    "n_jobs": -1,  # -1 = use all available CPU cores
}


def ritest_set(overrides: dict) -> None:
    """
    Update the global configuration.

    Parameters
    ----------
    overrides : dict
        A dictionary of config keys and new values.

    Raises
    ------
    ValueError
        If unknown keys or invalid values are passed.
    """

    allowed_ci_methods = {"cp", "normal"}
    allowed_ci_modes = {"bounds", "grid"}

    for key, val in overrides.items():
        if key not in DEFAULTS:
            raise ValueError(f"Invalid config key: '{key}'")

        # Per-key validation
        if key == "reps":
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"reps must be a positive integer (got {val})")
        elif key == "seed":
            if not isinstance(val, int):
                raise ValueError(f"seed must be an integer (got {val})")
        elif key == "alpha":
            if not (0 < val < 1):
                raise ValueError(f"alpha must be between 0 and 1 (got {val})")
        elif key == "ci_method":
            if val not in allowed_ci_methods:
                raise ValueError(f"ci_method must be one of {allowed_ci_methods} (got {val})")
        elif key == "ci_mode":
            if val not in allowed_ci_modes:
                raise ValueError(f"ci_mode must be one of {allowed_ci_modes} (got {val})")
        elif key in {"ci_range", "ci_step"}:
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{key} must be a positive number (got {val})")
        elif key == "coef_ci_generic":
            if not isinstance(val, bool):
                raise ValueError("coef_ci_generic must be True/False")
        elif key == "ci_tol":
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError("ci_tol must be a positive number")

        elif key == "n_jobs":
            if not isinstance(val, int):
                raise ValueError(f"n_jobs must be an integer (got {val})")

        # If all checks pass, update value
        DEFAULTS[key] = val
