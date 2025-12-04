from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from .config import ritest_config, ritest_get, ritest_reset, ritest_set
from .results import RitestResult
from .run import ritest as _core_ritest  # internal entrypoint

__all__ = [
    "ritest",
    "RitestResult",
    "ritest_set",
    "ritest_get",
    "ritest_reset",
    "ritest_config",
]

Alt = Literal["two-sided", "left", "right"]
CiMode = Literal["none", "bounds", "grid"]
CiMethod = Literal["clopper-pearson", "normal"]

# Canonical public kwargs -> potential core names
_SYN = {
    "df": ("df", "data", "dataset"),
    "formula": ("formula",),
    "permute_var": ("permute_var", "treat", "treatment", "assign", "perm_var"),
    "stat": ("stat", "coef", "target"),
    "stat_fn": ("stat_fn", "statfun", "stat_function"),
    "alternative": ("alternative", "tail"),
    "alpha": ("alpha",),
    "ci_method": ("ci_method", "pval_ci_method", "pvalue_ci_method"),
    "ci_mode": ("ci_mode", "coef_ci_mode"),  # prefer true mode param if core has it
    "coef_ci_generic": ("coef_ci_generic", "generic_ci", "generic"),
    "reps": ("reps", "B", "n_reps", "nperm", "n_permutations", "n_perms"),
    "n_jobs": ("n_jobs", "jobs", "threads"),
    "seed": ("seed", "random_state", "rng_seed"),
    "weights": ("weights", "w"),
    "strata": ("strata", "block", "blocks"),
    "cluster": ("cluster", "clusters", "cls"),
    "permutations": ("permutations", "perm_matrix", "perm_labels"),
    "ci_range": ("ci_range",),
    "ci_step": ("ci_step",),
}


def _translate_kwargs(core, **kw: Any) -> Dict[str, Any]:
    """
    Map canonical public kwargs onto the core entrypoint's parameters.

    Pass 1: keep any kw the core already accepts (so 'df' is preserved).
    Pass 2: translate remaining logical keys via synonyms.
    Pass 3: special handling for ci_mode -> ('ci_mode' string) or ('coef_ci'/'ci' bool).
    """
    params = set(inspect.signature(core).parameters.keys())
    out: Dict[str, Any] = {}

    # Pass-through
    for k, v in kw.items():
        if k in params:
            out[k] = v

    # Synonym translation
    for logical, choices in _SYN.items():
        if logical not in kw:
            continue
        if any(name in out for name in choices):
            continue
        for actual in choices:
            if actual in params:
                out[actual] = kw[logical]
                break

    # Special: ci_mode fallback. NEVER pass a string to a boolean slot.
    if "ci_mode" in kw:
        want = kw["ci_mode"]
        has_mode = "ci_mode" in params or "coef_ci_mode" in params
        if not has_mode:
            if "coef_ci" in params and "coef_ci" not in out:
                out["coef_ci"] = want != "none"
            elif "ci" in params and "ci" not in out:
                out["ci"] = want != "none"

    return out


def ritest(
    *,
    df,
    permute_var: str,
    # linear path
    formula: Optional[str] = None,
    stat: Optional[str] = None,
    # generic path
    stat_fn: Optional[Callable[[Any], float]] = None,
    # test controls
    alternative: Alt = "two-sided",
    reps: int = 999,
    alpha: float = 0.05,
    ci_method: CiMethod = "clopper-pearson",
    ci_mode: CiMode = "none",  # "none" | "bounds" | "grid"
    coef_ci_generic: bool = False,  # only relevant for stat_fn + grid
    # infra
    n_jobs: int = 1,
    seed: Optional[int] = None,
    # design
    weights: Optional[str] = None,
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    # optional prebuilt perms
    permutations=None,
    # optional band grid hints
    ci_range: Optional[Tuple[float, float]] = None,
    ci_step: Optional[float] = None,
) -> RitestResult:
    """Public entrypoint. Provide exactly one of `stat` (linear) or `stat_fn` (generic)."""
    if (stat is None) == (stat_fn is None):
        raise ValueError("Provide exactly one of 'stat' (linear) or 'stat_fn' (generic).")

    public_kwargs = dict(
        df=df,
        permute_var=permute_var,
        formula=formula,
        stat=stat,
        stat_fn=stat_fn,
        alternative=alternative,
        reps=reps,
        alpha=alpha,
        ci_method=ci_method,
        ci_mode=ci_mode,
        coef_ci_generic=coef_ci_generic,
        n_jobs=n_jobs,
        seed=seed,
        weights=weights,
        strata=strata,
        cluster=cluster,
        permutations=permutations,
        ci_range=ci_range,
        ci_step=ci_step,
    )
    core_kwargs = _translate_kwargs(_core_ritest, **public_kwargs)
    return _core_ritest(**core_kwargs)
