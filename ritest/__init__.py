"""Public interface for the ritest package.

This module exposes a stable, user-facing API:

- `ritest`: high-level entrypoint with friendly defaults.
- `RitestResult`: container for outputs and basic presentation helpers.
- Config helpers: `ritest_set`, `ritest_get`, `ritest_reset`, `ritest_config`.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

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


def ritest(
    *,
    df,
    permute_var: str,
    # linear path
    formula: Optional[str] = None,
    stat: Optional[str] = None,
    # generic path
    stat_fn: Optional[Callable[[Any], float]] = None,
    # test controls (None ⇒ use config.DEFAULTS)
    alternative: Alt = "two-sided",
    reps: Optional[int] = None,
    alpha: Optional[float] = None,
    ci_method: Optional[CiMethod] = None,
    ci_mode: Optional[CiMode] = None,
    coef_ci_generic: Optional[bool] = None,
    # infra
    n_jobs: Optional[int] = None,
    seed: Optional[int] = None,
    # design
    weights: Optional[str] = None,
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    # optional prebuilt perms
    permutations: Any | None = None,
    # optional band grid hints (match core: scalar half-range in SE units)
    ci_range: Optional[float] = None,
    ci_step: Optional[float] = None,
) -> RitestResult:
    """
    Public entrypoint for randomization inference.

    Either provide a linear model via `formula` and `stat`, or a custom
    statistic via `stat_fn`, but not both.

    All configuration knobs default to values from `ritest.config.DEFAULTS`
    when left as None.
    """
    # Enforce exactly one of (stat, stat_fn)
    has_stat = stat is not None
    has_stat_fn = stat_fn is not None
    if has_stat == has_stat_fn:
        raise ValueError(
            "Provide exactly one of `stat` (linear model path) or "
            "`stat_fn` (generic statistic path)."
        )

    return _core_ritest(
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
