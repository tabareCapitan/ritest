"""
ritest.run
==========

High-level user API for *Randomization inference*.

The single public function ``ritest()`` orchestrates:

1. Argument validation          →  validation.validate_inputs
2. Observed-stat computation    →  FastOLS  *or*  user stat_fn
3. Permutation loop             →  shuffle.permute_assignment
4. P-value + CI for p-value     →  ci.pvalue_ci.pvalue_ci
5. Coefficient CI (bounds)      →  ci.coef_ci.*  (FAST for formula, generic otherwise)
6. Result packaging             →  results.RitestResult
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .ci.coef_ci import Alt, coef_ci_band_fast, coef_ci_bounds_fast, coef_ci_bounds_generic
from .ci.pvalue_ci import _PValCIMethod, pvalue_ci
from .config import DEFAULTS
from .engine.fast_ols import FastOLS
from .engine.shuffle import generate_permuted_matrix
from .results import RitestResult
from .validation import validate_inputs

__all__ = ["ritest"]


# ------------------------------------------------------------------ #
# Public entry-point
# ------------------------------------------------------------------ #
def ritest(  # noqa: C901 – orchestrator is naturally a bit long
    df: pd.DataFrame,
    *,
    permute_var: str,
    # Linear-model path
    formula: str | None = None,
    stat: str | None = None,
    # User-supplied statistic function
    stat_fn: Callable[[pd.DataFrame], float] | None = None,
    # Optional columns
    cluster: str | None = None,
    strata: str | None = None,
    weights: str | None = None,
    # Hypothesis direction
    alternative: Alt = "two-sided",
) -> RitestResult:
    """
    Run a randomisation test.

    Parameters
    ----------
    df : DataFrame
        Original data.
    permute_var : str
        Column to permute.
    formula / stat *or* stat_fn :
        Exactly one of the two modelling approaches.
    cluster, strata, weights : str, optional
        Columns specifying design features.
    alternative : {"two-sided", "left", "right"}, default "two-sided"
        Null-hypothesis tail.

    Returns
    -------
    RitestResult
        Self-contained result object.
    """

    # -------------------------------------------------------------- #
    # 0. Pull global settings  ------------------------------------- #
    cfg = DEFAULTS  # live dict – changes via ritest_set() are visible

    reps: int = cfg["reps"]
    seed: int = cfg["seed"]
    alpha: float = cfg["alpha"]
    ci_method: _PValCIMethod = cfg["ci_method"]
    ci_mode: str = cfg["ci_mode"]
    ci_range: float = cfg["ci_range"]
    ci_step: float = cfg["ci_step"]
    ci_tol: float = cfg["ci_tol"]
    coef_ci_generic: bool = cfg["coef_ci_generic"]

    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------- #
    # 1. Validate & preprocess inputs  ----------------------------- #
    v = validate_inputs(
        df,
        permute_var=permute_var,
        formula=formula,
        stat=stat,
        stat_fn=stat_fn,
        cluster=cluster,
        strata=strata,
        weights=weights,
        alternative=alternative,
        alpha=alpha,
        ci_method=ci_method,
        ci_mode=ci_mode,
        ci_range=ci_range,
        ci_step=ci_step,
        ci_tol=ci_tol,
        coef_ci_generic=coef_ci_generic,
    )

    # -------------------------------------------------------------- #
    # 2. Observed statistic (and helpers for linear path) ---------- #
    if v.stat_fn is None:  # FORMULA PATH
        ols = FastOLS(
            v.y,
            v.X,
            v.treat_idx,
            weights=v.weights,
            cluster=v.cluster,
        )
        obs_stat = ols.beta_hat  # β̂
        se_obs = ols.se  # robust SE  (needed for CI width)
        c_vec = ols.c_vector  # noqa: F841 # cached cᵀ  (for fast perms & CI)
        K = ols.K  # cᵀ T      (shift constant)
        z_obs = obs_stat  # since β̂ = cᵀ y
        linear_model = True
    else:  # stat_fn PATH
        obs_stat = v.stat_fn(df)  # already warmed-up in validation
        se_obs = np.nan
        c_vec = None  # noqa: F841
        K = None  # noqa: F841
        z_obs = np.nan  # noqa: F841
        linear_model = False

    # Warn user if stat_fn + grid CI will be slow
    if not linear_model and ci_mode == "grid":
        grid_size = int((2 * ci_range) / ci_step)
        est_sec = v.warmup_time * grid_size * reps
        if est_sec > 10:
            warnings.warn(
                f"CI bands for stat_fn may take ~{est_sec:.1f} sec "
                f"(warmup {v.warmup_time:.3f}s × grid={grid_size} × reps={reps})."
            )

    # -------------------------------------------------------------- #
    # 3. Permutation statistics  ----------------------------------- #
    T_perms = generate_permuted_matrix(
        v.T,
        reps,
        cluster=v.cluster,
        strata=v.strata,
        rng=rng,
    )

    perm_stats = np.empty(reps, dtype=float)

    for r in range(reps):
        T_perm = T_perms[r]

        # 3b. compute statistic on permuted data
        if v.stat_fn is None:
            X_perm = v.X.copy()
            X_perm[:, v.treat_idx] = T_perm
            beta_r = FastOLS(
                v.y,
                X_perm,
                v.treat_idx,
                weights=v.weights,
                cluster=v.cluster,
            ).beta_hat
            perm_stats[r] = beta_r
        else:
            df_perm = df.copy()
            df_perm[permute_var] = T_perm
            perm_stats[r] = v.stat_fn(df_perm)

    # -------------------------------------------------------------- #
    # 4. P-value and its CI  --------------------------------------- #
    if alternative == "two-sided":
        extreme = np.abs(perm_stats) >= abs(obs_stat)
    elif alternative == "right":
        extreme = perm_stats >= obs_stat
    else:  # "left"
        extreme = perm_stats <= obs_stat

    c = int(extreme.sum())
    p_val = c / reps
    p_ci = pvalue_ci(c, reps, alpha=alpha, method=ci_method)

    # -------------------------------------------------------------- #
    # 5. Coefficient CI bounds  ------------------------------------ #
    coef_ci_bounds: Optional[tuple[float, float]]
    band_valid_linear = True

    if linear_model:
        # ---------- FAST bounds via cᵀy trick ---------------------- #
        assert K is not None
        coef_ci_bounds = coef_ci_bounds_fast(
            beta_obs=obs_stat,
            z_obs=z_obs,
            z_perm=perm_stats,  # each β_r is already cᵀ y_perm
            K=K,
            se=se_obs,
            alpha=alpha,
            ci_range=ci_range,
            tol=ci_tol,
            alternative=alternative,
        )
    else:
        # ---------- Generic fallback (stat_fn) --------------------- #
        def _runner(beta0: float) -> float:
            shifted = obs_stat - beta0
            extreme_shift = (
                np.abs(perm_stats - beta0) >= abs(shifted)
                if alternative == "two-sided"
                else (
                    perm_stats - beta0 >= shifted
                    if alternative == "right"
                    else perm_stats - beta0 <= shifted
                )
            )
            return extreme_shift.mean()

        coef_ci_bounds = coef_ci_bounds_generic(
            obs_stat,
            runner=_runner,
            alpha=alpha,
            ci_range=ci_range,
            ci_step=ci_step,
            se=se_obs,  # unused but required by signature
            alternative=alternative,
        )
        band_valid_linear = False  # bands unsafe for non-linear stats

    # -------------------------------------------------------------- #
    # 6. CI band (β grid)  ----------------------------------------- #
    coef_ci_band = None
    if ci_mode == "grid":
        if linear_model:
            # ---------- FAST band via cᵀy trick -------------------- #
            assert K is not None
            coef_ci_band = coef_ci_band_fast(
                beta_obs=obs_stat,
                z_obs=z_obs,
                z_perm=perm_stats,  # each row already β_r = cᵀ y_perm
                K=K,
                se=se_obs,
                ci_range=ci_range,
                ci_step=ci_step,
                alternative=alternative,
            )
            band_valid_linear = True
        else:
            # ---------- Generic band (stat_fn) --------------------- #
            def _runner(beta0: float) -> float:
                shifted = obs_stat - beta0
                extreme_shift = (
                    np.abs(perm_stats - beta0) >= abs(shifted)
                    if alternative == "two-sided"
                    else (
                        perm_stats - beta0 >= shifted
                        if alternative == "right"
                        else perm_stats - beta0 <= shifted
                    )
                )
                return extreme_shift.mean()

            grid = (
                np.arange(
                    -ci_range * se_obs,  # se_obs is nan but unused here
                    ci_range * se_obs + 1e-12,
                    ci_step * (1.0 if np.isnan(se_obs) else se_obs),
                )
                + obs_stat
            )
            pvals = np.array([_runner(b0) for b0 in grid])
            coef_ci_band = (grid, pvals)
            band_valid_linear = False  # warn that band isn't guaranteed linear

    # -------------------------------------------------------------- #
    # 7. Package and return result  -------------------------------- #
    res = RitestResult(
        obs_stat=obs_stat,
        coef_ci_bounds=coef_ci_bounds,
        pval=p_val,
        pval_ci=p_ci,
        reps=reps,
        c=c,
        alternative=alternative,
        stratified=v.has_strata,
        clustered=v.has_cluster,
        weights=v.weights is not None,
        coef_ci_band=coef_ci_band,
        band_valid_linear=band_valid_linear,
        settings=dict(alpha=alpha),
        perm_stats=perm_stats,
    )

    return res
