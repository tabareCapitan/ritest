from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, cast

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


def _coerce_n_jobs(val: int | None) -> int:
    if val is None:
        return 1
    if val == -1:
        return max(os.cpu_count() or 1, 1)
    try:
        v = int(val)
    except Exception:
        return 1
    return max(v, 1)


def _coerce_ci_method(x: str | _PValCIMethod | None, fallback: _PValCIMethod) -> _PValCIMethod:
    """Normalize user input to core's accepted labels: 'cp' or 'normal'."""
    if x is None:
        return fallback
    s = str(x).strip().lower()
    if s in {"cp", "clopper-pearson", "clopperpearson", "clopper", "exact"}:
        return "cp"  # type: ignore[return-value]
    if s in {"normal", "wald"}:
        return "normal"  # type: ignore[return-value]
    return fallback


def ritest(  # noqa: C901
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
    # --- public controls (override DEFAULTS when provided) ---
    reps: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
    ci_method: str | _PValCIMethod | None = None,
    ci_mode: str | None = None,  # "none" | "bounds" | "grid"
    ci_range: float | None = None,
    ci_step: float | None = None,
    coef_ci_generic: bool | None = None,  # relevant only when stat_fn and grid
    n_jobs: int | None = None,
    # prebuilt permutations (matrix of permuted T labels)
    permutations: np.ndarray | None = None,
) -> RitestResult:
    """
    Run a randomisation test.

    Exactly one of (`formula`, `stat`) or (`stat_fn`) must be provided.
    Public controls override DEFAULTS when specified.
    """
    t0 = time.perf_counter()

    # 0) Controls: pull from DEFAULTS, allow public overrides
    cfg = DEFAULTS
    reps = int(cfg["reps"]) if reps is None else int(reps)
    seed = int(cfg["seed"]) if seed is None else int(seed)
    alpha = float(cfg["alpha"]) if alpha is None else float(alpha)
    ci_method = _coerce_ci_method(ci_method, fallback=cfg["ci_method"])
    ci_mode = str(cfg["ci_mode"]) if ci_mode is None else str(ci_mode)
    ci_range = float(cfg["ci_range"]) if ci_range is None else float(ci_range)
    ci_step = float(cfg["ci_step"]) if ci_step is None else float(ci_step)
    ci_tol = float(cfg["ci_tol"])  # reserved for future use
    coef_ci_generic = (
        bool(cfg["coef_ci_generic"]) if coef_ci_generic is None else bool(coef_ci_generic)
    )
    n_jobs = _coerce_n_jobs(int(cfg.get("n_jobs", 1)) if n_jobs is None else int(n_jobs))

    rng = np.random.default_rng(seed)

    # 1) Validate & preprocess (enforces binary treatment and returns T as int8 0/1)
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

    # 2) Observed statistic
    linear_model = v.stat_fn is None
    if linear_model:
        ols = FastOLS(
            v.y, v.X, v.treat_idx, weights=v.weights, cluster=v.cluster, compute_vcov=True
        )
        obs_stat = float(ols.beta_hat)
        se_obs = float(ols.se)
        K_obs = float(ols.K)
        t_metric_lin = ols.t_metric  # branch-local, aligns with c_vector metric
    else:
        stat_fn_local = cast(Callable[[pd.DataFrame], float], v.stat_fn)
        obs_stat = float(stat_fn_local(df))
        se_obs = float("nan")
        K_obs = float("nan")
        t_metric_lin = None  # type: ignore[assignment]

    # Warn for potentially slow generic grid band when opted-in
    if (not linear_model) and (ci_mode == "grid") and coef_ci_generic:
        grid_size = max(int(round((2.0 * ci_range) / max(ci_step, 1e-12))) + 1, 1)
        est_sec = float(v.warmup_time) * grid_size * reps
        if est_sec > 10:
            warnings.warn(
                f"CI bands for stat_fn may take ~{est_sec:.1f} sec "
                f"(warmup {v.warmup_time:.3f}s × grid≈{grid_size} × reps={reps})."
            )

    # 3) Permutations (deterministic or prebuilt)
    if permutations is not None:
        # Expect shape (reps, n), where rows are permuted T-label vectors
        perms = np.asarray(permutations)
        if perms.ndim != 2 or perms.shape[1] != v.T.shape[0]:
            raise ValueError(
                f"permutations must have shape (reps, n={v.T.shape[0]}), got {perms.shape}"
            )
        T_perms = perms
        reps = int(T_perms.shape[0])  # lock reps to provided perms
    else:
        # NOTE: v.T is int8 0/1 for memory-compact permutations; the engine
        # preserves dtype, so T_perms is also int8. Assigning into float64 X
        # casts on write without large intermediate allocations.
        T_perms = generate_permuted_matrix(v.T, reps, cluster=v.cluster, strata=v.strata, rng=rng)

    perm_stats = np.empty(reps, dtype=np.float64)

    # 4) Evaluate statistic over permutations
    if linear_model:
        K_perm_local = np.empty(reps, dtype=np.float64)

        def _fit_one(args) -> tuple[int, float, float]:
            r, T_perm = args
            Xp = v.X.copy()
            Xp[:, v.treat_idx] = T_perm  # int8 → float cast on assignment
            ols_r = FastOLS(
                v.y, Xp, v.treat_idx, weights=v.weights, cluster=v.cluster, compute_vcov=False
            )
            beta_r = float(ols_r.beta_hat)
            Kr = float(ols_r.c_vector @ t_metric_lin)  # type: ignore[arg-type]
            return r, beta_r, Kr

        if n_jobs == 1:
            X_work = v.X.copy()
            for r in range(reps):
                X_work[:, v.treat_idx] = T_perms[r]  # int8 → float cast on assignment
                ols_r = FastOLS(
                    v.y,
                    X_work,
                    v.treat_idx,
                    weights=v.weights,
                    cluster=v.cluster,
                    compute_vcov=False,
                )
                perm_stats[r] = float(ols_r.beta_hat)
                K_perm_local[r] = float(ols_r.c_vector @ t_metric_lin)  # type: ignore[arg-type]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                for r, beta_r, Kr in ex.map(_fit_one, enumerate(T_perms)):
                    perm_stats[r] = beta_r
                    K_perm_local[r] = Kr
    else:
        stat_fn_local = cast(Callable[[pd.DataFrame], float], v.stat_fn)

        def _eval_one(args) -> tuple[int, float]:
            r, T_perm = args
            dfp = df.copy(deep=False)
            dfp[permute_var] = T_perm  # pandas will upcast to float when needed
            return r, float(stat_fn_local(dfp))

        if n_jobs == 1:
            for r in range(reps):
                dfp = df.copy(deep=False)
                dfp[permute_var] = T_perms[r]
                perm_stats[r] = float(stat_fn_local(dfp))
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                for r, z in ex.map(_eval_one, enumerate(T_perms)):
                    perm_stats[r] = z

    # 5) P-value + CI for p
    if alternative == "two-sided":
        extreme = np.abs(perm_stats) >= abs(obs_stat)
    elif alternative == "right":
        extreme = perm_stats >= obs_stat
    else:
        extreme = perm_stats <= obs_stat

    c = int(extreme.sum())
    p_val = c / reps
    p_ci = pvalue_ci(c, reps, alpha=alpha, method=ci_method)

    # 6) (Optionally) compute coefficient CI artifacts
    coef_ci_bounds: Optional[tuple[float, float]] = None
    coef_ci_band = None
    band_valid_linear = linear_model  # True for linear path; False for generic

    if ci_mode != "none":
        if linear_model:
            if ci_mode in {"bounds", "grid"}:
                coef_ci_bounds = coef_ci_bounds_fast(
                    beta_obs=obs_stat,
                    beta_perm=perm_stats,
                    K_obs=K_obs,  # type: ignore[arg-type]
                    K_perm=K_perm_local,  # type: ignore[arg-type]
                    se=se_obs,
                    alpha=alpha,
                    ci_range=ci_range,
                    ci_step=ci_step,
                    alternative=alternative,
                )
            if ci_mode == "grid":
                coef_ci_band = coef_ci_band_fast(
                    beta_obs=obs_stat,
                    beta_perm=perm_stats,
                    K_obs=K_obs,  # type: ignore[arg-type]
                    K_perm=K_perm_local,  # type: ignore[arg-type]
                    se=se_obs,
                    ci_range=ci_range,
                    ci_step=ci_step,
                    alternative=alternative,
                )
        else:
            if coef_ci_generic:

                def _runner(beta0: float) -> float:
                    shifted = obs_stat - beta0
                    if alternative == "two-sided":
                        return (np.abs(perm_stats - beta0) >= abs(shifted)).mean()
                    elif alternative == "right":
                        return (perm_stats - beta0 >= shifted).mean()
                    else:
                        return (perm_stats - beta0 <= shifted).mean()

                se_scale = float(np.nanstd(perm_stats, ddof=1)) if reps >= 2 else 1.0
                if not (se_scale > 0.0 and np.isfinite(se_scale)):
                    se_scale = 1.0

                if ci_mode in {"bounds", "grid"}:
                    coef_ci_bounds = coef_ci_bounds_generic(
                        obs_stat,
                        runner=_runner,
                        alpha=alpha,
                        ci_range=ci_range,
                        ci_step=ci_step,
                        se=se_scale,
                        alternative=alternative,
                    )
                if ci_mode == "grid":
                    grid = (
                        np.arange(
                            -ci_range * se_scale, ci_range * se_scale + 1e-12, ci_step * se_scale
                        )
                        + obs_stat
                    )
                    pvals = np.array([_runner(b0) for b0 in grid], dtype=np.float64)
                    coef_ci_band = (grid, pvals)
                    band_valid_linear = False

    # 7) STRICT GATING of outputs before building result
    _bounds = coef_ci_bounds
    _band = coef_ci_band
    _band_valid_linear = bool(band_valid_linear)

    if ci_mode == "none":
        _bounds = None
        _band = None
        _band_valid_linear = False
    elif ci_mode == "bounds":
        _band = None
        # KEEP the linear flag: True for linear, False for generic
    elif ci_mode == "grid":
        _bounds = None
        if (stat_fn is not None) and (not coef_ci_generic):
            _band = None
            _band_valid_linear = False
    else:
        raise ValueError(f"Unknown ci_mode: {ci_mode!r}")

    # 8) Package results
    settings: Dict[str, object] = {
        "alpha": alpha,
        "seed": seed,
        "reps": reps,
        "ci_method": ci_method,
        "ci_mode": ci_mode,
        "ci_range": ci_range,
        "ci_step": ci_step,
        "alternative": alternative,
        "n_jobs": n_jobs,
        "coef_ci_generic": coef_ci_generic,
        "runtime_sec": time.perf_counter() - t0,
    }

    res = RitestResult(
        obs_stat=float(obs_stat),
        coef_ci_bounds=_bounds,
        pval=float(p_val),
        pval_ci=p_ci,
        reps=reps,
        c=c,
        alternative=alternative,
        stratified=v.has_strata,
        clustered=v.has_cluster,
        weights=v.weights is not None,
        coef_ci_band=_band,
        band_valid_linear=_band_valid_linear,
        settings=settings,
        perm_stats=perm_stats,
    )
    return res
