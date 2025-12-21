"""Core randomization-inference engine for ritest.

This module contains the internal `ritest()` implementation used by the
public API. It coordinates:

- configuration (DEFAULTS and user overrides),
- validation and preprocessing,
- permutation generation (full matrix or streamed),
- model evaluation (FastOLS or user-supplied stat_fn),
- p-value and p-value CI calculation,
- optional coefficient CI bounds/band,
- packaging results into `RitestResult`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .ci.coef_ci import (
    Alt,
    coef_ci_band_fast,
    coef_ci_bounds_fast,
)
from .ci.pvalue_ci import _PValCIMethod, pvalue_ci
from .config import DEFAULTS, _canonical_ci_method

# FastOLS is the linear-path engine; NUMBA_OK tells us if its kernels are JITed.
try:
    from .engine.fast_ols import NUMBA_OK as FAST_OLS_NUMBA_OK  # type: ignore[attr-defined]
    from .engine.fast_ols import FastOLS
except Exception:  # pragma: no cover - defensive
    from .engine.fast_ols import FastOLS  # type: ignore[no-redef]

    FAST_OLS_NUMBA_OK = False  # conservative fallback
# Permutation providers: eager (full matrix) and streaming (chunked)
from .engine.shuffle import generate_permuted_matrix, iter_permuted_matrix
from .results import RitestResult
from .validation import validate_inputs

__all__ = ["ritest"]


def _coerce_n_jobs(val: int | None) -> int:
    """Normalise `n_jobs` to a sensible positive integer."""
    if val is None:
        return 1
    if val == -1:
        return max(os.cpu_count() or 1, 1)
    try:
        v = int(val)
    except Exception:
        return 1
    return max(v, 1)


def _coerce_ci_method(
    x: str | _PValCIMethod | None, fallback: _PValCIMethod
) -> _PValCIMethod:
    """Normalise user input to canonical labels."""
    if x is None:
        return fallback
    try:
        return _canonical_ci_method(x)  # type: ignore[return-value]
    except ValueError:
        return fallback


def _bytes_per_row(n_obs: int, label_itemsize: int) -> int:
    """
    Estimate bytes required per *row* of the permutation block.

    Rules
    -----
    - If FastOLS kernels are Numba-JITed (`FAST_OLS_NUMBA_OK`), labels are
      consumed as-is (int8) ⇒ 1 byte per entry is enough.
    - Otherwise, NumPy fallbacks may cast to float64 internally; assume
      8 bytes per entry for a safe upper bound.
    - Always respect the input label itemsize so we never underestimate.
    """
    per_entry = 1 if FAST_OLS_NUMBA_OK else 8
    # In case label_itemsize is larger than 1 (user-provided permutations),
    # also respect the input size to avoid underestimation.
    per_entry = max(per_entry, label_itemsize)
    return n_obs * per_entry


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
    ci_mode: str | None = None,  # "none" | "bounds" | "band"
    ci_range: float | None = None,
    ci_step: float | None = None,
    n_jobs: int | None = None,
    # prebuilt permutations (matrix of permuted T labels)
    permutations: np.ndarray | None = None,
) -> RitestResult:
    """
    Core randomisation test.

    Exactly one of (`formula`, `stat`) or (`stat_fn`) must be provided.
    Any explicit control provided here overrides the corresponding value
    in `config.DEFAULTS`.
    """

    # 0) Controls: pull from DEFAULTS, allow public overrides
    cfg = DEFAULTS
    reps = int(cfg["reps"]) if reps is None else int(reps)
    seed = int(cfg["seed"]) if seed is None else int(seed)
    alpha = float(cfg["alpha"]) if alpha is None else float(alpha)
    cfg_ci_method = _canonical_ci_method(cfg["ci_method"])
    ci_method = _coerce_ci_method(ci_method, fallback=cfg_ci_method)
    ci_mode = str(cfg["ci_mode"]) if ci_mode is None else str(ci_mode)
    ci_range = float(cfg["ci_range"]) if ci_range is None else float(ci_range)
    ci_step = float(cfg["ci_step"]) if ci_step is None else float(ci_step)
    ci_tol = float(cfg["ci_tol"])  # reserved for future use
    n_jobs = _coerce_n_jobs(
        int(cfg.get("n_jobs", 1)) if n_jobs is None else int(n_jobs)
    )
    # Memory/chunking knobs (soft budget)
    perm_chunk_bytes = int(cfg.get("perm_chunk_bytes", 256 * 1024 * 1024))
    perm_chunk_min_rows = int(cfg.get("perm_chunk_min_rows", 64))

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
    )

    # 2) Observed statistic
    linear_model = v.stat_fn is None
    need_coef_ci = linear_model and ci_mode != "none"
    if linear_model:
        ols = FastOLS(
            v.y,
            v.X,
            v.treat_idx,
            weights=v.weights,
            cluster=v.cluster,
            compute_vcov=True,
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

    # 3) Allocate outputs (small, O(reps))
    perm_stats = np.empty(reps, dtype=np.float64)
    K_perm_local = np.empty(reps, dtype=np.float64) if need_coef_ci else None

    # Helper workers (capture `v`, `t_metric_lin` by closure)
    def _solve_linear_perm(
        T_perm: np.ndarray,
        X_template: np.ndarray | None = None,
        *,
        reuse_template: bool = False,
    ) -> FastOLS:
        """Solve linear model for a permuted treatment vector."""
        if X_template is None:
            Xp = v.X.copy()
        elif reuse_template:
            Xp = X_template
        else:
            Xp = X_template.copy()
        Xp[:, v.treat_idx] = T_perm  # int8 → float cast on assignment
        return FastOLS(
            v.y,
            Xp,
            v.treat_idx,
            weights=v.weights,
            cluster=v.cluster,
            compute_vcov=False,
        )

    def _fit_one_beta(args: Tuple[int, np.ndarray]) -> Tuple[int, float]:
        """Linear-path worker: returns only beta_r for ci_mode='none' cases."""
        r_abs, T_perm = args
        ols_r = _solve_linear_perm(T_perm)
        return r_abs, float(ols_r.beta_hat)

    def _fit_one_beta_K(args: Tuple[int, np.ndarray]) -> Tuple[int, float, float]:
        """Linear-path worker: returns beta_r and Kr for CI computation."""
        r_abs, T_perm = args
        ols_r = _solve_linear_perm(T_perm)
        beta_r = float(ols_r.beta_hat)
        Kr = float(ols_r.c_vector @ t_metric_lin)  # type: ignore[arg-type]
        return r_abs, beta_r, Kr

    def _eval_one(args: Tuple[int, np.ndarray]) -> Tuple[int, float]:
        """Generic-path worker: (abs_index, T_perm_row) -> (abs_index, stat_r)."""
        r_abs, T_perm = args
        dfp = df.copy(deep=False)
        dfp[permute_var] = T_perm  # pandas will upcast to float when needed
        return r_abs, float(stat_fn_local(dfp))  # type: ignore[arg-type]

    # 4) Permutations: prebuilt, eager, or chunked (deterministic in all cases)
    if permutations is not None:
        # Expect shape (reps, n), where rows are permuted T-label vectors
        perms = np.asarray(permutations)
        if perms.ndim != 2 or perms.shape[1] != v.T.shape[0]:
            raise ValueError(
                f"permutations must have shape (reps, n={v.T.shape[0]}), got {perms.shape}"
            )
        if perms.shape[0] != reps:
            # Honor the supplied matrix size as the true reps
            reps = int(perms.shape[0])
            perm_stats = np.empty(reps, dtype=np.float64)
            if need_coef_ci:
                K_perm_local = np.empty(reps, dtype=np.float64)
        # Process rows directly (no chunking for user-supplied matrix)
        if linear_model:
            if n_jobs == 1:
                X_work = v.X.copy()
                for r in range(reps):
                    ols_r = _solve_linear_perm(
                        perms[r], X_template=X_work, reuse_template=True
                    )
                    perm_stats[r] = float(ols_r.beta_hat)
                    if need_coef_ci:
                        K_perm_local[r] = float(ols_r.c_vector @ t_metric_lin)  # type: ignore[index]
            else:
                worker = _fit_one_beta_K if need_coef_ci else _fit_one_beta
                with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    if need_coef_ci:
                        for r_abs, beta_r, Kr in ex.map(
                            worker, ((r, perms[r]) for r in range(reps))
                        ):
                            perm_stats[r_abs] = beta_r
                            K_perm_local[r_abs] = Kr  # type: ignore[index]
                    else:
                        for r_abs, beta_r in ex.map(
                            worker, ((r, perms[r]) for r in range(reps))
                        ):
                            perm_stats[r_abs] = beta_r
        else:
            if n_jobs == 1:
                for r in range(reps):
                    dfp = df.copy(deep=False)
                    dfp[permute_var] = perms[r]
                    perm_stats[r] = float(stat_fn_local(dfp))  # type: ignore[arg-type]
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    for r_abs, z in ex.map(
                        lambda a: _eval_one(a), ((r, perms[r]) for r in range(reps))
                    ):
                        perm_stats[r_abs] = z
    else:
        # Decide whether to allocate the full (reps, n) matrix or stream in chunks.
        n_obs = v.T.shape[0]
        itemsize = int(v.T.dtype.itemsize)  # int8 => 1
        bpr = _bytes_per_row(n_obs, itemsize)  # conservative when Numba is unavailable
        full_bytes = reps * bpr

        if full_bytes <= perm_chunk_bytes:
            # Eager path: build full matrix (current behavior)
            T_perms = generate_permuted_matrix(
                v.T, reps, cluster=v.cluster, strata=v.strata, rng=rng
            )
            if linear_model:
                if n_jobs == 1:
                    X_work = v.X.copy()
                    for r in range(reps):
                        ols_r = _solve_linear_perm(
                            T_perms[r], X_template=X_work, reuse_template=True
                        )
                        perm_stats[r] = float(ols_r.beta_hat)
                        if need_coef_ci:
                            K_perm_local[r] = float(ols_r.c_vector @ t_metric_lin)  # type: ignore[index]
                else:
                    worker = _fit_one_beta_K if need_coef_ci else _fit_one_beta
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        if need_coef_ci:
                            for r, beta_r, Kr in ex.map(
                                worker, ((r, T_perms[r]) for r in range(reps))
                            ):
                                perm_stats[r] = beta_r
                                K_perm_local[r] = Kr  # type: ignore[index]
                        else:
                            for r, beta_r in ex.map(
                                worker, ((r, T_perms[r]) for r in range(reps))
                            ):
                                perm_stats[r] = beta_r
            else:
                if n_jobs == 1:
                    for r in range(reps):
                        dfp = df.copy(deep=False)
                        dfp[permute_var] = T_perms[r]
                        perm_stats[r] = float(stat_fn_local(dfp))  # type: ignore[arg-type]
                else:
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        for r, z in ex.map(
                            _eval_one, ((r, T_perms[r]) for r in range(reps))
                        ):
                            perm_stats[r] = z
        else:
            # Streaming path: generate blocks with bounded memory and process each in turn.
            # Choose chunk_rows from budget; enforce a sensible minimum.
            chunk_rows = max(perm_chunk_min_rows, perm_chunk_bytes // max(bpr, 1))
            if chunk_rows <= 0:  # ultra-conservative fallback
                chunk_rows = perm_chunk_min_rows

            r0 = 0  # absolute write position into perm_stats (and K_perm_local when present)
            if n_jobs == 1:
                # Serial evaluation per block
                X_work: Optional[np.ndarray] = None
                if linear_model:
                    X_work = v.X.copy()

                for block in iter_permuted_matrix(
                    v.T,
                    reps,
                    cluster=v.cluster,
                    strata=v.strata,
                    rng=rng,
                    chunk_rows=int(chunk_rows),
                ):
                    m = block.shape[0]
                    if linear_model:
                        Xw = cast(np.ndarray, X_work)
                        for i in range(m):
                            ols_r = _solve_linear_perm(
                                block[i], X_template=Xw, reuse_template=True
                            )
                            perm_stats[r0 + i] = float(ols_r.beta_hat)
                            if need_coef_ci:
                                K_perm_local[r0 + i] = float(
                                    ols_r.c_vector @ t_metric_lin
                                )  # type: ignore[index]
                    else:
                        for i in range(m):
                            dfp = df.copy(deep=False)
                            dfp[permute_var] = block[i]
                            perm_stats[r0 + i] = float(stat_fn_local(dfp))  # type: ignore[arg-type]
                    r0 += m
            else:
                # Parallel evaluation per block; keep one pool for the entire run.
                with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    for block in iter_permuted_matrix(
                        v.T,
                        reps,
                        cluster=v.cluster,
                        strata=v.strata,
                        rng=rng,
                        chunk_rows=int(chunk_rows),
                    ):
                        m = block.shape[0]
                        if linear_model:
                            # Map absolute indices to rows in the current block
                            worker = _fit_one_beta_K if need_coef_ci else _fit_one_beta
                            if need_coef_ci:
                                for r_abs, beta_r, Kr in ex.map(
                                    worker, ((r0 + i, block[i]) for i in range(m))
                                ):
                                    perm_stats[r_abs] = beta_r
                                    K_perm_local[r_abs] = Kr  # type: ignore[index]
                            else:
                                for r_abs, beta_r in ex.map(
                                    worker, ((r0 + i, block[i]) for i in range(m))
                                ):
                                    perm_stats[r_abs] = beta_r
                        else:
                            for r_abs, z in ex.map(
                                _eval_one, ((r0 + i, block[i]) for i in range(m))
                            ):
                                perm_stats[r_abs] = z
                        r0 += m

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
    band_valid_linear = linear_model and need_coef_ci

    if need_coef_ci:
        K_perm_arr = cast(np.ndarray, K_perm_local)
        if ci_mode in {"bounds", "band"}:
            coef_ci_bounds = coef_ci_bounds_fast(
                beta_obs=obs_stat,
                beta_perm=perm_stats,
                K_obs=K_obs,  # type: ignore[arg-type]
                K_perm=K_perm_arr,
                se=se_obs,
                alpha=alpha,
                ci_range=ci_range,
                ci_step=ci_step,
                alternative=alternative,
            )
        if ci_mode == "band":
            coef_ci_band = coef_ci_band_fast(
                beta_obs=obs_stat,
                beta_perm=perm_stats,
                K_obs=K_obs,  # type: ignore[arg-type]
                K_perm=K_perm_arr,
                se=se_obs,
                ci_range=ci_range,
                ci_step=ci_step,
                alternative=alternative,
            )
            band_valid_linear = True

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
    elif ci_mode == "band":
        pass
    else:
        raise ValueError(f"Unknown ci_mode: {ci_mode!r}")

    # 8) Package results
    strata_count = (
        int(np.unique(v.strata).size) if v.has_strata and v.strata is not None else None
    )
    cluster_count = (
        int(np.unique(v.cluster).size)
        if v.has_cluster and v.cluster is not None
        else None
    )

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
        "strata": strata_count,
        "clusters": cluster_count,
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
