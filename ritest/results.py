"""
ritest.results
==============

Presentation utilities and container class for *randomization inference* output.

`RitestResult` is returned by `ritest()` (see run.py). It provides:
- A friendly `summary()` string (optionally printed),
- A concise `__repr__` and `__str__`,
- A simple `plot()` that visualizes the coefficient CI band p-profile (if computed),
- A short `explain()` helper for a plain-English takeaway.

Design goals:
- No global state; rely only on fields present in the object.
- Deterministic formatting (fixed decimals) for reproducibility.
- Lazy import of matplotlib inside `plot()`; return the Axes (no disk writes).
- Graceful handling when optional fields are not computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import indent
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # keeps Pylance happy without importing matplotlib at runtime
    from matplotlib.axes import Axes  # pragma: no cover

__all__ = ["RitestResult"]


# --------- formatting helpers (deterministic) --------- #
def _fmt_float(x: float, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _fmt_pct(p: float, nd: int = 1) -> str:
    try:
        return f"{float(p) * 100:.{nd}f}%"
    except Exception:
        return str(p)


def _fmt_ci(ci: Optional[Tuple[float, float]], nd: int = 4) -> str:
    if not ci or len(ci) != 2:
        return "not computed"
    lo, hi = ci
    return f"[{_fmt_float(lo, nd)}, {_fmt_float(hi, nd)}]"


def _yn(flag: bool) -> str:
    return "yes" if bool(flag) else "no"


def _get_alpha(settings: Dict[str, object], fallback: float = 0.05) -> float:
    a = settings.get("alpha", fallback)
    # Narrow for the type checker and be safe at runtime.
    if isinstance(a, (int, float, np.floating)):
        return float(a)
    if isinstance(a, str):
        try:
            return float(a)
        except ValueError:
            return float(fallback)
    return float(fallback)


def _get_runtime(settings: Dict[str, object], runtime_attr: Optional[float]) -> Optional[float]:
    # Prefer settings["runtime_sec"], fall back to the attribute if present.
    val: Optional[object] = settings.get("runtime_sec", None)
    if val is None:
        val = runtime_attr
        if val is None:
            return None

    if isinstance(val, (int, float, np.floating)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


# --------- main result container --------- #
@dataclass(slots=True)
class RitestResult:
    # Core permutation test outputs
    obs_stat: float
    pval: float
    pval_ci: Optional[Tuple[float, float]]
    reps: int
    c: int
    alternative: str  # "two-sided" | "left" | "right"

    # Design flags
    stratified: bool = False
    clustered: bool = False
    weights: bool = False

    # Coefficient CI outputs
    coef_ci_bounds: Optional[Tuple[float, float]] = None  # (lo, hi) or None
    coef_ci_band: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (beta_grid, pvals) or None
    band_valid_linear: bool = False  # True if fast-linear path; False if generic (when band exists)

    # Full permutation stats (optional, for diagnostics)
    perm_stats: Optional[np.ndarray] = None

    # Settings snapshot and optional runtime
    settings: Dict[str, object] = field(default_factory=dict)
    runtime: Optional[float] = None

    # --------------- dunder methods --------------- #
    def __repr__(self) -> str:
        return (
            f"RitestResult(obs={_fmt_float(self.obs_stat)}, "
            f"p={_fmt_float(self.pval, nd=4)}, alt='{self.alternative}', reps={self.reps})"
        )

    def __str__(self) -> str:
        return (
            f"RI: p={_fmt_float(self.pval, nd=4)} ({self.alternative}), "
            f"reps={self.reps}, β̂={_fmt_float(self.obs_stat)}"
        )

    # --------------- user-facing helpers --------------- #
    def explain(self, alpha: Optional[float] = None) -> str:
        """
        Return a brief, plain-language interpretation of the result.

        Parameters
        ----------
        alpha : float, optional
            Significance threshold to reference. If None, uses settings['alpha'] or 0.05.

        Returns
        -------
        str
            A 2–3 sentence takeaway.
        """
        a = _get_alpha(self.settings, 0.05) if alpha is None else float(alpha)
        tail = self.alternative
        p = float(self.pval)

        dir_phrase = {
            "two-sided": "in either direction",
            "left": "in the negative direction",
            "right": "in the positive direction",
        }.get(tail, "as or more extreme")

        lines = []
        lines.append(
            f"Under the sharp null of no effect for any unit, "
            f"{_fmt_pct(p)} of randomized assignments produced a statistic as or more extreme {dir_phrase} than observed."
        )
        if p <= a:
            lines.append(
                f"At α = {_fmt_float(a, 3)}, the result is **statistically significant** for a {tail} test."
            )
        else:
            lines.append(
                f"At α = {_fmt_float(a, 3)}, the result is **not statistically significant** for a {tail} test."
            )
        lines.append(
            "This p-value is finite-sample and design-based from permutation/randomization inference."
        )
        return " ".join(lines)

    def summary(self, print_out: bool = True) -> str:
        """
        Build a deterministic, human-friendly summary of the RI result.

        Parameters
        ----------
        print_out : bool, default True
            If True, prints the summary to stdout. The string is always returned.

        Returns
        -------
        str
            The formatted summary.
        """
        a = _get_alpha(self.settings, 0.05)
        ci_method = str(self.settings.get("ci_method", "unknown"))
        ci_mode = str(self.settings.get("ci_mode", "unknown"))
        n_jobs = self.settings.get("n_jobs", "unknown")
        seed = self.settings.get("seed", "unknown")

        rt = _get_runtime(self.settings, self.runtime)
        rt_line = f"{_fmt_float(rt, 3)} s" if rt is not None else "n/a"

        # Section: headline & coefficient
        lines: list[str] = []
        lines.append("Randomization Inference Result")
        lines.append("=" * 31)
        lines.append("")
        lines.append("Coefficient")
        lines.append("-----------")
        lines.append(f"Observed effect (β̂):   {_fmt_float(self.obs_stat)}")

        if self.coef_ci_bounds is not None:
            lines.append(f"Coefficient CI bounds: {_fmt_ci(self.coef_ci_bounds)}")
        else:
            lines.append("Coefficient CI bounds: not computed")

        # Band status
        if self.coef_ci_band is not None:
            band_kind = "fast-linear" if self.band_valid_linear else "generic"
            lines.append(f"Coefficient CI band:   available ({band_kind})")
        else:
            lines.append("Coefficient CI band:   not computed")

        lines.append("")
        # Section: permutation test
        lines.append("Permutation test")
        lines.append("----------------")
        lines.append(f"Tail (alternative):     {self.alternative}")
        lines.append(
            f"p-value:                {_fmt_float(self.pval, nd=4)} ({_fmt_pct(self.pval)})"
        )
        lines.append(f"P-value CI @ α={_fmt_float(a, 3)}: {_fmt_ci(self.pval_ci, nd=4)}")
        lines.append(f"As-or-more extreme:     {self.c} / {self.reps}")

        # Design flags
        lines.append("")
        lines.append("Test configuration")
        lines.append("------------------")
        lines.append(f"Stratified:             {_yn(self.stratified)}")
        lines.append(f"Clustered:              {_yn(self.clustered)}")
        lines.append(f"Weights:                {_yn(self.weights)}")

        # Settings
        lines.append("")
        lines.append("Settings")
        lines.append("--------")
        lines.append(f"alpha:                  {_fmt_float(a, 3)}")
        lines.append(f"seed:                   {seed}")
        lines.append(f"ci_method:              {ci_method}")
        lines.append(f"ci_mode:                {ci_mode}")
        lines.append(f"n_jobs:                 {n_jobs}")
        lines.append(f"runtime:                {rt_line}")

        # Interpretation (short)
        lines.append("")
        lines.append("Interpretation")
        lines.append("--------------")
        # Use the same alpha shown above for consistency
        expl = self.explain(alpha=a)
        lines.append(indent(expl, ""))

        out = "\n".join(lines)
        if print_out:
            print(out)
        return out

    def plot(self, *, show: bool = False) -> "Axes":
        """
        Plot the coefficient CI band p-profile, if available.

        The plot shows:
        - The p(β) profile (line),
        - A horizontal line at α (from settings or 0.05),
        - A vertical line at β̂ (observed effect),
        - Vertical lines at coefficient CI bounds, when available.

        Parameters
        ----------
        show : bool, default False
            If True, call `plt.show()` before returning.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes containing the plot.

        Raises
        ------
        ValueError
            If `coef_ci_band` is not available (e.g., ci_mode != "grid").
        """
        if self.coef_ci_band is None:
            raise ValueError("coef_ci_band is not available (likely ci_mode != 'grid').")

        # Lazy import to avoid side effects when only summarizing
        import matplotlib.pyplot as plt  # type: ignore

        beta_grid, pvals = self.coef_ci_band
        beta_grid = np.asarray(beta_grid, dtype=float)
        pvals = np.asarray(pvals, dtype=float)

        if beta_grid.ndim != 1 or pvals.ndim != 1 or beta_grid.shape[0] != pvals.shape[0]:
            raise ValueError("coef_ci_band must be (1D beta_grid, 1D pvals) of equal length.")

        # Sort by beta for a clean line (no assumptions about upstream order)
        order = np.argsort(beta_grid)
        beta_grid = beta_grid[order]
        pvals = pvals[order]
        pvals = np.clip(pvals, 0.0, 1.0)

        a = _get_alpha(self.settings, 0.05)

        fig, ax = plt.subplots()
        ax.plot(beta_grid, pvals, label="p(β)")

        # Alpha reference
        ax.axhline(y=a, linestyle="--", linewidth=1.0, label=f"α = {_fmt_float(a, 3)}")

        # Observed effect
        ax.axvline(
            x=self.obs_stat, linestyle=":", linewidth=1.0, label=f"β̂ = {_fmt_float(self.obs_stat)}"
        )

        # Bounds if available
        if self.coef_ci_bounds is not None:
            lo, hi = self.coef_ci_bounds
            ax.axvline(
                x=lo, linestyle="--", linewidth=1.0, alpha=0.8, label=f"lo = {_fmt_float(lo)}"
            )
            ax.axvline(
                x=hi, linestyle="--", linewidth=1.0, alpha=0.8, label=f"hi = {_fmt_float(hi)}"
            )

        # Cosmetics
        ax.set_xlabel("β")
        ax.set_ylabel("Permutation p-value")
        band_kind = "fast-linear" if self.band_valid_linear else "generic"
        ax.set_title(f"Coefficient CI band — {band_kind}")
        ax.set_ylim(0.0, 1.0)
        # x-limits: small padding around the grid if finite
        if np.all(np.isfinite(beta_grid)):
            span = float(np.max(beta_grid) - np.min(beta_grid))
            pad = 0.05 * span if span > 0 else 1.0
            ax.set_xlim(float(np.min(beta_grid) - pad), float(np.max(beta_grid) + pad))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", frameon=False)

        if show:
            plt.show()

        return ax
