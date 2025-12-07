"""
ritest.results
==============

Presentation utilities and container class for randomization-inference output.

`RitestResult` is the object returned by `ritest()` (see run.py). It is a
lightweight, self-contained container that:

- Stores the observed statistic, permutation p-value, and counts,
- Optionally stores p-value CIs and coefficient CIs,
- Knows enough settings to reconstruct a stable textual summary,
- Can produce a simple p(β) profile plot for the coefficient CI band.

Design notes
------------
- No global state: everything needed for summary/plotting is stored on the
  instance (obs_stat, pval, CIs, settings snapshot, etc.).
- Formatting is deterministic with fixed decimals to make comparisons and
  tests reproducible.
- Matplotlib is imported lazily inside `plot()`; the method returns an Axes.
- All helpers are read-only; no mutation of upstream data structures.
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
    """Format a scalar as a fixed-decimal string; fall back to `str(x)` on error."""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _fmt_pct(p: float, nd: int = 1) -> str:
    """Format a probability as a percentage string; fall back to `str(p)` on error."""
    try:
        return f"{float(p) * 100:.{nd}f}%"
    except Exception:
        return str(p)


def _fmt_ci(ci: Optional[Tuple[float, float]], nd: int = 4) -> str:
    """Format a (lo, hi) pair as `[lo, hi]`, or `'not computed'` if missing."""
    if not ci or len(ci) != 2:
        return "not computed"
    lo, hi = ci
    return f"[{_fmt_float(lo, nd)}, {_fmt_float(hi, nd)}]"


def _yn(flag: bool) -> str:
    """Return `'yes'` if true-like, `'no'` otherwise."""
    return "yes" if bool(flag) else "no"


def _get_alpha(settings: Dict[str, object], fallback: float = 0.05) -> float:
    """
    Extract `alpha` from a settings dict, with a numeric fallback.

    Accepts numeric or string representations; any other type falls back.
    """
    a = settings.get("alpha", fallback)
    if isinstance(a, (int, float, np.floating)):
        return float(a)
    if isinstance(a, str):
        try:
            return float(a)
        except ValueError:
            return float(fallback)
    return float(fallback)


# --------- main result container --------- #
@dataclass(slots=True)
class RitestResult:
    """
    Container for randomization-inference output and basic presentation helpers.

    This class does not perform any inference itself; it simply organises
    the outputs returned by `ritest()` and exposes:

    - `__repr__` / `__str__` for quick inspection,
    - `summary()` for a multi-section text report,
    - `explain()` for a short plain-language takeaway,
    - `plot()` for the coefficient CI band p-profile (when available).
    """

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
        """Concise repr with key scalar outputs."""
        return (
            f"RitestResult(obs={_fmt_float(self.obs_stat)}, "
            f"p={_fmt_float(self.pval, nd=4)}, alt='{self.alternative}', reps={self.reps})"
        )

    def __str__(self) -> str:
        """Short human-readable line summarising p-value, tail, reps, and β̂."""
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
            Significance threshold to reference. If None, uses
            `settings['alpha']` or 0.05.

        Returns
        -------
        str
            A 2–3 sentence summary in plain language.
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
            If True, print the summary to stdout. The string is always returned.

        Returns
        -------
        str
            The formatted multi-line summary.
        """
        a = _get_alpha(self.settings, 0.05)
        ci_method = str(self.settings.get("ci_method", "unknown"))
        ci_mode = str(self.settings.get("ci_mode", "unknown"))
        n_jobs = self.settings.get("n_jobs", "unknown")
        seed = self.settings.get("seed", "unknown")

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

        # Interpretation (short)
        lines.append("")
        lines.append("Interpretation")
        lines.append("--------------")
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
        - The p(β) profile as a line,
        - A horizontal reference at α,
        - A vertical line at β̂ (observed effect),
        - Vertical lines at coefficient CI bounds, when present.

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
            If `coef_ci_band` is not available (e.g. ci_mode != "grid").
        """
        if self.coef_ci_band is None:
            raise ValueError("coef_ci_band is not available (likely ci_mode != 'grid').")

        # Lazy import to avoid unnecessary dependency cost on summary-only use
        import matplotlib.pyplot as plt  # type: ignore

        beta_grid, pvals = self.coef_ci_band
        beta_grid = np.asarray(beta_grid, dtype=float)
        pvals = np.asarray(pvals, dtype=float)

        if beta_grid.ndim != 1 or pvals.ndim != 1 or beta_grid.shape[0] != pvals.shape[0]:
            raise ValueError("coef_ci_band must be (1D beta_grid, 1D pvals) of equal length.")

        # Sort by beta for a clean monotone x-axis
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
        if np.all(np.isfinite(beta_grid)):
            span = float(np.max(beta_grid) - np.min(beta_grid))
            pad = 0.05 * span if span > 0 else 1.0
            ax.set_xlim(float(np.min(beta_grid) - pad), float(np.max(beta_grid) + pad))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", frameon=False)

        if show:
            plt.show()

        return ax
