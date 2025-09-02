"""
ritest.results
==============

Container and presentation utilities for *Randomization inference* output.

`RitestResult` is returned by ``ritest()`` (run.py) and mirrors the style of
`statsmodels` / `sklearn` result objects:

>>> res = ritest(df, permute_var="T", formula="y ~ T + X")
>>> res.summary()          # pretty console table
>>> fig = res.plot()       # CI-band plot (if available)
>>> p = res.pval           # programmatic access

"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import indent
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = ["RitestResult"]


# ------------------------------------------------------------------ #
# Internal helper – simple table formatter
# ------------------------------------------------------------------ #
def _kv(key: str, val: str, width: int = 26) -> str:
    """Format ``key: padded_value`` for summary() rows."""
    return f"{key:<{width}} {val}"


# ------------------------------------------------------------------ #
# Main public container
# ------------------------------------------------------------------ #
@dataclass
class RitestResult:
    """
    Result object returned by :pyfunc:`ritest.ritest`.

    Parameters
    ----------
    obs_stat : float
        Observed statistic (β̂ if formula path, else stat_fn value).
    coef_ci_bounds : (lo, hi) or ``None``
        Coefficient CI bounds (``None`` if not computed).
    pval : float
        Randomisation p-value.
    pval_ci : (lo, hi)
        Clopper–Pearson or Wald CI for p-value.
    reps : int
        Number of permutation draws.
    c : int
        Count of permuted statistics as (or more) extreme than observed.
    alternative : {"two-sided", "left", "right"}
        Test direction.
    stratified, clustered, weights : bool
        Flags describing the design.
    coef_ci_band : (grid, pvals), optional
        Full (β₀, p(β₀)) curve when ``ci_mode == "grid"``.
    band_valid_linear : bool
        ``True`` if band was computed under a *linear* model; if ``False``
        a warning note is printed in :pyfunc:`summary`.
    settings : dict, optional
        Snapshot of key configuration parameters (useful for audit logs).
    perm_stats : ndarray, optional
        Vector of permuted statistics.
    runtime : float, optional
        Total runtime in seconds (not displayed, but stored for inspection).
    """

    # core results
    obs_stat: float
    coef_ci_bounds: Optional[Tuple[float, float]]
    pval: float
    pval_ci: Tuple[float, float]
    reps: int
    c: int
    alternative: str

    # design flags
    stratified: bool
    clustered: bool
    weights: bool

    # optional band
    coef_ci_band: Optional[Tuple[np.ndarray, np.ndarray]] = None
    band_valid_linear: bool = True  # warn if False

    # misc
    settings: Dict[str, object] = field(default_factory=dict)
    perm_stats: Optional[np.ndarray] = None
    runtime: Optional[float] = None  # stored, not shown

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401
        """Short repr for interactive sessions."""
        return f"<RitestResult obs={self.obs_stat:.4g} " f"p={self.pval:.4g} ({self.alternative})>"

    # ------------------------------------------------------------------ #
    # Public summary
    # ------------------------------------------------------------------ #
    def summary(self, *, print_out: bool = True) -> str:
        """
        Return (and optionally print) a formatted summary of results.

        Parameters
        ----------
        print_out : bool, default True
            If *True* the summary is printed to stdout.

        Returns
        -------
        summary_str : str
            The formatted summary table.
        """
        lines: list[str] = []
        lines.append("Randomization Inference Result")
        lines.append("-" * 30)
        lines.append("")

        # ----- Coefficient section ------------------------------------ #
        lines.append("Coefficient:")
        lines.append(_kv("- Observed effect:", f"{self.obs_stat:.6g}"))

        if self.coef_ci_bounds is not None:
            alpha = self.settings.get("alpha", 0.05)
            lo, hi = self.coef_ci_bounds
            lines.append(_kv(f"- Coef CI (alpha = {alpha:.3g}):", f"[{lo:.6g}, {hi:.6g}]"))
        else:
            lines.append(_kv("- Coef CI:", "not computed"))

        lines.append("")

        # ----- P-value section ---------------------------------------- #
        lines.append("P-value:")
        lines.append(_kv("- Reps (n):", f"{self.reps:,}"))
        lines.append(_kv("- More extreme (c):", f"{self.c:,}"))
        lines.append(_kv("- P = c / n:", f"{self.pval:.6g}"))

        alpha = self.settings.get("alpha", 0.05)
        lo_p, hi_p = self.pval_ci
        lines.append(_kv(f"- P-value CI (alpha = {alpha:.3g}):", f"[{lo_p:.6g}, {hi_p:.6g}]"))
        lines.append("")

        # ----- Interpretation (placeholder) --------------------------- #
        alt_text = {
            "two-sided": "in either direction",
            "right": "in the positive direction",
            "left": "in the negative direction",
        }[self.alternative]

        interp = (
            f"Under the sharp null hypothesis of no effect for any unit, "
            f"{self.pval:.1%} of permuted treatment assignments produced "
            f"an effect as or more extreme {alt_text} than the observed."
        )
        lines.append("Interpretation:")
        lines.append(indent(interp, prefix=""))
        lines.append("")

        # ----- Configuration ----------------------------------------- #
        lines.append("Test Configuration:")
        lines.append(_kv("- Alternative hypothesis:", self.alternative))
        lines.append(_kv("- Stratified:", "yes" if self.stratified else "no"))
        lines.append(_kv("- Clustered:", "yes" if self.clustered else "no"))
        lines.append(_kv("- Weights used:", "yes" if self.weights else "no"))
        lines.append("")

        # ----- Optional note ----------------------------------------- #
        if self.coef_ci_band is not None and not self.band_valid_linear:
            note = (
                "Coef CI band was computed using a generic statistic. "
                "Validity is *not* guaranteed unless the model is linear."
            )
            lines.append("Note:")
            lines.append(indent(note, prefix=""))
            lines.append("")

        summary_str = "\n".join(lines)

        if print_out:
            print(summary_str)

        return summary_str

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot(self, *, show: bool = True) -> "Figure":
        """
        Plot the coefficient CI band (β₀ vs p-value) if available.

        Raises
        ------
        ValueError
            If CI band was not computed (``ci_mode != "grid"``).
        """
        if self.coef_ci_band is None:
            raise ValueError(
                "Coefficient CI band was not computed. "
                "Enable it via `ritest_set({'ci_mode': 'grid'})` "
                "and rerun `ritest()`."
            )

        grid, pvals = self.coef_ci_band
        alpha = self.settings.get("alpha", 0.05)

        fig, ax = plt.subplots()
        ax.plot(grid, pvals, linewidth=1.6)
        ax.axhline(alpha, linestyle="--", linewidth=1)
        ax.axvline(self.obs_stat, linestyle=":", linewidth=1)

        ax.set_xlabel("β₀")
        ax.set_ylabel("Permutation p-value")
        ax.set_title("Coefficient CI band")
        ax.set_ylim(0, 1)
        ax.grid(True, linewidth=0.5, alpha=0.4)

        if show:
            plt.show()

        return fig
