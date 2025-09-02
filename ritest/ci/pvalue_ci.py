"""
ritest.ci.pvalue_ci

Compute confidence intervals for permutation p-values.

Supports:
- "cp":     Clopper–Pearson (exact)
- "normal": Wald with continuity correction
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

from scipy.stats import beta, norm

_PValCIMethod = Literal["cp", "normal"]


def pvalue_ci(
    c: int,
    reps: int,
    alpha: float = 0.05,
    method: _PValCIMethod = "cp",
) -> Tuple[float, float]:
    """
    CI for the true randomisation *p*-value.

    Parameters
    ----------
    c : int
        Number of permuted stats as or more extreme than observed.
    reps : int
        Total number of permutation draws.
    alpha : float
        Significance level (returns 1–α CI).
    method : {"cp", "normal"}
        CI method.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
    """
    if not isinstance(c, int) or not (0 <= c <= reps):
        raise ValueError(f"`c` must be integer in [0, reps] (got {c})")

    if method == "cp":
        lo_raw = beta.ppf(alpha / 2, c, reps - c + 1)
        hi_raw = beta.ppf(1 - alpha / 2, c + 1, reps - c)

        lo = float(0.0 if math.isnan(lo_raw) else lo_raw)
        hi = float(1.0 if math.isnan(hi_raw) else hi_raw)
        return lo, hi

    if method == "normal":
        # Make everything native float to silence type checkers
        p_hat = float(c / reps)
        z = float(norm.ppf(1 - alpha / 2))
        se = float(math.sqrt(p_hat * (1 - p_hat) / reps))
        eps = float(0.5 / reps)

        lo = float(max(0.0, p_hat - eps - z * se))
        hi = float(min(1.0, p_hat + eps + z * se))
        return lo, hi

    raise ValueError(f"Unknown ci method: {method!r}")  # should be unreachable
