"""
Confidence intervals for permutation p-values.

Interpretation
--------------
The observed exceedance count ``c`` is treated as a Binomial(n=reps, p_true)
draw, where ``p_true`` is the underlying randomisation p-value. This module
computes a (1 - alpha) confidence interval for ``p_true`` using one of two
methods:

- "cp":     Clopper–Pearson exact equal-tailed binomial CI.
- "normal": Wald normal approximation with a simple continuity correction
            of ± 0.5 / reps.

Boundary behaviour (Clopper–Pearson)
------------------------------------
- If ``c == 0``, the lower bound is set to 0.
- If ``c == reps``, the upper bound is set to 1.
All results are clamped to the unit interval as a safeguard.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

from scipy.stats import beta, norm

_PValCIMethod = Literal["cp", "normal"]

__all__ = ["pvalue_ci"]


def _clamp01(x: float) -> float:
    """Clamp a scalar to the closed interval [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def pvalue_ci(
    c: int,
    reps: int,
    alpha: float = 0.05,
    method: _PValCIMethod = "cp",
) -> Tuple[float, float]:
    """
    Compute a (1 - alpha) confidence interval for the true randomisation p-value.

    Parameters
    ----------
    c : int
        Number of permuted statistics at least as extreme as observed.
        Must satisfy 0 <= c <= reps.
    reps : int
        Total number of permutation draws; must be a positive integer.
    alpha : float, default 0.05
        Significance level for the equal-tailed interval.
    method : {"cp", "normal"}, default "cp"
        - "cp": Clopper–Pearson exact interval.
        - "normal": Wald interval with a continuity correction.

    Returns
    -------
    (lower, upper) : tuple of float
        CI bounds, both in [0, 1], satisfying lower <= upper.

    Raises
    ------
    ValueError
        If inputs are invalid or the method is unrecognised.
    """
    # Basic input checks (localised to avoid upstream reliance)
    if not isinstance(reps, int) or reps < 1:
        raise ValueError(f"`reps` must be a positive integer (got {reps!r})")

    if not isinstance(c, int) or not (0 <= c <= reps):
        raise ValueError(f"`c` must be an integer in [0, reps] (got {c!r})")

    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1) (got {alpha!r})")

    if method == "cp":
        # Clopper–Pearson equal-tailed binomial CI
        lo_raw = float(beta.ppf(alpha / 2.0, c, reps - c + 1))
        hi_raw = float(beta.ppf(1.0 - alpha / 2.0, c + 1, reps - c))

        # Handle NaNs at boundary cases and clamp to [0, 1]
        lo = 0.0 if math.isnan(lo_raw) else _clamp01(lo_raw)
        hi = 1.0 if math.isnan(hi_raw) else _clamp01(hi_raw)
        return (lo, hi)

    if method == "normal":
        # Wald CI with continuity correction
        p_hat = c / reps
        z = float(norm.ppf(1.0 - alpha / 2.0))
        se = math.sqrt(p_hat * (1.0 - p_hat) / reps)
        eps = 0.5 / reps

        lo = _clamp01(p_hat - eps - z * se)
        hi = _clamp01(p_hat + eps + z * se)
        return (lo, hi)

    raise ValueError(f"Unknown ci method: {method!r}")
