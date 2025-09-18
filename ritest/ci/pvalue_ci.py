"""
ritest.ci.pvalue_ci

Confidence intervals for permutation p-values.

We view the observed exceedance count `c` as a draw from Binomial(n=reps, p_true),
where p_true is the (unknown) true permutation p-value under the resampling scheme.
This module returns a (1 - alpha) confidence interval for p_true using either:

- "cp":     Clopper–Pearson (exact equal-tailed binomial CI)
- "normal": Wald (normal approximation) with a continuity correction of ± 0.5 / reps

Edge behavior for Clopper–Pearson:
- If c = 0, the lower bound is set to 0.
- If c = reps, the upper bound is set to 1.
All bounds are finally clamped to [0, 1] as a safety belt.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

from scipy.stats import beta, norm

_PValCIMethod = Literal["cp", "normal"]

__all__ = ["pvalue_ci"]


def _clamp01(x: float) -> float:
    """Clamp a float to [0, 1]."""
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
    (1 - alpha) confidence interval for the true randomisation p-value.

    Parameters
    ----------
    c : int
        Number of permuted statistics as or more extreme than observed.
        Must satisfy 0 <= c <= reps.
    reps : int
        Total number of permutation draws. Must be a positive integer (>= 1).
    alpha : float, default 0.05
        Significance level. Must be in (0, 1). Returns a (1 - alpha) CI.
    method : {"cp", "normal"}, default "cp"
        - "cp":     Clopper–Pearson exact binomial CI
        - "normal": Wald with continuity correction ± 0.5 / reps

    Returns
    -------
    (lower, upper) : tuple[float, float]
        Lower and upper CI bounds in [0, 1] with lower <= upper.

    Raises
    ------
    ValueError
        If inputs are invalid or method is unknown.
    """
    # Basic input validation (kept local to protect against misuse and direct calls)
    if not isinstance(reps, int) or reps < 1:
        raise ValueError(f"`reps` must be a positive integer (got {reps!r})")

    if not isinstance(c, int) or not (0 <= c <= reps):
        raise ValueError(f"`c` must be an integer in [0, reps] (got {c!r})")

    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1) (got {alpha!r})")

    if method == "cp":
        # Clopper–Pearson equal-tailed interval
        # Lower:  Beta^{-1}(α/2;    c,     reps - c + 1)
        # Upper:  Beta^{-1}(1-α/2;  c + 1, reps - c)
        lo_raw = float(beta.ppf(alpha / 2.0, c, reps - c + 1))
        hi_raw = float(beta.ppf(1.0 - alpha / 2.0, c + 1, reps - c))

        # Handle boundary NaNs from SciPy at c=0 or c=reps, then clamp to [0,1]
        lo = 0.0 if math.isnan(lo_raw) else _clamp01(lo_raw)
        hi = 1.0 if math.isnan(hi_raw) else _clamp01(hi_raw)
        return (lo, hi)

    if method == "normal":
        # Wald CI with continuity correction
        p_hat = c / reps  # float division
        z = float(norm.ppf(1.0 - alpha / 2.0))
        se = math.sqrt(p_hat * (1.0 - p_hat) / reps)
        eps = 0.5 / reps

        lo = _clamp01(p_hat - eps - z * se)
        hi = _clamp01(p_hat + eps + z * se)
        return (lo, hi)

    raise ValueError(f"Unknown ci method: {method!r}")
