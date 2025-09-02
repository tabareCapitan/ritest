# tests/test_pvalue_ci.py
"""
Unit tests for ritest.ci.pvalue_ci.pvalue_ci

Run with:
    pytest -q -s tests/test_pvalue_ci.py
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from ritest.ci.pvalue_ci import pvalue_ci

# ------------------------------------------------------------------ #
# Global tolerance
# ------------------------------------------------------------------ #
TOL = 1e-6


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def manual_wald_cc(c: int, reps: int, alpha: float) -> tuple[float, float]:
    """Exact clone of the logic used in pvalue_ci (normal + CC)."""
    p_hat = c / reps
    z = float(norm.ppf(1 - alpha / 2))
    se = math.sqrt(p_hat * (1 - p_hat) / reps)
    eps = 0.5 / reps
    lo = max(0.0, p_hat - eps - z * se)
    hi = min(1.0, p_hat + eps + z * se)
    return lo, hi


# ------------------------------------------------------------------ #
# 1. Clopper–Pearson vs statsmodels
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "c,reps,alpha",
    [
        (0, 10, 0.05),
        (1, 10, 0.05),
        (5, 10, 0.05),
        (10, 10, 0.05),
        (2, 20, 0.01),
    ],
)
def test_cp_matches_statsmodels(c: int, reps: int, alpha: float):
    if c == 0:
        print("\n[Section 1] Clopper–Pearson vs statsmodels:")
    lo, hi = pvalue_ci(c, reps, alpha=alpha, method="cp")
    lo_sm, hi_sm = proportion_confint(c, reps, alpha=alpha, method="beta")

    lo_f: float = float(lo)  # type: ignore[arg-type]
    hi_f: float = float(hi)  # type: ignore[arg-type]
    lo_sm_f: float = float(lo_sm)  # type: ignore[arg-type]
    hi_sm_f: float = float(hi_sm)  # type: ignore[arg-type]

    print(f"  CP   (c={c}, R={reps}, α={alpha})")
    print(f"    → yours:       ({lo_f:.6f}, {hi_f:.6f})")
    print(f"    → statsmodels: ({lo_sm_f:.6f}, {hi_sm_f:.6f})")
    assert math.isclose(lo_f, lo_sm_f, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi_f, hi_sm_f, rel_tol=TOL, abs_tol=TOL)


# ------------------------------------------------------------------ #
# 2. Normal (Wald + CC) vs manual formula
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "c,reps,alpha",
    [
        (0, 10, 0.05),
        (1, 10, 0.05),
        (5, 10, 0.05),
        (9, 10, 0.05),
        (10, 10, 0.05),
        (3, 30, 0.01),
    ],
)
def test_normal_wald_cc(c: int, reps: int, alpha: float):
    if c == 0:
        print("\n[Section 2] Normal (Wald + CC) vs manual_wald_cc:")
    lo, hi = pvalue_ci(c, reps, alpha=alpha, method="normal")
    lo_exp, hi_exp = manual_wald_cc(c, reps, alpha)

    lo_f: float = float(lo)  # type: ignore[arg-type]
    hi_f: float = float(hi)  # type: ignore[arg-type]
    lo_exp_f: float = float(lo_exp)  # type: ignore[arg-type]
    hi_exp_f: float = float(hi_exp)  # type: ignore[arg-type]

    print(f"  Wald (c={c}, R={reps}, α={alpha})")
    print(f"    → yours:    ({lo_f:.6f}, {hi_f:.6f})")
    print(f"    → expected: ({lo_exp_f:.6f}, {hi_exp_f:.6f})")
    assert math.isclose(lo_f, lo_exp_f, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi_f, hi_exp_f, rel_tol=TOL, abs_tol=TOL)


# ------------------------------------------------------------------ #
# 3. Edge-case clipping tests
# ------------------------------------------------------------------ #
def test_bound_clipping():
    print("\n[Section 3] Clipping behavior at c = 0 and c = R")
    lo0, hi0 = pvalue_ci(0, 50, method="normal")
    lo1, hi1 = pvalue_ci(50, 50, method="normal")

    lo0_f: float = float(lo0)  # type: ignore[arg-type]
    hi0_f: float = float(hi0)  # type: ignore[arg-type]
    lo1_f: float = float(lo1)  # type: ignore[arg-type]
    hi1_f: float = float(hi1)  # type: ignore[arg-type]

    print(f"  normal(c=0)  -> ({lo0_f:.4g}, {hi0_f:.4g})")
    print(f"  normal(c=R)  -> ({lo1_f:.4g}, {hi1_f:.4g})")
    print("  Clipping test – passed")

    assert lo0_f == 0.0 and 0.0 < hi0_f <= 1.0
    assert 0.0 <= lo1_f < 1.0 and hi1_f == 1.0


# ------------------------------------------------------------------ #
# 4. Input validation
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("bad_c", [-1, 11, 3.5])
def test_bad_c_raises(bad_c):
    print(f"\n[Section 4] Input validation: bad c = {bad_c}")
    with pytest.raises(ValueError, match="integer"):
        pvalue_ci(bad_c, reps=10)


def test_unknown_method_raises():
    print("\n[Section 4] Input validation: unknown method")
    with pytest.raises(ValueError, match="Unknown ci method"):
        pvalue_ci(1, 10, method="bogus")  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# 5. Simulated large-data case
# ------------------------------------------------------------------ #
def test_simulated_large_case():
    print("\n[Section 5] Simulated case: large R=10000, c from Binomial(100000, 0.25)")

    rng = np.random.default_rng(123)
    reps = 100_000
    c = int(rng.binomial(n=reps, p=0.25))
    alpha = 0.05

    # CP method
    t0 = time.perf_counter()
    lo_cp, hi_cp = pvalue_ci(c, reps, alpha=alpha, method="cp")
    t1 = time.perf_counter()
    lo_sm, hi_sm = proportion_confint(c, reps, alpha=alpha, method="beta")
    t2 = time.perf_counter()

    lo_cp_f: float = float(lo_cp)  # type: ignore[arg-type]
    hi_cp_f: float = float(hi_cp)  # type: ignore[arg-type]
    lo_sm_f: float = float(lo_sm)  # type: ignore[arg-type]
    hi_sm_f: float = float(hi_sm)  # type: ignore[arg-type]

    print(f"  CP (c={c})")
    print(f"    → yours:       ({lo_cp_f:.6f}, {hi_cp_f:.6f})   [{(t1 - t0)*1000:.2f} ms]")
    print(f"    → statsmodels: ({lo_sm_f:.6f}, {hi_sm_f:.6f})   [{(t2 - t1)*1000:.2f} ms]")
    assert math.isclose(lo_cp_f, lo_sm_f, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi_cp_f, hi_sm_f, rel_tol=TOL, abs_tol=TOL)

    # Normal method
    t3 = time.perf_counter()
    lo_norm, hi_norm = pvalue_ci(c, reps, alpha=alpha, method="normal")
    t4 = time.perf_counter()
    lo_exp, hi_exp = manual_wald_cc(c, reps, alpha)
    t5 = time.perf_counter()

    lo_norm_f = float(lo_norm)  # type: ignore[arg-type]
    hi_norm_f = float(hi_norm)  # type: ignore[arg-type]
    lo_exp_f = float(lo_exp)  # type: ignore[arg-type]
    hi_exp_f = float(hi_exp)  # type: ignore[arg-type]

    print("  Normal")
    print(f"    → yours:    ({lo_norm_f:.6f}, {hi_norm_f:.6f})   [{(t4 - t3)*1000:.2f} ms]")
    print(f"    → expected: ({lo_exp_f:.6f}, {hi_exp_f:.6f})   [{(t5 - t4)*1000:.2f} ms]")
    assert math.isclose(lo_norm_f, lo_exp_f, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi_norm_f, hi_exp_f, rel_tol=TOL, abs_tol=TOL)
