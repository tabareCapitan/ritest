# tests/test_pvalue_ci.py
"""
Comprehensive tests for ritest.ci.pvalue_ci.pvalue_ci

How to run:
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
# Global numerical tolerance (tight; SciPy/statsmodels agree closely)
# ------------------------------------------------------------------ #
TOL = 1e-12


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def manual_wald_cc(c: int, reps: int, alpha: float) -> tuple[float, float]:
    """
    Exact clone of the (normal + continuity correction) logic used in pvalue_ci.
    CC magnitude is ± 0.5/reps, and bounds are clipped to [0, 1].
    """
    p_hat = c / reps
    z = float(norm.ppf(1 - alpha / 2))
    se = math.sqrt(p_hat * (1 - p_hat) / reps)
    eps = 0.5 / reps
    lo = max(0.0, p_hat - eps - z * se)
    hi = min(1.0, p_hat + eps + z * se)
    return lo, hi


# ================================================================== #
# 1. Clopper–Pearson (exact) tests
# ================================================================== #


@pytest.mark.parametrize(
    "c,reps,alpha",
    [
        # edges & mids (α = 0.05)
        (0, 10, 0.05),
        (1, 10, 0.05),
        (5, 10, 0.05),
        (9, 10, 0.05),
        (10, 10, 0.05),
        # different α
        (2, 20, 0.01),
        (7, 20, 0.10),
        # larger reps
        (25, 100, 0.05),
        (0, 100, 0.05),
        (100, 100, 0.05),
    ],
)
def test_cp_matches_statsmodels(c: int, reps: int, alpha: float):
    """Clopper–Pearson should match statsmodels' 'beta' method exactly."""
    if c == 0:
        print("\n[1] CP vs statsmodels:'beta'")
    lo, hi = pvalue_ci(c, reps, alpha=alpha, method="clopper-pearson")
    lo_sm, hi_sm = proportion_confint(c, reps, alpha=alpha, method="beta")

    print(f"  CP   (c={c}, R={reps}, α={alpha})")
    print(f"    → yours:       ({lo:.12f}, {hi:.12f})")
    print(f"    → statsmodels: ({lo_sm:.12f}, {hi_sm:.12f})")

    assert math.isclose(lo, lo_sm, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi, hi_sm, rel_tol=TOL, abs_tol=TOL)


def test_cp_complementarity_property():
    """
    For equal-tailed CP intervals:
      Upper(c) == 1 - Lower(reps - c)
      Lower(c) == 1 - Upper(reps - c)
    This should hold up to numerical tolerance (edges included).
    """
    print("\n[1b] CP complementarity property")
    reps = 50
    alphas = (0.01, 0.05, 0.10)
    for alpha in alphas:
        for c in (0, 1, 5, 10, 25, 49, 50):
            lo_c, hi_c = pvalue_ci(c, reps, alpha=alpha, method="clopper-pearson")
            lo_m, hi_m = pvalue_ci(
                reps - c, reps, alpha=alpha, method="clopper-pearson"
            )

            print(f"  R={reps}, α={alpha}, c={c:2d} -> U(c)≈1-L(R-c), L(c)≈1-U(R-c)")
            assert math.isclose(hi_c, 1.0 - lo_m, rel_tol=1e-10, abs_tol=1e-10)
            assert math.isclose(lo_c, 1.0 - hi_m, rel_tol=1e-10, abs_tol=1e-10)


def test_cp_monotone_in_c():
    """
    As c increases, both lower and upper CP bounds should be non-decreasing.
    """
    print("\n[1c] CP monotonicity in c")
    reps = 80
    alpha = 0.05
    lows, highs = [], []
    for c in range(reps + 1):
        lo, hi = pvalue_ci(c, reps, alpha=alpha, method="clopper-pearson")
        lows.append(lo)
        highs.append(hi)
    assert all(x2 >= x1 for x1, x2 in zip(lows, lows[1:]))
    assert all(x2 >= x1 for x1, x2 in zip(highs, highs[1:]))


# ================================================================== #
# 2. Normal (Wald + continuity correction) tests
# ================================================================== #


@pytest.mark.parametrize(
    "c,reps,alpha",
    [
        (0, 10, 0.05),
        (1, 10, 0.05),
        (5, 10, 0.05),
        (9, 10, 0.05),
        (10, 10, 0.05),
        (3, 30, 0.01),
        (15, 60, 0.10),
        (0, 1, 0.05),
        (1, 1, 0.05),
    ],
)
def test_normal_wald_cc_equals_manual(c: int, reps: int, alpha: float):
    """Wald+CC variant should match our manual helper exactly."""
    if c == 0:
        print("\n[2] Normal (Wald + CC) vs manual formula")
    lo, hi = pvalue_ci(c, reps, alpha=alpha, method="normal")
    lo_exp, hi_exp = manual_wald_cc(c, reps, alpha)

    print(f"  Wald (c={c}, R={reps}, α={alpha})")
    print(f"    → yours:    ({lo:.12f}, {hi:.12f})")
    print(f"    → expected: ({lo_exp:.12f}, {hi_exp:.12f})")

    assert math.isclose(lo, lo_exp, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi, hi_exp, rel_tol=TOL, abs_tol=TOL)


def test_normal_edges_clip_to_unit_interval():
    """At c=0 and c=R the continuity-corrected bounds should clip nicely."""
    print("\n[2b] Normal edges & clipping")
    lo0, hi0 = pvalue_ci(0, 50, method="normal")
    loR, hiR = pvalue_ci(50, 50, method="normal")
    print(f"  c=0  -> ({lo0:.6f}, {hi0:.6f})")
    print(f"  c=R  -> ({loR:.6f}, {hiR:.6f})")
    assert lo0 == 0.0 and 0.0 < hi0 <= 1.0
    assert 0.0 <= loR < 1.0 and hiR == 1.0


# ================================================================== #
# 3. Alpha sensitivity (smaller alpha -> wider intervals)
# ================================================================== #


@pytest.mark.parametrize("method", ["clopper-pearson", "normal"])
def test_alpha_controls_width(method: str):
    print("\n[3] alpha sensitivity (smaller alpha => wider CI)")
    reps, c = 200, 40
    lo80, hi80 = pvalue_ci(c, reps, alpha=0.20, method=method)  # 80% CI
    lo99, hi99 = pvalue_ci(c, reps, alpha=0.01, method=method)  # 99% CI
    wid80 = hi80 - lo80
    wid99 = hi99 - lo99
    print(f"  {method}: width(80%)={wid80:.6f}, width(99%)={wid99:.6f}")
    assert wid99 > wid80
    assert lo99 <= lo80 <= hi80 <= hi99


# ================================================================== #
# 4. Input validation
# ================================================================== #


@pytest.mark.parametrize("bad_reps", [0, -1, 10.0])
def test_bad_reps_raises(bad_reps):
    print("\n[4] Input validation: bad reps")
    with pytest.raises(ValueError, match="positive integer"):
        pvalue_ci(0, bad_reps)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_c", [-1, 11, 3.5])
def test_bad_c_raises(bad_c):
    print("\n[4] Input validation: bad c")
    with pytest.raises(ValueError, match="integer"):
        pvalue_ci(bad_c, reps=10)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_bad_alpha_raises(bad_alpha):
    print("\n[4] Input validation: bad alpha")
    with pytest.raises(ValueError, match="alpha"):
        pvalue_ci(1, 10, alpha=bad_alpha)


def test_unknown_method_raises():
    print("\n[4] Input validation: unknown method")
    with pytest.raises(ValueError, match="Unknown ci method"):
        pvalue_ci(1, 10, method="bogus")  # type: ignore[arg-type]


def test_numpy_integer_rejected_currently():
    """
    Current behavior: dtype must be Python int, np.int64 is rejected by isinstance(c, int).
    If you later relax this (e.g., isinstance(c, (int, np.integer))), update this test.
    """
    print("\n[4] Input validation: numpy int rejected (current behavior)")
    with pytest.raises(ValueError):
        pvalue_ci(np.int64(1), 10)  # type: ignore[arg-type]


def test_cp_alias_is_supported():
    """Legacy short name 'cp' should remain accepted and map to canonical output."""
    c, reps, alpha = 7, 20, 0.05
    lo_alias, hi_alias = pvalue_ci(c, reps, alpha=alpha, method="cp")
    lo_full, hi_full = pvalue_ci(c, reps, alpha=alpha, method="clopper-pearson")
    assert lo_alias == lo_full and hi_alias == hi_full


# ================================================================== #
# 5. Type and range sanity
# ================================================================== #


@pytest.mark.parametrize("method", ["clopper-pearson", "normal"])
def test_types_and_range(method: str):
    print("\n[5] Type/range sanity")
    lo, hi = pvalue_ci(7, 20, alpha=0.05, method=method)
    assert isinstance(lo, float) and isinstance(hi, float)
    assert 0.0 <= lo <= hi <= 1.0


# ================================================================== #
# 6. Simulated large-data case (sanity + perf ballpark)
# ================================================================== #


def test_simulated_large_case():
    """
    Draw c ~ Binomial(R, p) with large R and compare:
    - CP to statsmodels 'beta'
    - Normal to manual_wald_cc
    """
    print("\n[6] Simulated large case: R=100000, p≈0.25")
    rng = np.random.default_rng(123)
    reps = 100_000
    c = int(rng.binomial(n=reps, p=0.25))
    alpha = 0.05

    # CP method
    t0 = time.perf_counter()
    lo_cp, hi_cp = pvalue_ci(c, reps, alpha=alpha, method="clopper-pearson")
    t1 = time.perf_counter()
    lo_sm, hi_sm = proportion_confint(c, reps, alpha=alpha, method="beta")
    t2 = time.perf_counter()

    print(f"  CP (c={c})")
    print(
        f"    → yours:       ({lo_cp:.12f}, {hi_cp:.12f})   [{(t1 - t0)*1000:.2f} ms]"
    )
    print(
        f"    → statsmodels: ({lo_sm:.12f}, {hi_sm:.12f})   [{(t2 - t1)*1000:.2f} ms]"
    )
    assert math.isclose(lo_cp, lo_sm, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi_cp, hi_sm, rel_tol=TOL, abs_tol=TOL)

    # Normal method
    t3 = time.perf_counter()
    lo_norm, hi_norm = pvalue_ci(c, reps, alpha=alpha, method="normal")
    t4 = time.perf_counter()
    lo_exp, hi_exp = manual_wald_cc(c, reps, alpha)
    t5 = time.perf_counter()

    print("  Normal")
    print(
        f"    → yours:    ({lo_norm:.12f}, {hi_norm:.12f})   [{(t4 - t3)*1000:.2f} ms]"
    )
    print(f"    → expected: ({lo_exp:.12f}, {hi_exp:.12f})   [{(t5 - t4)*1000:.2f} ms]")
    assert math.isclose(lo_norm, lo_exp, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(hi_norm, hi_exp, rel_tol=TOL, abs_tol=TOL)


# ================================================================== #
# 7. Fuzzing-lite: random parameter sweeps (robustness)
# ================================================================== #


def test_random_parameter_sweep():
    """
    Randomly sample (c, reps, alpha) and check basic invariants for both methods:
    - No exceptions
    - Bounds in [0, 1], and lo <= hi
    """
    print("\n[7] Random parameter sweep")
    rng = np.random.default_rng(987)
    for _ in range(50):
        reps = int(rng.integers(1, 200))
        c = int(rng.integers(0, reps + 1))
        alpha = float(
            rng.uniform(1e-6, 0.5)
        )  # avoid extreme tails for stability checks
        for method in ("clopper-pearson", "normal"):
            lo, hi = pvalue_ci(c, reps, alpha=alpha, method=method)
            assert isinstance(lo, float) and isinstance(hi, float)
            assert 0.0 <= lo <= hi <= 1.0
