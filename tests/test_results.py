# tests/test_results.py
import numpy as np
import pytest

from ritest.results import RitestResult

try:
    from matplotlib.axes import Axes  # type: ignore
except Exception:  # pragma: no cover
    Axes = object


def make_res(
    *,
    obs_stat: float = 0.1234,
    pval: float = 0.0321,
    pval_ci: tuple[float, float] | None = (0.018, 0.051),
    reps: int = 5000,
    c: int = 160,
    alternative: str = "two-sided",
    stratified: bool = True,
    clustered: bool = False,
    weights: bool = True,
    coef_ci_bounds: tuple[float, float] | None = (0.01, 0.25),
    with_band: bool = True,
    band_valid_linear: bool = True,
    runtime: float | None = None,
):
    settings: dict[str, object] = {
        "alpha": 0.05,
        "seed": 123,
        "ci_method": "clopper-pearson",
        "ci_mode": "grid" if with_band else "none",
        "n_jobs": 1,
        "runtime_sec": 0.123 if runtime is None else runtime,
    }
    if with_band:
        grid = np.linspace(-0.5, 0.5, 11)
        pvals = np.clip(2.0 * np.abs(grid), 0.0, 1.0)
        band = (grid, pvals)
    else:
        band = None

    return RitestResult(
        obs_stat=obs_stat,
        pval=pval,
        pval_ci=pval_ci,
        reps=reps,
        c=c,
        alternative=alternative,
        stratified=stratified,
        clustered=clustered,
        weights=weights,
        coef_ci_bounds=coef_ci_bounds,
        coef_ci_band=band,
        band_valid_linear=band_valid_linear,
        perm_stats=None,
        settings=settings,
        runtime=runtime,
    )


def test_summary_includes_sections_and_values():
    res = make_res()
    s = res.summary(print_out=False)

    assert "Randomization Inference Result" in s
    assert "Permutation test" in s
    assert "Test configuration" in s
    assert "Settings" in s
    assert "Interpretation" in s

    assert "Observed effect (β̂):" in s
    assert "Tail (alternative):     two-sided" in s
    assert "As-or-more extreme:     160 / 5000" in s
    assert "Coefficient CI band:   available (fast-linear)" in s

    # deterministic formatting
    assert "P-value CI @ α=0.050: [0.0180, 0.0510]" in s
    assert "alpha:                  0.050" in s
    assert "seed:                   123" in s
    assert "ci_method:              clopper-pearson" in s
    assert "ci_mode:                grid" in s
    assert "n_jobs:                 1" in s
    assert "runtime:" in s


def test_summary_handles_missing_optionals_cleanly():
    res = make_res(coef_ci_bounds=None, with_band=False, pval_ci=None)
    s = res.summary(print_out=False)

    assert "Coefficient CI bounds: not computed" in s
    assert "Coefficient CI band:   not computed" in s
    assert "P-value CI @ α=0.050: not computed" in s


def test_generic_band_note():
    res = make_res(band_valid_linear=False)
    s = res.summary(print_out=False)
    assert "Coefficient CI band:   available (generic)" in s


def test_explain_significant_and_not_significant():
    res_sig = make_res(pval=0.040)
    e1 = res_sig.explain(alpha=0.05).lower()
    assert "statistically significant" in e1
    assert "two-sided" in e1

    res_ns = make_res(pval=0.060)
    e2 = res_ns.explain(alpha=0.05).lower()
    assert "not statistically significant" in e2


def test_repr_and_str_are_concise():
    res = make_res()
    r = repr(res)
    s = str(res)
    assert "RitestResult(" in r and "obs=" in r and "p=" in r and "reps=5000" in r
    assert s.startswith("RI: p=") and "β̂=" in s


def test_plot_with_band_returns_axes_and_draws_lines():
    res = make_res()
    ax = res.plot(show=False)
    assert isinstance(ax, Axes)

    # band + alpha + beta-hat (>=3 lines); bounds may add 2 more
    assert len(ax.lines) >= 3

    y0, y1 = ax.get_ylim()
    assert y0 <= 0.0 + 1e-9 and y1 >= 1.0 - 1e-9


def test_plot_without_band_raises():
    res = make_res(with_band=False)
    with pytest.raises(ValueError, match="coef_ci_band is not available"):
        res.plot(show=False)
