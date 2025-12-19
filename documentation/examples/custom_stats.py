"""
Example: Custom statistics (generic stat_fn)

Goal
----
Show how to run randomization inference (RI) with *custom statistics* by passing
a `stat_fn(df_perm) -> float`. The model is whatever produces your statistic.

Stats covered (n=100, reps=500)
-------------------------------
1) Median difference (treated - control)
2) Trimmed mean difference (treated - control)
3) Rank-based (1): difference in mean ranks (treated - control)
4) Rank-based (2): rank-biserial correlation (effect size in [-1, 1])
5) Kolmogorov–Smirnov (two-sample) statistic (distributional difference)

Outputs
-------
Writes one TXT per statistic to `output/` (relative to this script).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, trim_mean

from ritest import ritest

HERE = Path(__file__).resolve().parent


# ----------------------------
# toy data (n=100)
# ----------------------------
def make_data(n: int = 100, seed: int = 1, tau: float = 0.6) -> pd.DataFrame:
    """
    Simple randomized treatment with one baseline covariate.
    Outcome is continuous with mild skew/heavy tails so distributional stats make sense.
    """
    rng = np.random.default_rng(seed)

    # Complete randomization: exactly n/2 treated
    z = np.zeros(n, dtype=int)
    z[: n // 2] = 1
    z = rng.permutation(z)

    x = rng.normal(size=n)

    # Mixture noise for heavier tails
    u = rng.uniform(size=n)
    eps = np.where(
        u < 0.85, rng.normal(scale=1.0, size=n), rng.normal(scale=3.0, size=n)
    )

    # Treatment shifts the location a bit (tau), plus baseline x effect
    y = 0.2 + tau * z + 0.5 * x + eps

    return pd.DataFrame({"y": y, "z": z, "x": x})


# ----------------------------
# helpers
# ----------------------------
def _split_groups(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=int)
    yt = y[z == 1]
    yc = y[z == 0]
    return yt, yc


def _ks_stat_and_pvalue(yt: np.ndarray, yc: np.ndarray) -> tuple[float, float]:
    r: Any = ks_2samp(yt, yc, alternative="two-sided")
    # works whether r is tuple-like or has attributes
    try:
        stat = float(r.statistic)
        pval = float(r.pvalue)
    except AttributeError:
        stat = float(r[0])
        pval = float(r[1])
    return stat, pval


# ----------------------------
# custom stats
# ----------------------------
def stat_median_diff(df: pd.DataFrame) -> float:
    """Median(y | z=1) - Median(y | z=0)."""
    yt, yc = _split_groups(df)
    return float(np.median(yt) - np.median(yc))


def stat_trimmed_mean_diff(df: pd.DataFrame, prop: float = 0.10) -> float:
    """
    Trimmed-mean difference (treated - control).
    prop=0.10 means 10% trimmed from each tail within each group.
    """
    yt, yc = _split_groups(df)
    mt = float(trim_mean(yt, proportiontocut=prop))
    mc = float(trim_mean(yc, proportiontocut=prop))
    return float(mt - mc)


def stat_mean_rank_diff(df: pd.DataFrame) -> float:
    """
    Difference in mean ranks (treated - control) computed on pooled ranks of y.
    Centered around 0 under the sharp null, works well with two-sided RI.
    """
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=int)
    r = rankdata(y, method="average")  # ranks 1..n
    return float(r[z == 1].mean() - r[z == 0].mean())


def stat_rank_biserial(df: pd.DataFrame) -> float:
    """
    Rank-biserial correlation in [-1, 1].
    Uses U derived from ranks:
      U = sum_ranks_t - n_t(n_t+1)/2
      rbc = 2U/(n_t*n_c) - 1
    Positive means treated values tend to be larger.
    """
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=int)
    r = rankdata(y, method="average")

    nt = int((z == 1).sum())
    nc = int((z == 0).sum())

    sum_ranks_t = float(r[z == 1].sum())
    u = sum_ranks_t - nt * (nt + 1) / 2.0
    rbc = 2.0 * u / (nt * nc) - 1.0
    return float(rbc)


def stat_ks(df: pd.DataFrame) -> float:
    """
    Two-sample Kolmogorov–Smirnov statistic D (>= 0).
    Larger means more separation between the treated and control distributions.
    """
    yt, yc = _split_groups(df)
    stat, _ = _ks_stat_and_pvalue(yt, yc)
    return float(stat)


# ----------------------------
# run helpers
# ----------------------------
Alt = Literal["two-sided", "left", "right"]


@dataclass(frozen=True)
class StatSpec:
    name: str
    description: str
    stat_fn: Callable[[pd.DataFrame], float]
    alternative: Alt  # "two-sided", "left", "right"
    extra_compare: Callable[[pd.DataFrame], str] | None = None  # optional text block


def _write_report(
    *, spec: StatSpec, df: pd.DataFrame, res, out_path: Path, runtime_seconds: float
) -> None:
    stat_obs = float(spec.stat_fn(df))

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Example: {spec.name}\n")
        f.write("=" * (9 + len(spec.name)) + "\n\n")

        f.write("Setup\n")
        f.write("-----\n")
        f.write(f"n = {len(df)}\n")
        f.write(f"reps = {res.reps}\n")
        f.write(f"seed = {res.settings.get('seed', 'unknown')}\n")
        f.write("permuted variable: z\n")
        f.write(f"alternative: {spec.alternative}\n")
        f.write(f"runtime (seconds): {runtime_seconds:.3f}\n\n")

        f.write("Statistic\n")
        f.write("---------\n")
        f.write(spec.description.strip() + "\n\n")

        f.write("Observed statistic (original data)\n")
        f.write("--------------------------------\n")
        f.write(f"stat_obs: {stat_obs:.6f}\n\n")

        if spec.extra_compare is not None:
            f.write("Model-based / asymptotic comparison (optional)\n")
            f.write("--------------------------------------------\n")
            f.write(spec.extra_compare(df).rstrip() + "\n\n")

        f.write("Randomization inference result (ritest)\n")
        f.write("--------------------------------------\n")
        f.write(res.summary(print_out=False))
        f.write("\n")


def _run_ri(df: pd.DataFrame, spec: StatSpec, *, seed: int):
    t0 = perf_counter()
    res = ritest(
        df=df,
        permute_var="z",
        stat_fn=spec.stat_fn,
        alternative=spec.alternative,
        reps=500,
        seed=seed,
        ci_method="clopper-pearson",
        ci_mode="none",
    )
    t1 = perf_counter()
    return res, (t1 - t0)


# ----------------------------
# optional “comparison” blocks for 2 stats
# ----------------------------
def _compare_wilcoxon_from_ranks(df: pd.DataFrame) -> str:
    """
    For rank-based stats: show the asymptotic Mann–Whitney / Wilcoxon p-value.
    This is *not* the RI p-value; it is a standard large-sample approximation.
    """
    from scipy.stats import mannwhitneyu

    yt, yc = _split_groups(df)
    test = mannwhitneyu(yt, yc, alternative="two-sided", method="asymptotic")
    return f"Mann–Whitney U p-value (asymptotic): {float(test.pvalue):.6f}"


def _compare_ks_asymptotic(df: pd.DataFrame) -> str:
    yt, yc = _split_groups(df)
    _, pval = _ks_stat_and_pvalue(yt, yc)
    return f"KS p-value (asymptotic): {pval:.6f}"


# ----------------------------
# main
# ----------------------------
def main() -> None:
    out_dir = HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = make_data(n=100, seed=1, tau=0.6)

    specs: list[StatSpec] = [
        StatSpec(
            name="Custom stat — Median difference",
            description="""
Median(y | z=1) - Median(y | z=0).

Use when you care about typical outcomes but want robustness to outliers.
""",
            stat_fn=stat_median_diff,
            alternative="two-sided",
        ),
        StatSpec(
            name="Custom stat — 10% trimmed mean difference",
            description="""
TrimmedMean_10%(y | z=1) - TrimmedMean_10%(y | z=0).

Use when you want a mean-like estimand but with reduced sensitivity to tails.
""",
            stat_fn=lambda d: stat_trimmed_mean_diff(d, prop=0.10),
            alternative="two-sided",
        ),
        StatSpec(
            name="Custom stat — Rank-based (1): difference in mean ranks",
            description="""
MeanRank(y | z=1) - MeanRank(y | z=0), computed from pooled ranks.

This is a simple “rank shift” statistic, centered around 0 under the sharp null.
""",
            stat_fn=stat_mean_rank_diff,
            alternative="two-sided",
            extra_compare=_compare_wilcoxon_from_ranks,
        ),
        StatSpec(
            name="Custom stat — Rank-based (2): rank-biserial correlation",
            description="""
Rank-biserial correlation (RBC), an effect size in [-1, 1].

Positive means treated outcomes tend to be larger than control outcomes.
""",
            stat_fn=stat_rank_biserial,
            alternative="two-sided",
            extra_compare=_compare_wilcoxon_from_ranks,
        ),
        StatSpec(
            name="Custom stat — Kolmogorov–Smirnov (KS) statistic",
            description="""
Two-sample KS statistic D (>=0): the maximum gap between empirical CDFs.

This is a pure distributional difference statistic. Large values are evidence
against the sharp null, so a right-tailed RI is natural.
""",
            stat_fn=stat_ks,
            alternative="right",
            extra_compare=_compare_ks_asymptotic,
        ),
    ]

    for i, spec in enumerate(specs, start=1):
        res, rt = _run_ri(df, spec, seed=20_000 + i)
        out_path = out_dir / f"example_custom_stats_{i:02d}.txt"
        _write_report(spec=spec, df=df, res=res, out_path=out_path, runtime_seconds=rt)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
