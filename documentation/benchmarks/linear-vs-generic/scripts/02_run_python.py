#!/usr/bin/env python
"""
Linear vs generic benchmark — compare `ritest` linear vs generic (statsmodels) engines.

Inputs
- data/simulated_data.csv   (created by 01_make_simulated_data.py)

Outputs
- output/python_linear.txt
- output/python_generic.txt

Default RI settings
- reps = 2000
- seed = 23
- permute_var = "treat"
- stat = "treat"

Run
  python documentation/benchmarks/linear-vs-generic/scripts/02_run_python_bench.py --engine both
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ritest import ritest

# -----------------------------
# Paths (keep relative to this benchmark folder)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
BENCH_DIR = THIS_DIR.parent

DEFAULT_DATA = BENCH_DIR / "data" / "simulated_data.csv"
DEFAULT_OUTDIR = BENCH_DIR / "output"

# -----------------------------
# Model specification
# -----------------------------
PERMUTE_VAR = "treat"
STAT = "treat"

COVARS = [
    "age",
    "female",
    "education_years",
    "log_income",
    "household_size",
    "urban",
    "tenure_months",
    "baseline_spend",
    "purchases_12m",
    "returns_12m",
    "support_tickets_6m",
    "app_sessions_30d",
    "days_since_last_purchase",
    "email_opt_in",
    "promo_exposure_30d",
    "prior_churn",
    "credit_score",
    "satisfaction_score",
    "region_1",
    "region_2",
    "region_3",
    "region_4",
]

FORMULA = "y ~ treat + " + " + ".join(COVARS)


def _versions_block() -> str:
    import pandas as _pd
    import statsmodels as _sm

    return "\n".join(
        [
            f"Python: {platform.python_version()}",
            f"Platform: {platform.platform()}",
            f"pandas: {_pd.__version__}",
            f"statsmodels: {_sm.__version__}",
        ]
    )


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"y", "treat", *COVARS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Enforce clean numeric types (the generator should already do this, but I keep it explicit here).
    df["y"] = pd.to_numeric(df["y"], errors="raise")
    df["treat"] = pd.to_numeric(df["treat"], errors="raise").astype("int8")

    for c in COVARS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if df.isna().any().any():
        raise ValueError("Unexpected missing values found in simulated_data.csv")

    # Sanity: binary treatment
    vals = set(pd.unique(df["treat"]))
    if vals != {0, 1}:
        raise ValueError(f"treat must be coded as 0/1. Found: {sorted(vals)}")

    return df


def _write_log(
    out_path: Path, header_lines: list[str], body: str, runtime_s: float
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(header_lines)
    out_path.write_text(
        header + "\n\n" + body + "\n" + f"\nRuntime (seconds): {runtime_s:.3f}\n",
        encoding="utf-8",
    )


def run_linear(df: pd.DataFrame, reps: int, seed: int, outdir: Path) -> None:
    t0 = time.perf_counter()
    res = ritest(
        df=df,
        permute_var=PERMUTE_VAR,
        formula=FORMULA,
        stat=STAT,
        reps=reps,
        seed=seed,
        ci_mode="none",
    )
    dt = time.perf_counter() - t0

    summary = res.summary(print_out=False)

    _write_log(
        out_path=outdir / "python_linear.txt",
        header_lines=[
            "Benchmark: Linear vs generic (Python)",
            "Engine: linear (FastOLS path via formula=...)",
            f"Data: {DEFAULT_DATA.name}",
            f"N: {len(df):,}",
            f"Reps: {reps}",
            f"Seed: {seed}",
            f"Formula: {FORMULA}",
            "",
            _versions_block(),
        ],
        body=summary,
        runtime_s=dt,
    )

    print(summary)
    print(f"Runtime (seconds): {dt:.3f}")


def run_generic_statsmodels(
    df: pd.DataFrame, reps: int, seed: int, outdir: Path
) -> None:
    """
    Generic engine: I call statsmodels OLS inside stat_fn and return the treat coefficient.

    Performance note:
    - This intentionally re-fits OLS for each permutation, so it is expected to be slow for large N.
    """
    y = df["y"].to_numpy(dtype=float, copy=False)

    # Precompute the non-treatment part of the design matrix once.
    x_other = df[COVARS].to_numpy(dtype=float, copy=False)

    # Build one design matrix and overwrite the treatment column for each permutation.
    # Column order: const, treat, covars...
    n = len(df)
    X = np.empty((n, 2 + x_other.shape[1]), dtype=float)
    X[:, 0] = 1.0
    X[:, 2:] = x_other

    def stat_fn(dfp: pd.DataFrame) -> float:
        X[:, 1] = dfp["treat"].to_numpy(dtype=float, copy=False)
        fit = sm.OLS(y, X).fit()
        return float(fit.params[1])  # treat coefficient (const is params[0])

    t0 = time.perf_counter()
    res = ritest(
        df=df,
        permute_var=PERMUTE_VAR,
        stat_fn=stat_fn,
        reps=reps,
        seed=seed,
        ci_mode="none",
    )
    dt = time.perf_counter() - t0

    summary = res.summary(print_out=False)

    _write_log(
        out_path=outdir / "python_generic.txt",
        header_lines=[
            "Benchmark: Linear vs generic (Python)",
            "Engine: generic (stat_fn=..., statsmodels OLS each permutation)",
            f"Data: {DEFAULT_DATA.name}",
            f"N: {len(df):,}",
            f"Reps: {reps}",
            f"Seed: {seed}",
            "Model: y ~ treat + " + " + ".join(COVARS),
            "",
            _versions_block(),
        ],
        body=summary,
        runtime_s=dt,
    )

    print(summary)
    print(f"Runtime (seconds): {dt:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["linear", "generic", "both"],
        default="both",
        help="Which engine(s) to run.",
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA), help="Path to simulated_data.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Output directory for .txt logs",
    )
    parser.add_argument(
        "--reps", type=int, default=2000, help="RI repetitions (default: 2000)"
    )
    parser.add_argument(
        "--seed", type=int, default=23, help="Random seed (default: 23)"
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()

    df = _load_data(data_path)

    if args.engine in {"linear", "both"}:
        run_linear(df=df, reps=args.reps, seed=args.seed, outdir=outdir)

    if args.engine in {"generic", "both"}:
        run_generic_statsmodels(df=df, reps=args.reps, seed=args.seed, outdir=outdir)


if __name__ == "__main__":
    main()
