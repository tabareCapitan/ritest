#!/usr/bin/env python
"""
Colombia benchmark — Python `ritest`, matching McDermott’s R example.

McDermott model (fixest):
  feols(
    dayscorab ~ b_treat + b_dayscorab | b_pair + miss_b_dayscorab + round2 + round3,
    vcov = ~b_block, data = colombia
  )

Python equivalent (Patsy, linear path):
  dayscorab ~ b_treat + b_dayscorab
    + C(b_pair) + C(miss_b_dayscorab) + C(round2) + C(round3)

Generic equivalent (pyfixest, absorbed FE as estimator):
  dayscorab ~ b_treat + b_dayscorab | b_pair + miss_b_dayscorab + round2 + round3

Engines:
- linear: ritest with a Patsy-built linear model (FastOLS path).
- generic: ritest with a pyfixest `feols` stat function (absorbed FE estimator).

RI design:
  permute_var = b_treat
  strata      = b_pair
  cluster     = b_block
  reps        = 5000
  seed        = 546

Data handling (minimal, correctness-critical):
- b_treat must be numeric/boolean for ritest validation.
- dayscorab must be numeric; exported CSV may contain string numbers and missing codes (e.g. ".r").
  Coerce to numeric and let the estimator handle NAs:
    - Patsy handles NA dropping in the formula path
    - pyfixest handles NA dropping in feols
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ritest import ritest

# -----------------------------
# Paths
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
CO_DIR = THIS_DIR.parent

DATA = CO_DIR / "data" / "colombia.csv"
OUT_TABLES = CO_DIR / "output" / "logs"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# -----------------------------
# McDermott specification
# -----------------------------
FORMULA = (
    "dayscorab ~ b_treat + b_dayscorab "
    "+ C(b_pair) + C(miss_b_dayscorab) + C(round2) + C(round3)"
)

FML_PYFIXEST = (
    "dayscorab ~ b_treat + b_dayscorab | b_pair + miss_b_dayscorab + round2 + round3"
)

PERMUTE_VAR = "b_treat"
STAT = "b_treat"
CLUSTER = "b_block"
STRATA = "b_pair"
REPS = 5000
SEED = 546


def _coerce_dayscorab_numeric(df: pd.DataFrame) -> None:
    if "dayscorab" not in df.columns:
        raise ValueError("Missing required column: dayscorab")
    df["dayscorab"] = pd.to_numeric(df["dayscorab"], errors="coerce")


def _recode_b_treat(df: pd.DataFrame) -> None:
    """
    Ensure b_treat is a 0/1 numeric indicator with 1 = Treatment, 0 = Control.

    Handles:
    - Strings: 'Treatment'/'Control' (case/space robust)
    - Numeric two-level codes: {1,2} where 1=Treatment and 2=Control
    - Already {0,1} (kept)
    """
    if PERMUTE_VAR not in df.columns:
        raise ValueError("Missing column: b_treat")

    s = df[PERMUTE_VAR]

    # Case 1: strings like "Treatment"/"Control"
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        mapped = (
            s.astype(str).str.strip().str.lower().map({"treatment": 1, "control": 0})
        )
        if mapped.isna().any():
            bad = sorted(s.loc[mapped.isna()].astype(str).unique())
            raise ValueError(
                "Unexpected b_treat labels. Expected 'Treatment'/'Control'. "
                f"Unexpected: {bad}"
            )
        df[PERMUTE_VAR] = mapped.astype("int8")
        return

    # Case 2: numeric/bool
    sn = pd.to_numeric(s, errors="coerce")
    vals = pd.unique(sn.dropna())

    if len(vals) != 2:
        raise ValueError(
            f"b_treat must have exactly 2 unique non-missing values; got {sorted(vals)}"
        )

    if set(vals) == {0, 1}:
        df[PERMUTE_VAR] = sn.astype("Int64").astype("int8")
        return

    if set(vals) == {1, 2}:
        df[PERMUTE_VAR] = sn.map({1: 1, 2: 0}).astype("Int64").astype("int8")
        return

    raise ValueError(
        "b_treat has unexpected numeric coding. Expected {0,1} or {1,2} (1=Treatment,2=Control). "
        f"Got {sorted(vals)}"
    )


def _write_summary(out_path: Path, summary: str, runtime_s: float) -> None:
    out_path.write_text(
        summary + "\n" + f"Runtime (seconds): {runtime_s:.3f}\n", encoding="utf-8"
    )


def run_linear(df: pd.DataFrame) -> None:
    t0 = time.perf_counter()
    res = ritest(
        df=df,
        permute_var=PERMUTE_VAR,
        formula=FORMULA,
        stat=STAT,
        cluster=CLUSTER,
        strata=STRATA,
        reps=REPS,
        seed=SEED,
    )
    dt = time.perf_counter() - t0

    summary = res.summary(print_out=False)
    print(summary)
    print(f"Runtime (seconds): {dt:.3f}")

    _write_summary(OUT_TABLES / "colombia_ritest_summary.txt", summary, dt)


def run_generic_pyfixest(df: pd.DataFrame) -> None:
    try:
        import pyfixest as pf
    except ImportError as e:
        raise ImportError(
            "pyfixest is required for the generic benchmark: pip install pyfixest"
        ) from e

    def stat_fn(dfp: pd.DataFrame) -> float:
        # Cluster affects SEs, not the point estimate; included for parity with the R call.
        fit = pf.feols(fml=FML_PYFIXEST, data=dfp, vcov={"CRV1": CLUSTER})
        return float(fit.coef().loc[STAT])

    t0 = time.perf_counter()
    res = ritest(
        df=df,
        permute_var=PERMUTE_VAR,
        stat_fn=stat_fn,
        cluster=CLUSTER,
        strata=STRATA,
        reps=REPS,
        seed=SEED,
    )
    dt = time.perf_counter() - t0

    summary = res.summary(print_out=False)
    print(summary)
    print(f"Runtime (seconds): {dt:.3f}")

    _write_summary(OUT_TABLES / "colombia_ritest_summary_pyfixest.txt", summary, dt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["linear", "generic", "both"],
        default="linear",
        help="Which benchmark engine(s) to run. 'both' runs linear+generic.",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA)

    _coerce_dayscorab_numeric(df)
    _recode_b_treat(df)

    if args.engine in {"linear", "both"}:
        run_linear(df)

    if args.engine in {"generic", "both"}:
        run_generic_pyfixest(df)


if __name__ == "__main__":
    main()
