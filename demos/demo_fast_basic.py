#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Literal, Tuple, cast

import numpy as np
import pandas as pd

# Narrow type for the test tail (to satisfy Pylance)
Alt = Literal["two-sided", "left", "right"]

# Use a non-interactive backend by default to avoid blocking GUI windows
os.environ.setdefault("MPLBACKEND", "Agg")

from ritest import ritest  # relies on package defaults unless overridden


# ---------- data ----------
def make_data(n: int, beta_T: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    T = rng.integers(0, 2, size=n)  # 0/1 treatment
    y = beta_T * T + 0.4 * X1 - 0.2 * X2 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "T": T, "X1": X1, "X2": X2})


# ---------- helpers ----------
def ols_T_coef(df: pd.DataFrame) -> Tuple[float, float, Tuple[float, float]]:
    """
    Return (beta_T_hat, se_T, (ci_lo, ci_hi)) for y ~ 1 + T + X1 + X2 using numpy.
    """
    y = df["y"].to_numpy()
    X = np.column_stack(
        [np.ones(len(df)), df["T"].to_numpy(), df["X1"].to_numpy(), df["X2"].to_numpy()]
    )
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    dof = len(df) - X.shape[1]
    s2 = float((resid @ resid) / dof)
    cov = s2 * np.linalg.inv(XtX)
    se_T = float(np.sqrt(cov[1, 1]))
    beta_T_hat = float(beta[1])
    z = 1.96
    return beta_T_hat, se_T, (beta_T_hat - z * se_T, beta_T_hat + z * se_T)


def header(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


# ---------- runs ----------
def run_minimal(df: pd.DataFrame, reps: int, seed: int, alt: Alt) -> None:
    header(f"Run A [{alt}]: minimal (ci_mode='none')")
    bT, seT, (ci_lo, ci_hi) = ols_T_coef(df)
    treated = int(df["T"].sum())
    print(f"n={len(df)}  treated={treated} ({treated/len(df):.1%})")
    print(f"OLS beta_T: {bT:.6g}  (SE {seT:.6g}; 95% CI [{ci_lo:.6g}, {ci_hi:.6g}])")

    res = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        alternative=alt,
        reps=reps,
        seed=seed,
        ci_mode="none",
        n_jobs=4,
    )

    lo, hi = cast(Tuple[float, float], res.pval_ci)
    print(res.summary(print_out=False))
    print(f"obs_stat (β̂ for T): {res.obs_stat:.6g}")
    print(f"pval:               {res.pval:.6g}   pval_ci: ({lo:.6g}, {hi:.6g})")
    print(f"as-or-more-extreme: {res.c} / {res.reps}")
    print(f"coef_ci_bounds:     {res.coef_ci_bounds}")
    print(f"coef_ci_band:       {res.coef_ci_band}")


def run_bounds(df: pd.DataFrame, reps: int, seed: int, alt: Alt) -> None:
    header(f"Run B [{alt}]: coefficient CI bounds (ci_mode='bounds')")
    bT, seT, (ci_lo, ci_hi) = ols_T_coef(df)
    print(f"OLS beta_T: {bT:.6g}  (SE {seT:.6g}; 95% CI [{ci_lo:.6g}, {ci_hi:.6g}])")

    res = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        alternative=alt,
        reps=reps,
        seed=seed,
        ci_mode="bounds",
        n_jobs=4,
    )

    lo, hi = cast(Tuple[float, float], res.pval_ci)
    print(res.summary(print_out=False))
    print(f"pval: {res.pval:.6g}   pval_ci: ({lo:.6g}, {hi:.6g})")
    print(f"coef_ci_bounds (beta_T): {res.coef_ci_bounds}")
    print(f"coef_ci_band: {res.coef_ci_band}")


def run_band(
    df: pd.DataFrame, grid_reps: int, seed: int, alt: Alt, plot: bool, outpath: str
) -> None:
    header(f"Run C [{alt}]: coefficient band (ci_mode='grid')")
    bT, seT, (ci_lo, ci_hi) = ols_T_coef(df)
    print(f"OLS beta_T: {bT:.6g}  (SE {seT:.6g}; 95% CI [{ci_lo:.6g}, {ci_hi:.6g}])")

    res = ritest(
        df=df,
        permute_var="T",
        formula="y ~ T + X1 + X2",
        stat="T",
        alternative=alt,
        reps=grid_reps,
        seed=seed,
        ci_mode="grid",
        n_jobs=4,
    )

    lo, hi = cast(Tuple[float, float], res.pval_ci)
    print(res.summary(print_out=False))
    print(f"pval: {res.pval:.6g}   pval_ci: ({lo:.6g}, {hi:.6g})")

    band = res.coef_ci_band
    if band is None:
        print("band: None (unexpected for fast-linear 'grid')")
        return

    # Current API: tuple (beta_grid, pvals)
    beta_grid, pvals = cast(Tuple[np.ndarray, np.ndarray], band)
    idx_min = int(np.argmin(pvals))
    beta_at_min = float(beta_grid[idx_min])
    p_min = float(pvals[idx_min])

    print(
        f"band grid: size={len(beta_grid)}, "
        f"range=[{float(beta_grid.min()):.6g}, {float(beta_grid.max()):.6g}]"
    )
    print(
        f"min p(beta)={p_min:.6g} at beta={beta_at_min:.6g} "
        f"(distance to OLS β̂: {abs(beta_at_min - bT):.6g})"
    )

    # Plot non-blocking or save
    import matplotlib.pyplot as plt

    ax = res.plot()
    ax.set_title(f"Permutation-based coefficient band for beta_T — alt={alt}")
    if plot:
        # Non-blocking show to avoid stuck GUI loops
        plt.show(block=False)
        plt.pause(2.0)
        plt.close()
    else:
        # include alt in filename to avoid overwriting
        root, ext = os.path.splitext(outpath)
        out = f"{root}_{alt.replace('-', '')}{ext}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"(saved figure to {out})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ritest fast-OLS demo")
    ap.add_argument("-n", "--n", type=int, default=1000, help="sample size")
    ap.add_argument("--beta", type=float, default=0.002, help="true treatment effect")
    ap.add_argument("--seed", type=int, default=2025, help="rng seed for data")
    ap.add_argument(
        "--alt",
        type=str,
        default="two-sided",
        choices=["two-sided", "left", "right"],
        help="test tail: two-sided | left | right",
    )
    ap.add_argument("--reps", type=int, default=2000, help="permutation reps for A/B")
    ap.add_argument("--grid-reps", type=int, default=1500, help="permutation reps for band")
    ap.add_argument("--plot", action="store_true", help="show plot interactively (non-blocking)")
    ap.add_argument(
        "--out", type=str, default="demos/band.png", help="output path for saved band figure"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = make_data(n=args.n, beta_T=args.beta, seed=args.seed)

    print(f"\nDataset: n={len(df)}, true beta_T={args.beta}, noise sd≈1")

    alt: Alt = cast(Alt, args.alt)  # narrow CLI string to the Literal type

    run_minimal(df, reps=args.reps, seed=123, alt=alt)
    # run_bounds(df, reps=args.reps, seed=123, alt=alt)
    # run_band(df, grid_reps=args.grid_reps, seed=123, alt=alt, plot=args.plot, outpath=args.out)


if __name__ == "__main__":
    main()
