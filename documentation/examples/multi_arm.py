"""
Multi-arm (3-arm) experiment demo: A vs Control via ritest (Python)

This script shows two sensible randomization-inference (RI) setups when you have
three arms:

    T = 0 (Control), 1 (Treatment A), 2 (Treatment B)

Goal: test/estimate the A-vs-Control contrast using the linear path in ritest.

Option 1 (drop B):
- Restrict the dataset to units with T in {0,1}
- Permute A/Control labels among those units
- Estimate: y ~ 1 + A + x1 + x2

Option 2 (keep B in estimation, fix B in re-randomization):
- Use the full dataset for estimation
- Include a separate indicator for B in the regression (do not pool B into Control)
- Permute A/Control labels only among units with T in {0,1}, keeping all T=2 units fixed
- Estimate: y ~ 1 + A + B + x1 + x2
- Implement the “fix B” idea by supplying a custom permutation matrix to ritest.

Outputs
-------
Writes two text files to: ./output/  (relative to this script)

- multiarm_drop_B.txt
- multiarm_keep_B_fixed_in_RI.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ritest import ritest


def simulate_data(n: int = 1000, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Random assignment into 3 arms (roughly balanced)
    T = rng.choice([0, 1, 2], size=n, replace=True, p=[1 / 3, 1 / 3, 1 / 3]).astype(int)

    # Two covariates with arm-specific means (to make “using all arms” matter for covariate fit)
    x1 = rng.normal(loc=0.0, scale=1.0, size=n) + 0.6 * (T == 2) - 0.3 * (T == 1)
    x2 = rng.normal(loc=0.0, scale=1.0, size=n) + 0.4 * (T == 1) - 0.2 * (T == 2)

    # True arm effects
    tau_A = 0.8
    tau_B = 0.2

    # Outcome model (simple, additive)
    eps = rng.normal(loc=0.0, scale=1.0, size=n)
    y = 1.0 + tau_A * (T == 1) + tau_B * (T == 2) + 1.2 * x1 - 0.7 * x2 + eps

    df = pd.DataFrame({"treatment": T, "x1": x1, "x2": x2, "y": y})
    df["A"] = (df["treatment"] == 1).astype(int)
    df["B"] = (df["treatment"] == 2).astype(int)

    return df


def write_option1(df: pd.DataFrame, out_path: Path) -> None:
    """
    Option 1: drop Treatment B entirely; run RI on the {Control, A} subsample.
    """
    df_sub = df[df["treatment"].isin([0, 1])].copy()

    # Observed statistic (OLS coefficient on A)
    ols = smf.ols("y ~ 1 + A + x1 + x2", data=df_sub).fit()
    beta_hat = float(ols.params["A"])

    # Randomization inference (permute A among these units)
    ri = ritest(
        df=df_sub,
        permute_var="A",
        formula="y ~ 1 + A + x1 + x2",
        stat="A",
        reps=2000,
        seed=123,
        alternative="two-sided",
        ci_mode="none",
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Option 1: Drop Treatment B (use only Control and A)\n")
        f.write("==================================================\n\n")
        f.write("Estimation sample: treatment in {0,1}\n")
        f.write("Re-randomization: permute A labels within that sample\n\n")

        f.write("A) OLS on the estimation sample\n")
        f.write("-------------------------------\n")
        f.write(ols.summary().as_text())
        f.write("\n\n")
        f.write(f"Observed coefficient on A: {beta_hat:.6f}\n\n")

        f.write("B) Randomization inference (ritest)\n")
        f.write("----------------------------------\n")
        f.write("ritest call used:\n")
        f.write(
            "  ritest(df_sub, permute_var='A', formula='y ~ 1 + A + x1 + x2', stat='A', reps=2000, seed=123)\n\n"
        )
        f.write(ri.summary(print_out=False))
        f.write("\n")


def write_option2(df: pd.DataFrame, out_path: Path) -> None:
    """
    Option 2: keep Treatment B in estimation (as its own indicator),
              but fix B in re-randomization by supplying custom permutations.
    """
    n = len(df)

    # Observed statistic (OLS coefficient on A) using the full sample
    ols = smf.ols("y ~ 1 + A + B + x1 + x2", data=df).fit()
    beta_hat = float(ols.params["A"])

    # Build a permutation matrix that only permutes A among units that are not in B.
    # Units in B always have A=0 and remain fixed across permutations.
    A = df["A"].to_numpy(dtype=np.int8)
    mask = df["B"].to_numpy(dtype=np.int8) == 0  # eligible: Control or A

    reps = 2000
    rng = np.random.default_rng(123)
    perms = np.empty((reps, n), dtype=np.int8)

    for r in range(reps):
        pr = A.copy()
        pr[mask] = rng.permutation(pr[mask])
        perms[r] = pr

    ri = ritest(
        df=df,
        permute_var="A",
        permutations=perms,  # overrides internal shuffling
        formula="y ~ 1 + A + B + x1 + x2",
        stat="A",
        reps=reps,  # optional; perms controls the effective reps
        seed=123,
        alternative="two-sided",
        ci_mode="none",
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Option 2: Keep Treatment B in estimation, fix B in re-randomization\n")
        f.write(
            "==================================================================\n\n"
        )
        f.write("Estimation sample: all units (treatment in {0,1,2})\n")
        f.write("Regression includes B indicator so B is not pooled into Control\n")
        f.write(
            "Re-randomization: permute A labels only among units with treatment in {0,1}\n\n"
        )

        f.write("A) OLS on the full sample\n")
        f.write("-------------------------\n")
        f.write(ols.summary().as_text())
        f.write("\n\n")
        f.write(f"Observed coefficient on A: {beta_hat:.6f}\n\n")

        f.write("B) Randomization inference (ritest)\n")
        f.write("----------------------------------\n")
        f.write("ritest call used (conceptually):\n")
        f.write(
            "  ritest(df, permute_var='A', permutations=perms, formula='y ~ 1 + A + B + x1 + x2', stat='A', reps=2000, seed=123)\n\n"
        )
        f.write("Notes on permutations used:\n")
        f.write("- perms is a (reps x n) matrix of A assignments\n")
        f.write("- entries for B units are always 0 (B is fixed)\n")
        f.write(
            "- within {Control, A}, the number of A units is preserved each draw\n\n"
        )
        f.write(ri.summary(print_out=False))
        f.write("\n")


def main() -> None:
    df = simulate_data(n=1000, seed=123)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_option1(df, out_dir / "multiarm_drop_B.txt")
    write_option2(df, out_dir / "multiarm_keep_B_fixed_in_RI.txt")

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
