#!/usr/bin/env python3
"""
Two-period DiD equivalence demo (statsmodels) + ritest fast linear path example.

What this script does:
1) Simulates a balanced 2-period panel (same units pre/post) with:
   - treated-group indicator D_i (time-invariant)
   - post indicator Post_t
   - one time-varying covariate X_it
2) Estimates DiD in the standard way:
      y ~ D + Post + D:Post + X
   and records the interaction coefficient (the DiD effect).
3) Transforms the data to one row per unit:
      dy_i = y_post - y_pre
      dx_i = x_post - x_pre
   then estimates:
      dy ~ D + dx
   and shows that the coefficient on D matches the interaction coefficient above.
4) Runs ritest on the *change-score* regression using the fast linear path.

Outputs:
- Writes a single text file to: output/did_equivalence.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ritest import ritest


def main() -> None:
    # -----------------------------
    # 1) Simulate a 2-period panel
    # -----------------------------
    rng = np.random.default_rng(123)

    n_units = 1000
    ids = np.arange(n_units)

    # Treatment assignment at the unit level (time-invariant)
    D = rng.integers(0, 2, size=n_units).astype(int)

    # Two time periods: 0 (pre), 1 (post)
    t = np.array([0, 1], dtype=int)

    # Expand to panel (2 rows per unit)
    df = pd.DataFrame(
        {
            "id": np.repeat(ids, 2),
            "post": np.tile(t, n_units),
        }
    )
    df["D"] = np.repeat(D, 2)

    # One time-varying covariate
    # (baseline per unit + a common post shift + idiosyncratic noise)
    x_base = rng.normal(0.0, 1.0, size=n_units)
    df["x"] = (
        np.repeat(x_base, 2)
        + 0.4 * df["post"].to_numpy()
        + rng.normal(0.0, 1.0, size=len(df))
    )

    # Data generating process
    alpha = 1.0
    gamma = 0.3
    lam = 0.2
    delta_true = 2.0
    beta = 1.5

    u_i = rng.normal(0.0, 1.0, size=n_units)  # time-invariant unit component
    eps = rng.normal(0.0, 1.0, size=len(df))

    df["y"] = (
        alpha
        + gamma * df["D"]
        + lam * df["post"]
        + delta_true * (df["D"] * df["post"])
        + beta * df["x"]
        + np.repeat(u_i, 2)
        + eps
    )

    # -------------------------------------------------------
    # 2) Standard DiD regression: y ~ D + post + D:post + x
    # -------------------------------------------------------
    res_levels = smf.ols("y ~ D + post + D:post + x", data=df).fit()
    did_levels = float(res_levels.params["D:post"])

    # -------------------------------------------------------
    # 3) Change-score transformation: dy ~ D + dx
    # -------------------------------------------------------
    wide_y = df.pivot(index="id", columns="post", values="y")
    wide_x = df.pivot(index="id", columns="post", values="x")

    df_diff = pd.DataFrame(
        {
            "id": ids,
            "D": D,
            "dy": (wide_y[1] - wide_y[0]).to_numpy(),
            "dx": (wide_x[1] - wide_x[0]).to_numpy(),
        }
    )

    res_diff = smf.ols("dy ~ D + dx", data=df_diff).fit()
    did_diff = float(res_diff.params["D"])

    # Numerical check (should be extremely close)
    abs_diff = abs(did_levels - did_diff)

    # -------------------------------------------------------
    # 4) ritest fast linear path on the change-score regression
    # -------------------------------------------------------
    ri = ritest(
        df=df_diff,
        permute_var="D",
        formula="dy ~ D + dx",
        stat="D",
        reps=2000,
        seed=123,
        alternative="two-sided",
        ci_mode="none",
    )

    # -----------------------------
    # 5) Write output
    # -----------------------------
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "did_equivalence.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Two-period DiD equivalence demo\n")
        f.write("===============================\n\n")

        f.write("A) Standard DiD regression (panel): y ~ D + post + D:post + x\n")
        f.write("------------------------------------------------------------\n")
        f.write(res_levels.summary().as_text())
        f.write("\n\n")
        f.write(f"DiD effect (interaction coefficient)  D:post = {did_levels:.10f}\n\n")

        f.write("B) Change-score regression (one row per unit): dy ~ D + dx\n")
        f.write("----------------------------------------------------------\n")
        f.write(res_diff.summary().as_text())
        f.write("\n\n")
        f.write(f"DiD effect (coefficient on D)         D      = {did_diff:.10f}\n")
        f.write(f"Absolute difference |D:post - D|             = {abs_diff:.3e}\n\n")

        f.write("C) ritest on the fast linear path (change-score form)\n")
        f.write("-----------------------------------------------------\n")
        f.write("ritest call used:\n")
        f.write(
            "  ritest(df_diff, permute_var='D', formula='dy ~ D + dx', stat='D', reps=2000, seed=123)\n\n"
        )
        f.write(ri.summary(print_out=False))
        f.write("\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
