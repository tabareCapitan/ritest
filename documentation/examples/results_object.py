"""
Example: Working with RitestResult (results container)

- Generate a tiny clean dataset (n=1000), binary treatment, no covariates.
- Run linear-path ritest with coefficient CI bounds.
- Show how to extract pieces from the RitestResult object and reuse them.
- Write a labeled, easy-to-read TXT report.

Assumes: `ritest` is installed and importable (pip install / editable install).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from ritest import ritest

HERE = Path(__file__).resolve().parent


def make_data(
    n: int = 1000, seed: int = 123, tau: float = 1.0, sigma: float = 1.0
) -> pd.DataFrame:
    """
    Create a simple dataset:
      y = tau * z + noise
    where z is randomized 0/1 with equal probability.
    """
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)  # 0/1 treatment
    y = tau * z + rng.normal(loc=0.0, scale=sigma, size=n)
    return pd.DataFrame({"y": y, "z": z})


def main() -> None:
    # ----------------------------
    # 1) data
    # ----------------------------
    df = make_data(n=1000, seed=1, tau=1.0, sigma=1.0)

    # ----------------------------
    # 2) run randomization inference (linear path)
    # ----------------------------
    # - permute_var: the assignment variable to permute
    # - formula/stat: linear path (fast OLS engine), no covariates
    # - ci_mode="bound": compute coefficient CI bounds
    # - reps kept at 1000 for speed in docs
    res = ritest(
        df=df,
        permute_var="z",
        formula="y ~ z",
        stat="z",
        alternative="two-sided",
        reps=1000,
        seed=23,
        ci_mode="bounds",
        ci_range=3.0,
        ci_step=0.05,
    )

    # ----------------------------
    # 3) extract key pieces (store in variables, print them)
    # ----------------------------
    # Core observed estimate and RI p-value
    bhat = float(res.obs_stat)
    p_ri = float(res.pval)

    # P-value CI
    p_ci = res.pval_ci  # (lo, hi) or None

    # Coefficient CI bounds (lo, hi) or None
    b_ci = res.coef_ci_bounds

    # Coefficient CI band (beta_grid, pvals) or None
    band = res.coef_ci_band

    # Example: unpack the band safely
    if band is not None:
        beta_grid, pvals = band
        beta_grid = np.asarray(beta_grid, dtype=float)
        pvals = np.asarray(pvals, dtype=float)
    else:
        beta_grid, pvals = np.array([]), np.array([])

    # ----------------------------
    # 4) write a clean report
    # ----------------------------
    out_dir = HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "example_results_object.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Example: Working with RitestResult\n")
        f.write("=" * 34 + "\n\n")

        f.write("A) Quick object display\n")
        f.write("-" * 24 + "\n")
        f.write(f"repr(res): {repr(res)}\n")
        f.write(f"str(res):  {str(res)}\n\n")

        f.write("B) Full summary() output\n")
        f.write("-" * 24 + "\n")
        f.write(res.summary(print_out=False))
        f.write("\n\n")

        f.write("C) Plain-language explain() output\n")
        f.write("-" * 34 + "\n")
        f.write(res.explain() + "\n\n")

        f.write("D) Extracting specific fields into variables\n")
        f.write("-" * 46 + "\n")
        f.write(f"Observed coefficient (bhat): {bhat}\n")
        f.write(f"RI p-value (p_ri):          {p_ri}\n")
        f.write(f"P-value CI (p_ci):          {p_ci}\n")
        f.write(f"Coef CI bounds (b_ci):      {b_ci}\n")
        f.write("\n")

        f.write("E) Working with the coefficient CI band (beta_grid, pvals)\n")
        f.write("-" * 62 + "\n")
        if band is None:
            f.write("coef_ci_band is None (band not computed)\n")
        else:
            f.write(f"Band length: {beta_grid.size}\n")
            f.write("First 10 grid points:\n")
            for i in range(min(10, beta_grid.size)):
                f.write(f"  beta={beta_grid[i]: .6f}  p(beta)={pvals[i]: .6f}\n")
        f.write("\n")

        f.write("F) Settings snapshot stored on the result\n")
        f.write("-" * 42 + "\n")
        # res.settings is intended to be stable enough to print
        for k in sorted(res.settings.keys()):
            f.write(f"{k}: {res.settings[k]}\n")

    # Also print the key extracted values to stdout (minimal, for quick runs)
    print(f"Wrote: {out_path}")
    print(f"bhat={bhat}  p_ri={p_ri}  b_ci={b_ci}  p_ci={p_ci}")


if __name__ == "__main__":
    main()
