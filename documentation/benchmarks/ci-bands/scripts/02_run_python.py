# scripts/02_run_python.py

import time
from pathlib import Path
from typing import cast

import pandas as pd
from matplotlib.figure import Figure

from ritest import ritest


def main() -> None:
    # paths
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "ci_bands.csv"
    out_dir = root / "output"
    out_dir.mkdir(exist_ok=True)

    # load data
    df = pd.read_csv(data_path)

    # start timer
    t0 = time.perf_counter()

    # run ritest (linear path, CI band)
    res = ritest(
        df=df,
        permute_var="treatment",
        formula="y ~ treatment",
        stat="treatment",
        reps=500,
        ci_mode="grid",
        seed=123,
    )

    runtime_sec = time.perf_counter() - t0

    # save text summary
    summary = res.summary(print_out=False)
    summary += f"\nRuntime (seconds): {runtime_sec:.3f}\n"

    (out_dir / "python_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    # save CI band grid
    if res.coef_ci_band is not None:
        beta, pvals = res.coef_ci_band
        pd.DataFrame({"beta": beta, "pval": pvals}).to_csv(
            out_dir / "python_ci_band.csv", index=False
        )

    # save plot
    ax = res.plot(show=False)
    fig = cast(Figure, ax.get_figure())
    fig.savefig(out_dir / "python_ci_band.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
