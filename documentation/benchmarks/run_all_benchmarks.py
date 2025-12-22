"""
Run all Python benchmark scripts under:

  documentation/benchmarks/**/scripts/*.py

…in stable “best conditions” for timing.

Why this exists
---------------
NumPy-heavy workloads (like RI with many linear algebra calls) can have huge run-to-run
swings if BLAS decides to use many threads (oversubscription). This runner:

1) Forces single-thread BLAS/NumExpr (very often faster + much more stable).
2) Runs each benchmark in a fresh Python process.
3) Repeats each script multiple times and *drops the first run* (to reduce first-run noise).
4) Writes a consolidated log + a CSV summary with min/median/max.


Common options
--------------
  # Total runs per script (includes the dropped run)
  python benchmarks/run_all_python_benchmarks.py --repeats 6 --drop-first 1

  # Include data generators like 01_make_simulated_data.py
  python benchmarks/run_all_python_benchmarks.py --include-generators

  # Pin to specific CPUs (Linux only), e.g. physical cores 0-7
  BENCH_CPUS=0-7 python benchmarks/run_all_python_benchmarks.py

Notes
-----
- "Best conditions" here means: avoid oversubscription + reduce jitter. It cannot control
  all OS background activity; for clean benchmarks, close heavy apps and avoid swap use.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

# -----------------------------
# Paths (derived from this file)
# -----------------------------
BENCHMARKS_DIR = Path(__file__).resolve().parent  # .../documentation/benchmarks
DOCS_DIR = BENCHMARKS_DIR.parent  # .../documentation
OUTPUT_DIR = BENCHMARKS_DIR / "output"  # .../documentation/benchmarks/output


# -----------------------------------------
# Identify "generator" scripts to skip by default
# -----------------------------------------
GENERATOR_TOKENS = ("gen", "make", "simulate", "simulated", "create", "build")


def is_generator_script(path: Path) -> bool:
    """
    Heuristic to identify scripts that generate data rather than run benchmarks.

    Your repo often uses '01_*' for data generation (e.g. 01_make_simulated_data.py),
    so we treat those as generators.
    """
    name = path.name.lower()
    if name.startswith("01_"):
        return True
    # Additional conservative heuristic:
    if any(tok in name for tok in GENERATOR_TOKENS) and "run" not in name:
        return True
    return False


def iter_python_scripts(include_generators: bool) -> list[Path]:
    """
    Find all .py files under benchmarks/**/scripts/.
    """
    scripts = sorted(BENCHMARKS_DIR.glob("**/scripts/*.py"))
    scripts = [p for p in scripts if p.is_file() and p.name != "__init__.py"]

    if include_generators:
        return scripts

    return [p for p in scripts if not is_generator_script(p)]


# -----------------------------
# Environment stabilization
# -----------------------------
def stable_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """
    Create a *copy* of the current environment with performance-stabilizing settings.

    IMPORTANT:
    - os.environ is not a plain dict; it's a mapping-like object. Accepts Mapping[str, str]
      and then copy into a real dict.
    """
    env: dict[str, str] = dict(base_env)

    # Force single-threaded BLAS/NumExpr to prevent thread oversubscription.
    # Oversubscription can make "more threads" much slower than 1 thread.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # Extra safety for other BLAS stacks (harmless if unused).
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"

    # Prevent OpenMP from changing threads dynamically at runtime.
    env.setdefault("OMP_DYNAMIC", "FALSE")

    # Helps reduce a small source of noise for some workloads.
    env.setdefault("PYTHONHASHSEED", "0")

    return env


# -----------------------------
# CPU affinity (Linux only)
# -----------------------------
def parse_cpu_spec(spec: str) -> list[int]:
    """
    Parse BENCH_CPUS format like:
      "0-7" or "0,2,4,6" or "0-3,8-11"
    """
    spec = spec.strip()
    if not spec:
        return []

    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def set_affinity_linux(pid: int, cpus: Iterable[int]) -> None:
    """
    Pin a process to a set of CPUs (Linux only).

    If the OS or permissions don't allow it, silently do nothing.
    """
    try:
        os.sched_setaffinity(pid, set(cpus))
    except Exception:
        return


# -----------------------------
# Running commands
# -----------------------------
def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_fp,
    cpus: list[int] | None,
    echo_to_console: bool,
) -> tuple[int, float]:
    """
    Run a subprocess command, timing wall time.

    - Always writes stdout/stderr to log file.
    - Optionally echoes output to console (to keep console readable with repeats).
    """
    start = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    if cpus is not None and sys.platform.startswith("linux"):
        set_affinity_linux(proc.pid, cpus)

    assert proc.stdout is not None
    for line in proc.stdout:
        log_fp.write(line)
        if echo_to_console:
            sys.stdout.write(line)

    rc = proc.wait()
    elapsed = time.perf_counter() - start
    return rc, elapsed


@dataclass
class ScriptResult:
    script: Path
    run_times: list[float]  # raw wall times for each run (including dropped runs)
    run_rcs: list[int]  # return code per run
    kept_times: list[float]  # times after dropping first K runs (and rc==0 only)


def format_seconds(x: float) -> str:
    return f"{x:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all Python benchmark scripts with stable timing conditions."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="Total number of runs per script (includes dropped runs). Default: 6",
    )
    parser.add_argument(
        "--drop-first",
        type=int,
        default=1,
        help="Number of initial runs to drop per script (first-run noise). Default: 1",
    )
    parser.add_argument(
        "--include-generators",
        action="store_true",
        help="Include generator scripts (often 01_*). Default: off.",
    )
    parser.add_argument(
        "--echo-all",
        action="store_true",
        help="Echo stdout for every repeat to console. Default: only first run per script.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be >= 1")
    if args.drop_first < 0:
        raise SystemExit("--drop-first must be >= 0")
    if args.drop_first >= args.repeats:
        raise SystemExit("--drop-first must be < --repeats (otherwise nothing is kept)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"python_benchmark_run_{ts}.log"
    csv_path = OUTPUT_DIR / f"python_benchmark_summary_{ts}.csv"

    scripts = iter_python_scripts(include_generators=args.include_generators)
    if not scripts:
        print("No Python benchmark scripts found under benchmarks/**/scripts/.")
        return 1

    # Optional CPU pinning via BENCH_CPUS env var (Linux only).
    cpus: list[int] | None = None
    cpu_spec = os.environ.get("BENCH_CPUS", "").strip()
    if cpu_spec and sys.platform.startswith("linux"):
        parsed = parse_cpu_spec(cpu_spec)
        cpus = parsed if parsed else None

    env = stable_env(os.environ)

    header = (
        f"Python benchmark run: {ts}\n"
        f"Docs dir: {DOCS_DIR}\n"
        f"Benchmarks dir: {BENCHMARKS_DIR}\n"
        f"Python executable: {sys.executable}\n"
        f"Platform: {sys.platform}\n"
        f"CPU affinity: {cpus if cpus is not None else 'default (no pin)'}\n"
        f"Repeats: {args.repeats} (drop first: {args.drop_first})\n"
        f"Include generators: {args.include_generators}\n"
        f"Thread env: "
        f"OMP={env.get('OMP_NUM_THREADS')} "
        f"MKL={env.get('MKL_NUM_THREADS')} "
        f"OPENBLAS={env.get('OPENBLAS_NUM_THREADS')} "
        f"NUMEXPR={env.get('NUMEXPR_NUM_THREADS')} "
        f"VECLIB={env.get('VECLIB_MAXIMUM_THREADS')} "
        f"BLIS={env.get('BLIS_NUM_THREADS')}\n" + "-" * 80 + "\n"
    )

    print(header, end="")

    results: list[ScriptResult] = []

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write(header)

        # Global warmup:
        # - Tries to load NumPy and do a small matrix multiply to “wake up” CPU + libraries.
        # - This is *not* the benchmark. It just reduces weird first-call artifacts.
        warmup_cmd = [
            sys.executable,
            "-c",
            "import numpy as np; a=np.random.rand(2000,200); _=a.T@a; print('warmup ok')",
        ]
        log_fp.write("Warmup...\n")
        print("Warmup...")
        rc, elapsed = run_cmd(
            warmup_cmd,
            cwd=DOCS_DIR,
            env=env,
            log_fp=log_fp,
            cpus=cpus,
            echo_to_console=True,
        )
        log_fp.write(f"Warmup rc={rc} time={format_seconds(elapsed)} s\n\n")
        print(f"Warmup rc={rc} time={format_seconds(elapsed)} s\n")

        for script in scripts:
            rel = script.relative_to(DOCS_DIR)
            banner = f"\n=== {rel} ===\n"
            print(banner, end="")
            log_fp.write(banner)

            run_times: list[float] = []
            run_rcs: list[int] = []

            # Run the script multiple times in fresh processes.
            # We drop the first K runs when computing summary stats.
            for i in range(args.repeats):
                # Keep console output readable:
                # - By default, only echo the first run’s stdout to console (i == 0).
                # - All runs always go into the log file.
                echo = args.echo_all or (i == 0)

                log_fp.write(f"\n--- Run {i+1}/{args.repeats} ---\n")
                if echo:
                    print(f"--- Run {i+1}/{args.repeats} ---")

                cmd = [sys.executable, str(script)]
                rc, t = run_cmd(
                    cmd,
                    cwd=script.parent,  # run from the script directory (matches how you'd run it manually)
                    env=env,
                    log_fp=log_fp,
                    cpus=cpus,
                    echo_to_console=echo,
                )

                run_rcs.append(rc)
                run_times.append(t)

                line = (
                    f"Run {i+1}/{args.repeats}: rc={rc}, time={format_seconds(t)} s\n"
                )
                log_fp.write(line)
                if echo:
                    print(line, end="")

            # Build kept times:
            # - Drop the first args.drop_first runs (common benchmark practice).
            # - Only keep successful runs (rc == 0).
            kept: list[float] = []
            for i in range(args.drop_first, args.repeats):
                if run_rcs[i] == 0:
                    kept.append(run_times[i])

            results.append(
                ScriptResult(
                    script=script,
                    run_times=run_times,
                    run_rcs=run_rcs,
                    kept_times=kept,
                )
            )

            # Quick per-script summary to console/log.
            if kept:
                mn = min(kept)
                md = median(kept)
                mx = max(kept)
                summ = (
                    f"Kept runs: {len(kept)} "
                    f"(dropped {args.drop_first}; failed kept runs excluded)\n"
                    f"min/median/max: {format_seconds(mn)} / {format_seconds(md)} / {format_seconds(mx)} s\n"
                )
            else:
                summ = "Kept runs: 0 (all kept runs failed or were dropped)\n"

            log_fp.write(summ)
            print(summ, end="")

        # Write CSV summary (one row per script).
        # This makes it easy to compare across machines/commits later.
        csv_header = (
            "script,python,platform,cpu_affinity,repeats,drop_first,"
            "kept_n,failed_n,kept_min_s,kept_median_s,kept_max_s,all_times_s,all_rcs\n"
        )
        with open(csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write(csv_header)
            for r in results:
                rel = r.script.relative_to(DOCS_DIR).as_posix()
                failed_n = sum(1 for rc in r.run_rcs[args.drop_first :] if rc != 0)

                if r.kept_times:
                    kept_min = min(r.kept_times)
                    kept_med = median(r.kept_times)
                    kept_max = max(r.kept_times)
                    kept_min_s = format_seconds(kept_min)
                    kept_med_s = format_seconds(kept_med)
                    kept_max_s = format_seconds(kept_max)
                else:
                    kept_min_s = ""
                    kept_med_s = ""
                    kept_max_s = ""

                all_times_s = ";".join(format_seconds(x) for x in r.run_times)
                all_rcs_s = ";".join(str(x) for x in r.run_rcs)

                row = (
                    f"{rel},"
                    f"{sys.executable},"
                    f"{sys.platform},"
                    f"\"{cpus if cpus is not None else 'default'}\","
                    f"{args.repeats},"
                    f"{args.drop_first},"
                    f"{len(r.kept_times)},"
                    f"{failed_n},"
                    f"{kept_min_s},"
                    f"{kept_med_s},"
                    f"{kept_max_s},"
                    f'"{all_times_s}",'
                    f'"{all_rcs_s}"\n'
                )
                fcsv.write(row)

    print(f"\nLog written to: {log_path}")
    print(f"CSV summary written to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
