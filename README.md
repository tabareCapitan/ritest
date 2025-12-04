# ritest

Fast, deterministic **randomization inference** (permutation tests + test-inversion CIs) for Python.

- Exact (or Monte-Carlo) **permutation tests** for arbitrary statistics
- **Test-inversion** confidence **bounds**/**bands**
- High-performance **OLS/WLS** engine with analytic CI shifting (`FastOLS`)
- Correct **strata**/**cluster** permutation schemes with strong guards
- Deterministic float64 numerics; optional **Numba** acceleration (NumPy fallback)

> **Status:** actively developed; core engines and CI for permutation *p*-values are in place. This README is a working draft and will evolve as we finalize public APIs and docs.

---

## Why randomization inference?

Randomization inference (RI) evaluates statistical significance by comparing the observed statistic to its **permutation distribution** under a design-consistent reshuffling scheme (e.g., within strata, by clusters). It is exact under the null for finite samples and remains valid with small *n*, complex dependence, or heteroskedasticity—when asymptotics are fragile.

---

## Features at a glance

- **Engine**
  - `FastOLS`: efficient OLS/WLS with:
    - dot-product trick for rapid coefficient updates under permutations
    - robust VCOV (HC1) and cluster-robust (CRV1)
    - Cholesky solves; stable float64 determinism
    - supports **contrast vectors** (`c_vector`) for analytic CI shifting
  - `shuffle`: design-correct permutations
    - **strata** (no contiguity assumptions; explicit index scatter)
    - **cluster** (vectorized; safe label checks; cluster-constancy asserted)
    - **cluster×strata** (vectorized per stratum)
    - uniqueness diagnostics (cheap, optional)

- **CI modules**
  - `ci/pvalue_ci.py`: Clopper–Pearson (“cp”) and Wald+CC (“normal”) CIs for permutation *p*-values
  - `ci/coef_ci.py`: test-inversion CIs for coefficients (analytic/generic)

- **Orchestration & results**
  - `run.py`: wires validation → shuffling → stat/engine → CI/bands
  - `results.py`: `RitestResult` with `.summary()` and `.plot()` helpers
  - `config.py`: central knobs; `ritest_set(...)` for overrides

- **Validation**
  - `validation.py`: converts DataFrames to exact dtypes, validates **user-facing knobs** (`alternative`, `alpha`, `ci_method`, …), factorizes clusters/strata, ensures positive weights, etc.

---

## Installation (dev)

```bash
# inside your virtualenv
pip install -e .            # editable install

# optional (for tests and dev tooling)
pip install -e ".[dev]"     # add pytest/ruff/black/statsmodels, etc.

# optional acceleration (install if you want numba JIT paths)
pip install numba
```

> Runtime dependency: `scipy` (for Beta/Normal quantiles in CI).
> `statsmodels` is **test-only** (used by tests to cross-check Clopper–Pearson).

---

## Quick start (draft API)

```python
import numpy as np
import pandas as pd
from ritest.run import ritest   # orchestrator

# DataFrame with outcome y, treatment T, and controls X1, X2
df = pd.DataFrame({
    "y":  np.random.randn(200),
    "T":  np.random.randint(0, 2, size=200),
    "X1": np.random.randn(200),
    "X2": np.random.randn(200),
    "strata": np.repeat(np.arange(10), 20),
})

# Randomization inference on the treatment coefficient in a linear model
res = ritest(
    df=df,
    formula="y ~ T + X1 + X2",      # patsy formula
    permute_var="T",                 # what we shuffle under the null
    alternative="two-sided",         # {"two-sided","left","right"}
    B=5000,                          # number of permutations
    strata="strata",                 # (optional) respect design strata
    alpha=0.05,                      # CI level for p-value & bands
    ci=True,                         # request CI bounds/bands
    ci_method="cp",                  # {"cp","normal"} for p-value CI
    ci_mode="bounds",                # or "bands"
)

print(res.summary())
res.plot()  # optional visualization
```

> The high-level API above is stable in spirit but still being refined; exact argument names may change as we finalize `run.py`.

---

## Statistical notes: permutation *p*-value confidence intervals

We treat the observed exceedance count `c` as a draw from `Binomial(n = reps, p_true)`,
where `p_true` is the (unknown) true permutation *p*-value under the resampling scheme.
`ritest.ci.pvalue_ci.pvalue_ci(c, reps, alpha, method)` returns a `(1 - alpha)` CI for `p_true`:

- `method="cp"` — **Clopper–Pearson** (exact equal-tailed binomial CI) using beta quantiles.
  Edge behavior: if `c = 0`, lower bound is 0; if `c = reps`, upper bound is 1.

- `method="normal"` — **Wald (normal approximation)** with a continuity correction of **± 0.5 / reps**,
  then clipped to the unit interval.

Implementation notes:
- The CP interval matches `statsmodels.stats.proportion.proportion_confint(c, reps, method="beta")`.
- The Wald variant here includes a continuity correction; `statsmodels`’ `"normal"` method does **not**
  include this correction, so values will differ by ≈ `0.5 / reps`.

See `tests/README.md` for references and verification details.

---

## Engines (implementation highlights)

### Fast OLS / WLS (`engine/fast_ols.py`)
- **Dot-product trick**: re-use `X'X` and `X'y` with a **contrast vector** `c_vector` for fast coefficient updates under permutations.
- Weighted path uses √w in the right places; **precompute** a weighted `c_vector` once (callers stay in original space).
- Robust VCOV: **HC1**; cluster-robust **CRV1**.
- Linear algebra via **Cholesky** (stable, deterministic); strict float64.

### Permutation generator (`engine/shuffle.py`)
- **Strata**: no contiguity assumptions; explicit index mapping.
- **Cluster**: vectorized with `return_inverse` fallback; asserts **cluster-constancy** of permuted variable.
- **Cluster×strata**: vectorized per stratum.
- **Uniqueness diagnostics** (optional) to detect too-few unique draws in clustered modes.

---

## Validation & assumptions

- **Inputs** are validated centrally in `validation.py`:
  - `alternative ∈ {"two-sided","left","right"}`
  - `alpha ∈ (0,1)`; `ci_method ∈ {"cp","normal"}`
  - `permute_var` exists, numeric, no NA
  - `cluster`/`strata` are factorized to dense int codes, no NA; clusters ≥ 2 groups
  - weights (if any) strictly positive, no NA
- **Runtime invariants** (e.g., permutation counts `c`, total `reps`) are validated inside the CI/math helpers.

---

## Reproducibility

- Deterministic float64 numerics (no hidden randomness).
- All permutation draws are seeded (user-provided or fixed), and design constraints are enforced.
- Optional **Numba** paths are functionally identical to NumPy baselines.

---

## Performance tips

- Use **strata/cluster** constraints only when design demands them—unnecessary constraints reduce unique permutations.
- For heavy runs, **pre-generate** a permutation matrix with `generate_permuted_matrix(...)` and reuse it.
- Enable **Numba** if available for hot loops; ensure consistent seeds and avoid data copies in tight loops.

---

## Testing

```bash
# core tests (prints enabled)
pytest -q -s

# coverage
pytest -q --maxfail=1 --disable-warnings --cov=ritest --cov-report=term-missing
```

Cross-checks:
- CP intervals are validated against **statsmodels** (`method="beta"`).
- Wald+CC intervals are validated against the **explicit formula** used in `ritest`.

Dev extras (`pyproject.toml`):
```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "black", "statsmodels"]
```

---

## Roadmap (draft)

- Finalize public API for `ritest(...)` and results object
- Complete `coef_ci.py` bands & documentation
- End-to-end examples (linear model; arbitrary `stat_fn`)
- Packaging & release (wheels, conda-forge if demand)
- Sphinx/Quarto documentation site with theory + how-tos
- Benchmarks vs. baseline implementations

---

## Contributing

- Style: **ruff** + **black**
- Tests: **pytest** with readable diagnostics; prefer small data w/ seeds
- Avoid silent assumptions; add guards with clear error messages
- Keep NumPy and Numba paths **behavior-identical**

---

## License

TBD (likely MIT).

---

## Citation

TBD (paper/preprint in progress). If you use `ritest` in a paper, please cite the repo and the specific version/tag.
