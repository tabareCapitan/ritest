# Tests: `pvalue_ci`

This folder contains comprehensive tests for `ritest.ci.pvalue_ci.pvalue_ci`.

## References

- **Clopper–Pearson (exact)**: validated against
  `statsmodels.stats.proportion.proportion_confint(c, n, alpha, method="beta")`.
- **Wald (normal approximation)**: validated against the **explicit formula** used in `ritest`
  (with continuity correction ± `0.5 / n`). Note that `statsmodels`’ `"normal"` method does **not**
  apply this correction and will therefore differ.

## Running

```bash
pytest -q -s tests/test_pvalue_ci.py
pytest -q --maxfail=1 --disable-warnings --cov=ritest --cov-report=term-missing
