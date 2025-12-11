# ritest

[![PyPI version](https://img.shields.io/pypi/v/ritest-python.svg)](https://pypi.org/project/ritest-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/ritest-python.svg)](https://pypi.org/project/ritest-python/)
[![CI](https://github.com/tabareCapitan/ritest/actions/workflows/ci.yml/badge.svg)](https://github.com/tabareCapitan/ritest/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`ritest` provides randomization inference in Python. It supports simple treatment assignment as well as more complex designs (stratified, cluster,  cluster-within-strata). The package includes a fast OLS-based path for linear models and a generic path for any statistic.

## Installation

From PyPI:

```bash
pip install ritest-python
```

Optional extras:

```bash
pip install ritest-python[plot]      # plotting support
pip install ritest-python[numba]     # Numba acceleration
```

## Quickstart

### Linear model (formula interface)

```python
import pandas as pd
from ritest import ritest

df = pd.DataFrame({
    "y": [1, 2, 3, 4],
    "treat": [0, 1, 0, 1],
    "x": [5, 6, 7, 8],
})

res = ritest(
    df=df,
    permute_var="treat",
    formula="y ~ treat + x",
    stat="treat",
    reps=1000,
)

print(res.summary())
```

### Custom statistic (`stat_fn`)

```python
def my_stat(df):
    return df["y"].corr(df["treat"])

res = ritest(
    df=df,
    permute_var="treat",
    stat_fn=my_stat,
    reps=1000,
)

print(res.pvalue)
```

## Features

- p-values for linear models and arbitrary statistics.
- Fast coefficient-shift inversion for confidence interval computation (linear models).
- Generic confidence intervals via user-supplied statistic functions.
- Stratified, clustered, and cluster-within-strata designs.
- Vectorized p-value confidence intervals.
- Clean result object (`RitestResult`) with summary and optional plotting.
- Optional Numba acceleration for permutation paths.

## Documentation

This README covers the essentials.
Full documentation, extended examples, and conceptual notes are available at:

**https://tabarecapitan.com/projects/ritest**

## Relationship to existing RI tools

The API and workflow are influenced by existing RI implementations in Stata (`ritest`) and R packages that follow Fisher–Neyman randomization logic. This package provides a native Python alternative with a unified interface for linear and non-linear statistics, explicit support for stratified and clustered designs, and vectorized CI computations.

Benchmarks will be reported separately. A placeholder observation: RI routines in `pyfixest` are likely faster for high-dimensional fixed-effects models due to fixest’s optimized FE machinery. `ritest` focuses on general-purpose RI, transparent design handling, and flexible user-defined statistics rather than FE-heavy workflows.

## Citation

A software citation entry will be provided once the package reaches its first stable release and the accompanying paper/notes are finalized. For now, cite the GitHub repository:

```
Tabaré Capitán (2025). ritest: Randomization inference in Python.
https://github.com/tabareCapitan/ritest
```

## Contributing and issues

Bug reports and feature requests can be filed at:

https://github.com/tabareCapitan/ritest/issues

Pull requests should be focused and include tests for new behavior.

## License

MIT License.
See the `LICENSE` file for full terms.
