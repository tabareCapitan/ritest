# ritest

[![PyPI version](https://img.shields.io/pypi/v/ritest-python.svg)](https://pypi.org/project/ritest-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/ritest-python.svg)](https://pypi.org/project/ritest-python/)
[![CI](https://github.com/tabareCapitan/ritest/actions/workflows/ci.yml/badge.svg)](https://github.com/tabareCapitan/ritest/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

`ritest` provides fast randomization inference (RI) tools for linear models and arbitrary statistics. It supports weights as well as stratified and clustered designs. Reports coefficient confidence interval by default.

📑 Documentation: [https://tabarecapitan.com/projects/ritest](https://tabarecapitan.com/projects/ritest).


## Features

* Linear-model RI with efficient computation.
* Generic RI for arbitrary scalar statistics via `stat_fn`.
* Ultra-fast coefficient bounds and bands.
* Stratified, clustered, and stratified-clustered designs.
* Weighted least squares (WLS) support.
* Deterministic seeding, reproducible permutations.

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

## Citation

A software citation entry will be provided once the package reaches its first stable release and the accompanying paper/notes are finalized. For now, cite the GitHub repository:

```
Tabaré Capitán (2025). ritest: Randomization inference in Python.
https://github.com/tabareCapitan/ritest
```

## Disclaimer

Use this software at your own risk. I make no guarantees of correctness or fitness for any purpose. I use it in my own work, but you should review the code to ensure it meets your needs. If you find an issue, please report it.


## Contributing and issues

Bug reports and feature requests can be filed at:

https://github.com/tabareCapitan/ritest/issues

Pull requests should be focused and include tests for new behavior.

## License

MIT License.
See the `LICENSE` file for full terms.
