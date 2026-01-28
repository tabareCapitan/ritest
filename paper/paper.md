---
title: "ritest: randomization inference in Python"
tags:
  - Python
  - statistics
  - causal inference
  - econometrics
  - experimental design
  - randomization inference
authors:
  - name: "Tabaré Capitán"
    affiliation: 1
affiliations:
  - name: "Independent researcher"
    index: 1
date: "22 December 2025"
bibliography: paper.bib
---


# Summary

Randomization inference (RI) provides design-based inference for experiments by comparing an observed test statistic to its distribution over re-randomizations of the treatment assignment. Because the reference distribution is defined by the assignment mechanism, RI can be especially attractive when sample sizes are modest or when researchers want inference that depends primarily on the randomization protocol rather than large-sample approximations [@fisher1935; @imbensrubin2015].

`ritest` is a Python package implementing RI in a workflow similar to Stata’s `ritest` command [@hess2017], but designed around Python’s numerical stack and extended to support computationally feasible test inversion. It supports (i) a fast linear-model path for regression coefficients, and (ii) a generic path that accepts arbitrary user-defined statistics. For linear models, `ritest` additionally implements test inversion to obtain coefficient confidence interval (CI) bounds and a full $p(\beta_0)$ profile (“CI band”) over candidate null values, designed to be practical even with large permutation counts.


# Features

`ritest` exposes one user-facing function, `ritest()`, which takes a dataframe, a binary assignment variable, and either:

1. A linear-model specification via a formula and a focal coefficient/statistic (fast path), or
2. A user-supplied statistic function `stat_fn(df) -> float` (generic path).

## Assignment mechanisms

Permutation generation supports four common assignment modes:

- **Plain**: unrestricted permutation of the assignment vector.
- **Stratified**: permutation within each stratum to preserve treated counts per stratum.
- **Cluster**: permutation of cluster-level assignments, broadcast to units within clusters.
- **Cluster-within-strata**: cluster permutation applied separately within each stratum.

These modes are designed to match typical experimental designs and to keep the RI reference distribution aligned with the original randomization protocol [@gerbergreen2012; @imbensrubin2015].

## Fast linear-model path

For linear models, `ritest` uses a specialized OLS/WLS solver (`FastOLS`) for each permutation. Because the treatment column changes, the regression is still solved once per permutation, but permutation fits skip variance–covariance computation and avoid general-purpose model overhead. For a fixed design matrix, the focal coefficient can be written as a dot product
$\hat{\beta} = c^\top y$, where $c$ is determined by the design matrix (and, if applicable, weights). `FastOLS` computes $c$ using Cholesky-based linear algebra.

## Monte Carlo $p$-values and $p$-value confidence intervals

With $R$ random permutations, the RI $p$-value is estimated by an exceedance proportion. `ritest` reports a CI for this Monte Carlo estimate by treating the exceedance count as binomial (Clopper–Pearson and normal-approximation options).

## Coefficient confidence intervals by test inversion

For linear models, `ritest` can compute coefficient CI **bounds** and a full $p(\beta_0)$ **profile** over a grid of candidate nulls. Test inversion constructs a CI as the set of null values $\beta_0$ that would not be rejected by the RI test at level $\alpha$.

For candidate $\beta_0$, `ritest` evaluates $p(\beta_0)$ using shifted coefficients $\beta_{\mathrm{obs}}-\beta_0 K_{\mathrm{obs}}$ and $\beta_r-\beta_0 K_r$ (with $K=c^\top T_{\text{metric}}$). This reuses $(\beta_r, K_r)$ from the permutation fits, avoiding a refit for each $\beta_0$.

A naive implementation would refit the model for each permutation *and* for each candidate null value, which is often impractical. `ritest` implements a fast approach that reuses precomputed invariants from the observed fit and the permuted fits, making it feasible to report coefficient CIs as a default for linear models.


# Examples and typical applications

`ritest` is designed to support common applications of RI:

- **Regression adjustment in randomized experiments:** RI on the treatment coefficient in an OLS model with covariates.
- **Stratified field experiments:** constrained re-randomization within pre-treatment strata.
- **Cluster randomized trials:** permutation at the cluster level, including cluster-within-strata designs.
- **Custom estimands:** any scalar statistic defined by the user, including distributional or robust summaries (generic path).

These patterns cover settings frequently encountered in applied economics and political science field experiments, psychology and education trials, and agricultural or ecological intervention studies.


# Software design

Randomization inference is computationally dominated by repeated estimation under permuted treatment assignments. The central design problem in `ritest` is therefore how to make this estimation fast enough to be practical for applied work, while preserving the assignment mechanism that defines the RI reference distribution.

`ritest` addresses this by deliberately specializing in linear models, which admit a common linear-algebraic solution. By restricting the optimized path to this class, the package can reuse design-matrix invariants and avoid general-purpose model-fitting overhead. This makes large permutation counts feasible and enables routine computation of coefficient confidence intervals and full $p(\beta_0)$ profiles by test inversion. The trade-off is that non-linear models are not supported on this fast path.

This restriction reflects common applied practice. In randomized experiments, linear regressions are widely used to estimate average treatment effects and to report design-based inference for regression coefficients. Supporting non-linear models efficiently would require a different architecture and substantially greater complexity, with limited benefit for this core use case.

To retain flexibility, `ritest` also provides a generic path that accepts any user-defined statistic, trading performance for generality. This two-tier design keeps the implementation predictable and performant.


# Statement of need

RI is used across fields with randomized or quasi-randomized designs, including economics, political science, agricultural science, sociology, psychology, education, public health, and biosciences. Typical settings include A/B tests, laboratory experiments, greenhouse experiments, field experiments, and cluster randomized trials. In these applicatons, researchers often face three recurring challenges.

First, assignment mechanisms are frequently non-standard. Stratified randomization, cluster assignment, and combinations of the two are common in applied work, and valid inference requires the reference distribution to respect these design constraints. Second, sample sizes are often small or moderate, making large-sample approximations fragile or difficult to justify. Third, researchers increasingly emphasize transparent reporting of uncertainty, with interest in design-based $p$-values that depend directly on the randomization protocol rather than on model-based assumptions.

Python has a strong ecosystem for estimation and model-based inference (e.g., linear models and generalized linear models in `statsmodels` [@seabold2010statsmodels] and numerical foundations in `NumPy` and `SciPy` [@harris2020numpy; @virtanen2020scipy]). However, a complete randomization inference workflow that mirrors common applied practice requires additional components: constrained re-randomization that matches the study design, explicit reporting of Monte Carlo uncertainty in $p$-values, and (for regression coefficients) a way to obtain confidence intervals by inverting the RI test.

`ritest` is designed to fill this gap. It provides a unified implementation of design-based inference for Python users that supports common experimental assignment mechanisms and integrates directly with regression-based analysis workflows. By making randomization inference computationally feasible for routine use, including for coefficient confidence intervals obtained by test inversion, `ritest` enables applied researchers to report design-based uncertainty alongside familiar regression estimates within a single, reproducible analysis pipeline.


# State of the field

Randomization inference has a long history in statistics and is widely used in applied experimental research. In practice, two mature software implementations define the current standard workflow: Stata's `ritest` command [@hess2017] and the corresponding implementations available in R (e.g., `ritest` [@ritestR]). These tools support inference that is explicitly tied to the experimental assignment mechanism and are commonly used in economics and other related fields.

While Python offers a rich ecosystem for estimation and resampling-based methods, an equivalent implementation of general randomization inference has not been available. Existing Python tools focus primarily on permutation tests in the narrow sense—typically unrestricted shuffling of labels—rather than on randomization inference defined by the original experimental design. In particular, they do not natively support constrained re-randomization schemes such as stratified assignment, cluster randomization, or cluster-within-strata designs, nor are they structured around regression-based estimands that are central to applied work.

Moreover, the architecture of existing permutation-testing tools does not readily accommodate extensions that are important for randomization inference in practice, such as reporting Monte Carlo uncertainty for $p$-values or obtaining confidence intervals for regression coefficients by inverting the RI test. These features require repeated evaluation of a common estimand under many assignment draws and benefit from reusing invariants of the design matrix—an approach that differs from the design of generic resampling utilities.

Stata and R implementations already embody these ideas, but contributing the optimized approach used in `ritest` upstream was not feasible. The fast linear-model path relies on tight integration with Python’s numerical stack and on architectural choices tailored to Python’s array-based computation model. As a result, `ritest `fills a distinct gap in the Python ecosystem by providing a design-based inference tool that aligns with established RI workflows while making computationally demanding procedures, such as coefficient test inversion, practical for routine use.


# Research impact statement

`ritest` provides a design-based inference workflow for Python users working with randomized experiments, a capability that was previously missing from the Python ecosystem as an integrated and computationally practical tool. While the package was first released in December 2025 and has not yet accumulated citations, its potential research impact is well defined and grounded in existing applied practice.

Randomization inference is widely used in applied economics and related fields, where Stata’s implementation of ritest [@hess2017] alone has accumulated several hundred citations. This sustained use reflects both the methodological relevance of RI and the demand for software that closely mirrors experimental design. At the same time, Python has become an increasingly common environment for data analysis and experimentation, particularly in interdisciplinary research and in settings where reproducible, end-to-end pipelines are valued.

By bringing an established RI workflow to Python, `ritest` lowers the barrier to applying design-based inference in these settings. Beyond basic permutation tests, the package makes it computationally feasible to report richer uncertainty measures, most notably confidence intervals for regression coefficients and full $p(\beta_0)$ profiles obtained by test inversion. Performance benchmarks included with the software demonstrate substantial speed gains relative to existing implementations, making these procedures practical at permutation counts that would otherwise be prohibitive.

The software is released under an open-source license and is accompanied by extensive [documentation](https://tabarecapitan.com/projects/ritest/), worked examples, reproducible benchmarks, and technical notes describing both the statistical and computational aspects of the implementation. Together, these features are intended to support adoption, scrutiny, and extension by the research community, and to encourage wider use of design-based inference within Python-based experimental research.


# AI usage disclosure

Generative AI tools were used during the development of this project. ChatGPT, by Open AI, was used to assist with software development tasks, including code drafting, refactoring, and test scaffolding. Limited AI assistance was also used to generate example code and benchmark scaffolding, both used for the documentation. The manuscript text and documentation were written by the author, with minor AI-assisted copy-editing.

The specific model versions used evolved over the development period, and exact version identifiers were not systematically recorded. All AI-assisted outputs were reviewed, edited, and validated by the author, who takes full responsibility for the accuracy, originality, licensing, and ethical compliance of the software and the paper. All architectural decisions, statistical methodology, and mathematical derivations were designed by the author. No AI tools were used for interactions with editors or reviewers.


# Availability

`ritest` is open source, distributed on PyPI as `ritest-python` [@ritestPyPI], and developed on GitHub [@ritestGitHub]. Documentation is hosted on the project website [@ritestDocs].


# Acknowledgements

The author received no specific funding for this work and has no additional acknowledgements to declare.
