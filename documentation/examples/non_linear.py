"""
Example: Non-linear models (generic statistic wrappers)

Goal
----
Show how to use `ritest()` with *non-linear* models by supplying a `stat_fn`
that returns a single scalar statistic (here: the coefficient on treatment).

Models covered (n=100, reps=500)
--------------------------------
1) Binary outcome: Logit and Probit (GLM binomial with logit/probit link)
2) Count outcome: Poisson (GLM Poisson)
3) Overdispersed counts: Negative Binomial (GLM NegativeBinomial, alpha fixed)
4) Fractional outcome in [0, 1]: Fractional logit / quasi-binomial (GLM Binomial logit)
5) Censored outcome at 0: Tobit (custom MLE via scipy.optimize)

Outputs
-------
Writes one TXT per model to `output/` (relative to this script).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from ritest import ritest

HERE = Path(__file__).resolve().parent


# ----------------------------
# data generators (n=100)
# ----------------------------
def _make_common_design(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)  # treatment to permute
    x = rng.normal(size=n)  # one baseline covariate (fixed)
    return pd.DataFrame({"z": z, "x": x})


def make_binary_data(n: int = 100, seed: int = 1, tau: float = 0.8) -> pd.DataFrame:
    df = _make_common_design(n, seed)
    rng = np.random.default_rng(seed + 10_000)
    # logistic DGP with a modest effect size to avoid separation at n=100
    p = expit(-0.2 + tau * df["z"].to_numpy() + 0.6 * df["x"].to_numpy())
    y = rng.binomial(1, p, size=n)
    df["y"] = y.astype(int)
    return df


def make_poisson_data(n: int = 100, seed: int = 2, tau: float = 0.35) -> pd.DataFrame:
    df = _make_common_design(n, seed)
    rng = np.random.default_rng(seed + 10_000)
    lam = np.exp(0.3 + tau * df["z"].to_numpy() + 0.4 * df["x"].to_numpy())
    y = rng.poisson(lam, size=n)
    df["y"] = y.astype(int)
    return df


def make_negbin_data(
    n: int = 100, seed: int = 3, tau: float = 0.35, alpha: float = 1.0
) -> pd.DataFrame:
    """
    Negative Binomial DGP with mean mu and variance mu + alpha * mu^2.

    Generate via numpy's negative_binomial parameterization:
      size = 1/alpha
      p = size / (size + mu)
    """
    df = _make_common_design(n, seed)
    rng = np.random.default_rng(seed + 10_000)

    mu = np.exp(0.3 + tau * df["z"].to_numpy() + 0.4 * df["x"].to_numpy())
    size = 1.0 / float(alpha)
    p = size / (size + mu)
    y = rng.negative_binomial(size, p, size=n)
    df["y"] = y.astype(int)
    return df


def make_fractional_data(
    n: int = 100, seed: int = 4, tau: float = 0.6, phi: float = 30.0
) -> pd.DataFrame:
    """
    Fractional outcome in (0,1) generated from Beta(mean=m, precision=phi).
    """
    df = _make_common_design(n, seed)
    rng = np.random.default_rng(seed + 10_000)

    m = expit(-0.1 + tau * df["z"].to_numpy() + 0.5 * df["x"].to_numpy())
    m = np.clip(m, 1e-4, 1 - 1e-4)
    a = m * phi
    b = (1.0 - m) * phi
    y = rng.beta(a, b, size=n)
    df["y"] = y.astype(float)
    return df


def make_tobit_data(
    n: int = 100, seed: int = 5, tau: float = 1.0, sigma: float = 1.0
) -> pd.DataFrame:
    """
    Latent: y* = -0.3 + tau*z + 0.6*x + eps, eps ~ N(0, sigma^2)
    Observed: y = max(0, y*)  (left-censored at 0)
    """
    df = _make_common_design(n, seed)
    rng = np.random.default_rng(seed + 10_000)

    mu = -0.3 + tau * df["z"].to_numpy() + 0.6 * df["x"].to_numpy()
    y_latent = mu + rng.normal(scale=sigma, size=n)
    y = np.maximum(0.0, y_latent)
    df["y"] = y.astype(float)
    return df


# ----------------------------
# model wrappers: return (bhat_z, se_z, model_pvalue_z)
# ----------------------------
def _glm_coef_se_pvalue(df: pd.DataFrame, family) -> Tuple[float, float, float]:
    X = sm.add_constant(df[["z", "x"]], has_constant="add")
    y = df["y"]
    fit = sm.GLM(y, X, family=family).fit(disp=False)
    bhat = float(fit.params["z"])
    se = float(fit.bse["z"])
    p = float(fit.pvalues["z"])
    return bhat, se, p


def fit_logit(df: pd.DataFrame) -> Tuple[float, float, float]:
    return _glm_coef_se_pvalue(df, sm.families.Binomial(link=sm.families.links.Logit()))


def fit_probit(df: pd.DataFrame) -> Tuple[float, float, float]:
    return _glm_coef_se_pvalue(
        df, sm.families.Binomial(link=sm.families.links.Probit())
    )


def fit_poisson(df: pd.DataFrame) -> Tuple[float, float, float]:
    return _glm_coef_se_pvalue(df, sm.families.Poisson())


def fit_negbin_fixed_alpha(
    df: pd.DataFrame, alpha: float = 1.0
) -> Tuple[float, float, float]:
    fam = sm.families.NegativeBinomial(alpha=float(alpha))
    return _glm_coef_se_pvalue(df, fam)


def fit_fractional_logit(df: pd.DataFrame) -> Tuple[float, float, float]:
    # Fractional logit / quasi-binomial via GLM Binomial with logit link.
    return _glm_coef_se_pvalue(df, sm.families.Binomial(link=sm.families.links.Logit()))


# ----------------------------
# Tobit (left-censored at 0): minimal MLE
# ----------------------------
def _tobit_mle_coef_se_pvalue(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Tobit with left-censoring at 0:
      y = max(0, Xβ + ε), ε ~ N(0, σ^2)

    Returns (bhat_z, se_z, pvalue_z).
    """
    y = df["y"].to_numpy(dtype=float)
    X = sm.add_constant(df[["z", "x"]], has_constant="add")

    is_cens = y <= 0.0
    is_obs = ~is_cens

    # theta = [beta0, beta_z, beta_x, log_sigma]
    def nll(theta: np.ndarray) -> float:
        beta = theta[:-1]
        log_sigma = theta[-1]
        sigma = np.exp(log_sigma)

        mu = X @ beta
        z = (y - mu) / sigma

        ll = 0.0
        if np.any(is_obs):
            ll += np.sum(norm.logpdf(z[is_obs]) - log_sigma)
        if np.any(is_cens):
            ll += np.sum(norm.logcdf((0.0 - mu[is_cens]) / sigma))

        return -float(ll)

    # starting values: OLS on uncensored (fallback to zeros)
    beta0 = np.zeros(X.shape[1], dtype=float)
    log_sigma0 = np.log(np.std(y[is_obs]) + 1e-3) if np.any(is_obs) else 0.0
    if np.any(is_obs):
        try:
            beta0 = np.linalg.lstsq(X[is_obs], y[is_obs], rcond=None)[0]
        except Exception:
            beta0 = np.zeros(X.shape[1], dtype=float)

    theta0 = np.concatenate([beta0, np.array([log_sigma0])])

    opt = minimize(
        nll,
        theta0,
        method="BFGS",
        options={"maxiter": 120, "gtol": 1e-6},
    )

    theta_hat = opt.x
    beta_hat = theta_hat[:-1]

    cov = None
    if hasattr(opt, "hess_inv") and opt.hess_inv is not None:
        try:
            cov = np.asarray(opt.hess_inv, dtype=float)
        except Exception:
            cov = None

    bhat_z = float(beta_hat[1])  # const, z, x
    if cov is None or cov.shape[0] < 2:
        se_z = float("nan")
        p_z = float("nan")
    else:
        se_z = float(np.sqrt(max(cov[1, 1], 0.0)))
        if se_z == 0.0 or not np.isfinite(se_z):
            p_z = float("nan")
        else:
            w = bhat_z / se_z
            p_z = float(2.0 * (1.0 - norm.cdf(abs(w))))

    return bhat_z, se_z, p_z


def fit_tobit(df: pd.DataFrame) -> Tuple[float, float, float]:
    return _tobit_mle_coef_se_pvalue(df)


# ----------------------------
# run helpers
# ----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    df: pd.DataFrame
    fit_fn: Callable[[pd.DataFrame], Tuple[float, float, float]]


def _write_report(
    *, spec: ModelSpec, res, out_path: Path, runtime_seconds: float
) -> None:
    bhat, se, p_model = spec.fit_fn(spec.df)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Example: {spec.name}\n")
        f.write("=" * (9 + len(spec.name)) + "\n\n")

        f.write("Setup\n")
        f.write("-----\n")
        f.write(f"n = {len(spec.df)}\n")
        f.write(f"reps = {res.reps}\n")
        f.write(f"seed = {res.settings.get('seed', 'unknown')}\n")
        f.write("permuted variable: z\n")
        f.write(f"runtime (seconds): {runtime_seconds:.3f}\n\n")

        f.write("Model and statistic\n")
        f.write("-------------------\n")
        f.write(spec.description.strip() + "\n\n")
        f.write("Statistic used in RI:\n")
        f.write("  coefficient on z from the model fit on (y ~ z + x)\n\n")

        f.write("Observed (model-based) estimate on the original data\n")
        f.write("----------------------------------------------------\n")
        f.write(f"bhat_z: {bhat:.6f}\n")
        f.write(f"se_z:   {se:.6f}\n")
        f.write(f"p_z:    {p_model:.6f}\n\n")

        f.write("Randomization inference result (ritest)\n")
        f.write("--------------------------------------\n")
        f.write(res.summary(print_out=False))
        f.write("\n")


def _run_ritest_for_spec(spec: ModelSpec, *, seed: int) -> Tuple[object, float]:
    """
    Generic-statistic path:
      - stat_fn(df_perm) returns the scalar coefficient on z.
      - permute_var="z" tells ritest what to shuffle.
    """

    def stat_fn(df_perm: pd.DataFrame) -> float:
        bhat, _, _ = spec.fit_fn(df_perm)
        return float(bhat)

    t0 = perf_counter()
    res = ritest(
        df=spec.df,
        permute_var="z",
        stat_fn=stat_fn,
        alternative="two-sided",
        reps=500,
        seed=seed,
        ci_method="clopper-pearson",
        alpha=0.05,
        ci_mode="none",  # focus on p-value + observed coefficient here
    )
    t1 = perf_counter()
    return res, (t1 - t0)


def main() -> None:
    out_dir = HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[ModelSpec] = []

    # 1) Binary: logit/probit (same data)
    df_bin = make_binary_data(n=100, seed=1, tau=0.8)
    specs.append(
        ModelSpec(
            name="Binary outcome — Logit (GLM binomial, logit link)",
            description="""
Use case: treatment -> binary outcome (adoption, survival, compliance).
Fit logistic regression by GLM; RI statistic is the treatment coefficient.
""",
            df=df_bin,
            fit_fn=fit_logit,
        )
    )
    specs.append(
        ModelSpec(
            name="Binary outcome — Probit (GLM binomial, probit link)",
            description="""
Same data as the logit example, but the model uses a probit link.
RI statistic is still the treatment coefficient.
""",
            df=df_bin,
            fit_fn=fit_probit,
        )
    )

    # 2) Count: Poisson
    df_pois = make_poisson_data(n=100, seed=2, tau=0.35)
    specs.append(
        ModelSpec(
            name="Count outcome — Poisson (GLM Poisson)",
            description="""
Use case: treatment -> event counts (visits, clicks, defects).
Fit Poisson GLM; RI statistic is the treatment coefficient.
""",
            df=df_pois,
            fit_fn=fit_poisson,
        )
    )

    # 3) Overdispersed counts: Negative Binomial (alpha fixed)
    alpha_nb = 1.0
    df_nb = make_negbin_data(n=100, seed=3, tau=0.35, alpha=alpha_nb)
    specs.append(
        ModelSpec(
            name="Count outcome — Negative Binomial (GLM NB, alpha fixed)",
            description=f"""
Use case: overdispersed counts (variance >> mean).
Fit Negative Binomial GLM with alpha fixed at {alpha_nb:.1f};
RI statistic is the treatment coefficient.
""",
            df=df_nb,
            fit_fn=lambda d: fit_negbin_fixed_alpha(d, alpha=alpha_nb),
        )
    )

    # 4) Fractional outcome: fractional logit / quasi-binomial
    df_frac = make_fractional_data(n=100, seed=4, tau=0.6, phi=30.0)
    specs.append(
        ModelSpec(
            name="Fractional outcome — Fractional logit (quasi-binomial GLM)",
            description="""
Use case: proportions in [0,1] not coming from binomial counts (e.g., budget shares).
Fit binomial GLM with logit link (quasi-likelihood); RI statistic is the treatment coefficient.
""",
            df=df_frac,
            fit_fn=fit_fractional_logit,
        )
    )

    # 5) Tobit: censored at 0
    df_tob = make_tobit_data(n=100, seed=5, tau=1.0, sigma=1.0)
    specs.append(
        ModelSpec(
            name="Limited / censored outcome — Tobit (MLE, left-censored at 0)",
            description="""
Use case: outcomes censored at 0 (time, expenditures, effort).
Fit Tobit by custom maximum likelihood (Normal errors);
RI statistic is the treatment coefficient.
""",
            df=df_tob,
            fit_fn=fit_tobit,
        )
    )

    for i, spec in enumerate(specs, start=1):
        res, rt = _run_ritest_for_spec(spec, seed=10_000 + i)
        out_path = out_dir / f"example_nonlinear_{i:02d}.txt"
        _write_report(spec=spec, res=res, out_path=out_path, runtime_seconds=rt)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
