"""
Create a realistic-looking toy experimental dataset.

Design goals
- 10,000 observations
- Randomized treatment (50/50), no strata, no clusters
- 20 covariates with realistic distributions and correlations
- Continuous outcome with a *very small* heterogeneous treatment effect
- Save CSV to: data/simulated_data.csv

Run
  python scripts/01_make_simulated_data.py

Optional
  python scripts/01_make_simulated_data.py --n 20000 --out data/simulated_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NOISE_SD = 5.0  # SE=NOISE_SD/sqrt(n); set to 5.0 for n=10,000 to get SE≈0.05


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_dataset(n: int, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic dataset with correlated covariates and a tiny heterogeneous treatment effect.

    Notes on realism:
    - Covariates are driven by a few latent factors (socioeconomic status, engagement, risk),
      which induces correlations typical in applied data.
    - Values are clipped to plausible ranges and cast to appropriate dtypes.
    """
    rng = np.random.default_rng(seed)

    # Latent factors (correlated drivers of observed covariates)
    socio = rng.normal(0.0, 1.0, size=n)
    engage = 0.6 * socio + rng.normal(0.0, 1.0, size=n)
    risk = -0.3 * socio + 0.4 * engage + rng.normal(0.0, 1.0, size=n)

    # Treatment: randomized, 50/50
    treat = rng.binomial(1, 0.5, size=n).astype(np.int8)

    # Covariates (20 total)
    # 1) Age (years): 18–70
    age = np.clip(rng.normal(38.0, 12.0, size=n) + 1.5 * risk, 18.0, 70.0).astype(
        np.float32
    )

    # 2) Female indicator (roughly balanced, slightly varying)
    female = rng.binomial(1, _sigmoid(0.1 - 0.15 * risk)).astype(np.int8)

    # 3) Education years: 9–20
    education_years = np.clip(
        rng.normal(13.5, 2.2, size=n) + 0.8 * socio, 9.0, 20.0
    ).astype(np.float32)

    # 4) Log income (natural log of monthly income proxy)
    log_income = (rng.normal(10.2, 0.35, size=n) + 0.35 * socio - 0.05 * risk).astype(
        np.float32
    )

    # 5) Household size: 1–7
    household_size = np.clip(rng.poisson(1.2, size=n) + 1, 1, 7).astype(np.int16)

    # 6) Urban indicator
    urban = rng.binomial(1, _sigmoid(0.4 + 0.35 * socio)).astype(np.int8)

    # 7) Tenure (months): 0–120
    tenure_months = np.clip(
        rng.exponential(scale=18.0, size=n) + 6.0 * engage, 0.0, 120.0
    ).astype(np.float32)

    # 8) Baseline spend (monthly): lognormal, depends on income + engagement
    baseline_spend = np.exp(
        log_income - 9.6 + 0.25 * engage + rng.normal(0.0, 0.35, size=n)
    )
    baseline_spend = np.clip(baseline_spend, 0.0, 6000.0).astype(np.float32)

    # 9) Purchases in last 12 months: Poisson with log-link (more spend/engage -> more purchases)
    lam_purch = np.exp(-0.2 + 0.25 * engage + 0.0003 * baseline_spend)
    purchases_12m = rng.poisson(lam=np.clip(lam_purch, 0.05, 30.0)).astype(np.int16)

    # 10) Returns in last 12 months: more purchases + more risk -> more returns
    lam_ret = np.exp(-1.0 + 0.08 * purchases_12m + 0.25 * risk)
    returns_12m = rng.poisson(lam=np.clip(lam_ret, 0.01, 10.0)).astype(np.int16)

    # 11) Support tickets (6 months): risk-driven
    lam_tix = np.exp(-1.2 + 0.45 * risk + 0.05 * purchases_12m)
    support_tickets_6m = rng.poisson(lam=np.clip(lam_tix, 0.01, 12.0)).astype(np.int16)

    # 12) App sessions (30 days): engagement-driven
    lam_sess = np.exp(2.2 + 0.55 * engage - 0.10 * risk)
    app_sessions_30d = rng.poisson(lam=np.clip(lam_sess, 0.5, 300.0)).astype(np.int16)

    # 13) Days since last purchase: inverse of engagement/purchases
    days_since_last_purchase = rng.exponential(scale=30.0, size=n) / (
        1.0 + 0.15 * purchases_12m + 0.25 * engage.clip(-2, 2)
    )
    days_since_last_purchase = np.clip(days_since_last_purchase, 0.0, 365.0).astype(
        np.float32
    )

    # 14) Email opt-in
    email_opt_in = rng.binomial(1, _sigmoid(0.2 + 0.35 * engage - 0.10 * risk)).astype(
        np.int8
    )

    # 15) Promo exposures (30 days)
    lam_promo = np.exp(0.6 + 0.25 * engage + 0.15 * urban)
    promo_exposure_30d = rng.poisson(lam=np.clip(lam_promo, 0.2, 40.0)).astype(np.int16)

    # 16) Prior churn indicator
    prior_churn = rng.binomial(1, _sigmoid(-1.0 + 0.55 * risk - 0.20 * engage)).astype(
        np.int8
    )

    # 17) Credit score: 300–850
    credit_score = np.clip(
        670.0 + 45.0 * socio - 35.0 * risk + rng.normal(0.0, 35.0, size=n), 300.0, 850.0
    ).astype(np.float32)

    # 18) Satisfaction score: 1–5 (treated as numeric)
    sat_latent = (
        3.4 + 0.35 * engage - 0.30 * risk + 0.10 * socio + rng.normal(0.0, 0.7, size=n)
    )
    satisfaction_score = np.clip(sat_latent, 1.0, 5.0).astype(np.float32)

    # 19–22) Region dummies (4 dummies for 5 regions)
    # Create a plausible region assignment correlated with socio/urban.
    # region ∈ {0,1,2,3,4} and we include 4 dummies (omit region 0).
    region_score = 0.6 * socio + 0.3 * urban + rng.normal(0.0, 1.0, size=n)
    # Convert to 5 bins with roughly even mass
    q = np.quantile(region_score, [0.2, 0.4, 0.6, 0.8])
    region = np.digitize(region_score, bins=q).astype(np.int8)  # 0..4

    region_1 = (region == 1).astype(np.int8)
    region_2 = (region == 2).astype(np.int8)
    region_3 = (region == 3).astype(np.int8)
    region_4 = (region == 4).astype(np.int8)

    # Very small heterogeneous treatment effect
    # Range is tiny by construction: roughly [0.005, 0.035] on the outcome scale.
    spend_z = (baseline_spend - baseline_spend.mean()) / (baseline_spend.std() + 1e-12)
    sat_z = (satisfaction_score - satisfaction_score.mean()) / (
        satisfaction_score.std() + 1e-12
    )
    hetero_index = np.clip(0.5 * spend_z + 0.5 * sat_z, -2.0, 2.0)
    tau = (0.02 + 0.0075 * hetero_index).astype(np.float32)

    # Outcome: continuous "value" with realistic signal + noise
    # Keep coefficients moderate; noise dominates so the treatment effect is hard to detect.
    eps = rng.normal(0.0, NOISE_SD, size=n).astype(np.float32)

    y = (
        45.0
        + 0.20 * age
        + 1.5 * female
        + 1.2 * education_years
        + 6.0 * (log_income - 10.0)
        + 0.8 * urban
        + 0.06 * tenure_months
        + 0.0020 * baseline_spend
        + 0.35 * purchases_12m
        - 0.60 * returns_12m
        - 0.35 * support_tickets_6m
        + 0.03 * app_sessions_30d
        - 0.015 * days_since_last_purchase
        + 0.7 * email_opt_in
        + 0.04 * promo_exposure_30d
        - 1.4 * prior_churn
        + 0.010 * (credit_score - 650.0)
        + 0.9 * satisfaction_score
        + 0.6 * region_1
        + 0.3 * region_2
        - 0.2 * region_3
        - 0.5 * region_4
        + treat * tau
        + eps
    ).astype(np.float32)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=np.int32),
            "y": y,
            "treat": treat,
            "age": age,
            "female": female,
            "education_years": education_years,
            "log_income": log_income,
            "household_size": household_size,
            "urban": urban,
            "tenure_months": tenure_months,
            "baseline_spend": baseline_spend,
            "purchases_12m": purchases_12m,
            "returns_12m": returns_12m,
            "support_tickets_6m": support_tickets_6m,
            "app_sessions_30d": app_sessions_30d,
            "days_since_last_purchase": days_since_last_purchase,
            "email_opt_in": email_opt_in,
            "promo_exposure_30d": promo_exposure_30d,
            "prior_churn": prior_churn,
            "credit_score": credit_score,
            "satisfaction_score": satisfaction_score,
            "region_1": region_1,
            "region_2": region_2,
            "region_3": region_3,
            "region_4": region_4,
        }
    )

    # Sanity: no NAs expected
    if df.isna().any().any():
        raise RuntimeError("Unexpected missing values were generated.")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic experimental dataset (clean, realistic-looking)."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10_000,
        help="Number of observations (default: 10,000).",
    )
    parser.add_argument(
        "--seed", type=int, default=23, help="Random seed (default: 23)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/simulated_data.csv",
        help="Output CSV path (relative to project root).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_dataset(n=args.n, seed=args.seed)

    # CSV is intentionally used for easy cross-language interoperability (Python/R/Stata).
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns to: {out_path}")


if __name__ == "__main__":
    main()
