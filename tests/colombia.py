import pandas as pd
from ritest import ritest, ritest_set
from time import time
from linearmodels.panel import PanelOLS
from pyfixest.estimation import feols

df = pd.read_csv(r"tests/data/colombia.csv")

#  clean data

df["dayscorab"] = df["dayscorab"].replace({".": None, ".r": None, ".d": None})
df["dayscorab"] = pd.to_numeric(df["dayscorab"], errors="raise")

df["b_treat"] = df["b_treat"].map({"Control": 0, "Treatment": 1})

df["miss_b_dayscorab"] = pd.to_numeric(df["miss_b_dayscorab"], errors="raise")
df["round2"] = pd.to_numeric(df["round2"], errors="raise")
df["round3"] = pd.to_numeric(df["round3"], errors="raise")


ritest_set({
    "reps": 1000,
    "seed": 546,
    "ci_method": "cp",   # matches Stata
    "alpha": 0.05
})


# Run permutation inference
t0 = time()

def make_fe_stat_fn_PanelOLS(coef: str):
    def fn(df):
        # Index needs to be [unit, time] — just use a dummy 'time' index if not panel
        # Here, use 'obs_id' if available or create one
        df = df.copy()
        if 'obs_id' not in df.columns:
            df['obs_id'] = range(len(df))
        df = df.set_index(['b_pair', 'obs_id'])
        # Prepare the exog and endog (drop the fixed effect col itself)
        exog = df[['b_treat', 'b_dayscorab', 'miss_b_dayscorab', 'round2', 'round3']]
        endog = df['dayscorab']
        mod = PanelOLS(endog, exog, entity_effects=True)  # entity_effects = b_pair FE
        res = mod.fit(cov_type='clustered', clusters=df['b_block'])
        return res.params[coef]
    return fn


def make_fe_stat_fn_pyfixest(coef: str):
    def fn(df):
        # Ensure b_treat is numeric (0/1), not string
        if df["b_treat"].dtype == object:
            df = df.copy()
            df["b_treat"] = df["b_treat"].map({"Control": 0, "Treatment": 1})

        # Fit fixed effects model using pyfixest syntax
        fml = "dayscorab ~ b_treat + b_dayscorab + miss_b_dayscorab + round2 + round3 | b_pair"
        res = feols(fml, data=df, vcov={"CRV1": "b_block"})
        return res.coef()[coef]
    return fn




stat_fn = make_fe_stat_fn_pyfixest("b_treat")

res = ritest(
    df=df,
    stat_fn=stat_fn,
    permute_var="b_treat",
    strata="b_pair",
    cluster="b_block"
)

t1 = time()
print(f"\nElapsed time: {t1 - t0:.2f} seconds")


print(f"Observed statistic: {res['stat']:.5f}")
print(f"c = {res['count']}")
print(f"n = {res['reps']}")
print(f"p = c/n = {res['pval']:.4f}")
print(f"SE(p): {res['pval_se']:.4f}")
print(f"P-value CI: ({res['pval_ci'][0]:.4f}, {res['pval_ci'][1]:.4f})")
