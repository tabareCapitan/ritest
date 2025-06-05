import pandas as pd
import numpy as np
import time
from ritest import ritest
from ritest import ritest, ritest_set



# ------------------------------
# Set random seed and create test data
# ------------------------------
np.random.seed(23)
n = 300

# Binary treatment
treatment = np.random.binomial(1, 0.5, size=n)

# Outcomes with smaller treatment effect and more noise
y_constant = 50.0 + 1.0 * treatment + np.random.normal(0, 25, size=n)

# Heterogeneous treatment effect with same average but more variability
tau = np.random.normal(loc=1.0, scale=1.5, size=n) # change scale for more heterogeneity
tau[treatment == 0] = 0
y_heterogeneous = 50.0 + tau + np.random.normal(0, 25, size=n)

# DataFrame
df = pd.DataFrame({
    "treatment": treatment,
    "y_constant": y_constant,
    "y_heterogeneous": y_heterogeneous
})

df.to_csv("tests/data/ritest_testdata.csv", index=False)

# ------------------------------
# Run tests
# ------------------------------

# Override default config
ritest_set({
    "ci_method": "cp",   # 'cp' or 'normal',
    "reps": 5000,  # Number of permutations
})

start = time.time()

df["y"] = df["y_constant"]  # Use the heterogeneous outcome for testing

res = ritest(
    df=df,
    formula="y ~ treatment",
    stat="treatment",
    permute_var="treatment",
)

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

print("Observed statistic:", res["stat"])
print("c =", res["count"])
print("n =", res["reps"])
print("p = c/n =", res["pval"])
print("SE(p):", res["pval_se"])
print("P-value CI:", res["pval_ci"])