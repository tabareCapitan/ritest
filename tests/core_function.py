import pandas as pd 
import numpy as np
from ritest import ritest

np.random.seed(23)
n = 300

# Treatment assignment
treatment = np.random.binomial(1, 0.5, size=n)

# Reduced constant effect: 1.0 instead of 1.5, more noise
y_constant = 20.0 + 1.0 * treatment + np.random.normal(0, 3, size=n)

# Heterogeneous effects: still average to 1.0, with variation
tau = np.random.normal(loc=1.0, scale=0.6, size=n)
tau[treatment == 0] = 0  # zero effect for control group
y_heterogeneous = 20.0 + tau + np.random.normal(0, 3, size=n)

df = pd.DataFrame({
    "treatment": treatment,
    "y_constant": y_constant,
    "y_heterogeneous": y_heterogeneous
})

df.to_csv("tests/ritest_testdata.csv", index=False)


df["y"] = df["y_constant"]  # Use heterogeneous effects for testing

res = ritest(
    df=df,
    formula="y ~ treatment",
    stat="treatment",
    permute_var="treatment",
    alternative="two-sided"
)

print(res)
