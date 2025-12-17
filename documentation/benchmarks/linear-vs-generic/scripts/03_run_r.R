# simulated-data benchmark (R)
# - reads CSV created by 01_make_simulated_data.py
# - estimates OLS via lm()
# - runs randomization inference via ritest
# - writes output to: documentation/benchmarks/linear-vs-generic/output/r.txt
#
# Run from repo root:
#   Rscript documentation/benchmarks/linear-vs-generic/scripts/04_run_r.R

library(ritest)

# ---- paths (relative to repo root) ----
data_path <- "documentation/benchmarks/linear-vs-generic/data/simulated_data.csv"
out_path  <- "documentation/benchmarks/linear-vs-generic/output/r.txt"

# ---- logging ----
sink(out_path, split = TRUE)
cat("=== ritest linear-vs-generic benchmark (R) ===\n")
cat("Timestamp:", format(Sys.time()), "\n\n")

# ---- load data (assume clean) ----
df <- read.csv(data_path)

# ---- model ----
# Keep formula aligned with the Python benchmark.
fml <- y ~ treat +
  age + female + education_years + log_income + household_size + urban + tenure_months +
  baseline_spend + purchases_12m + returns_12m + support_tickets_6m + app_sessions_30d +
  days_since_last_purchase + email_opt_in + promo_exposure_30d + prior_churn + credit_score +
  satisfaction_score + region_1 + region_2 + region_3 + region_4

est <- lm(fml, data = df)

cat("OLS (lm) fit:\n")
print(summary(est))

# ---- randomization inference ----
# Parameters: reps=2000, seed=23, no cluster, no strata, two-sided default.
tic <- Sys.time()
ri <- ritest(est, "treat", reps = 2000, seed = 23L)
toc <- Sys.time() - tic

cat("\nRI runtime:\n")
print(toc)

cat("\nRI result:\n")
print(ri)

cat("\n=== END OF LOG ===\n")
sink()
