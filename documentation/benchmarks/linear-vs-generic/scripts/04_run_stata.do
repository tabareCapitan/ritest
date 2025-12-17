*******************************************************
* simulated-data benchmark (Stata)
* - reads CSV created by 01_make_simulated_data.py
* - estimates OLS via reg
* - runs randomization inference via ritest
* - writes output to: documentation/benchmarks/simulated-data/output/stata.txt
*
* Run from repo root (recommended):
*   stata -b do documentation/benchmarks/simulated-data/scripts/03_run_stata.do
*******************************************************

version 17.0
clear all
set more off


cd "~/projects/ritest/documentation/benchmarks/linear-vs-generic/"


* ---- logging ----
capture log close _all
log using "output/stata.txt", text replace

display "=== ritest linear-vs-generic benchmark (Stata) ==="
display "Timestamp: " c(current_date) " " c(current_time)
display ""

* ---- load data (assume clean) ----
import delimited using "data/simulated_data.csv", clear varnames(1)

* ---- model covariates (aligned with Python/R) ----
local xvars ///
    age female education_years log_income household_size urban tenure_months ///
    baseline_spend purchases_12m returns_12m support_tickets_6m app_sessions_30d ///
    days_since_last_purchase email_opt_in promo_exposure_30d prior_churn credit_score ///
    satisfaction_score region_1 region_2 region_3 region_4

display "OLS (reg) fit:"
reg y treat `xvars'

* ---- randomization inference ----
* Same parameters as Python/R: reps=2000, seed=23, two-sided, no cluster/strata.
display ""
display "RI (ritest) starting..."
timer clear 1
timer on 1

ritest treat _b[treat], reps(2000) seed(23) nodots: reg y treat `xvars'

timer off 1
display ""
display "RI runtime (seconds):"
timer list 1

display ""
display "=== END OF LOG ==="
log close
