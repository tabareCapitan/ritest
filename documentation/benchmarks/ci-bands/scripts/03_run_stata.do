/* Coefficient bands, follows stata's ritest example */

cd "~/projects/ritest/documentation/benchmarks/ci-bands"


capture log close _all

log using "output/stata.txt", text replace

display "=== ritest ci-bands benchmark (Stata) ==="
display "Timestamp: " c(current_date) " " c(current_time)
display ""

import delimited "data/ci_bands.csv", clear


// run regular ritest
ritest treatment (_b[treatment]), nodots reps(500) seed(123): ///
	reg y treatment


timer clear 1
timer on 1

//run ritest to find which hypotheses for the treatment effect in [-1,1] can[not] be rejected
tempfile gridsearch
postfile pf TE pval using `gridsearch'
forval i=-1(0.05)1 {
// 	qui ritest treatment (_b[treatment]/_se[treatment]), reps(500) null(y `i') seed(123): reg y treatment //run ritest for the ols reg with the studentized treatment effect
	qui ritest treatment (_b[treatment]), reps(500) null(y `i') seed(123): reg y treatment // tc: run with _b[treatment] only to compare


	mat pval = r(p)
	post pf (`i') (pval[1,1])
}
postclose pf


timer off 1
display ""
display "RI runtime (seconds):"
timer list 1


//show results to illustrate confidence intervals
use `gridsearch', clear
tw line pval TE , yline(0.05) xline(-0.2 0.2 0.6)
graph export "output/stata_ci_bands.png", replace

display ""
display "=== END OF LOG ==="
log close
