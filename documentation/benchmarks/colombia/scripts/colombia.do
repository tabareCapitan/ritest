// author: tabarecapitan.com
// data downloaded on 2025-06-03


**** Example 3: Iacovone and McKenzie on supply chain shortening - clustered randomization, with 63 clusters


cd "~/projects/ritest/documentation/benchmarks/colombia/"

import delimited using "data/colombia.csv", clear


tab b_treat, mi

rename b_treat tempvar

	gen b_treat = (tempvar == "Treatment")
	tab b_treat
	drop tempvar

count


log using "output/colombia_stata", text replace

timer clear
timer on 1

ritest b_treat _b[b_treat], nodots reps(5000) cluster(b_block) strata(b_pair) seed(546): areg dayscorab b_treat b_dayscorab miss_b_dayscorab round2 round3, cluster(b_block) a(b_pair)

timer off 1
timer list 1    // GET A CLEAN TIME WITH PC RESTING

log close
