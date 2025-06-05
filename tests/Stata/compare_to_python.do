cd D:/Dropbox/T/projects/ritest/tests

import delimited "./data/ritest_testdata.csv", clear



*** TWO-SIDED ******************************************************************

// constant treatment ✅
ritest treatment _b[treatment], reps(100) seed(23) nodots: ///
										regress y_constant treatment



// heterogeneous treatment ✅
ritest treatment _b[treatment], reps(100) seed(23) nodots: ///
										regress y_heterogeneous treatment



*** LEFT-SIDED *****************************************************************

// constant treatment ✅
ritest treatment _b[treatment], reps(100) seed(23) nodots left: ///
										regress y_constant treatment



// heterogeneous treatment ✅
ritest treatment _b[treatment], reps(100) seed(23) nodots left: ///
										regress y_heterogeneous treatment

*** RIGHT-SIDED ****************************************************************

// constant treatment ✅
ritest treatment _b[treatment], reps(1000) seed(23) nodots right: ///
										regress y_constant treatment



// heterogeneous treatment ✅
ritest treatment _b[treatment], reps(1000) seed(23) nodots right: ///
										regress y_heterogeneous treatment



*** CI COEF ******************************************************************

//generate mock data (from Stata's ritest readme)
set seed 123
clear
set obs 100
gen treatment = _n>_N/2 //half are treated
gen y = 0.3*treatment + rnormal() //there's a treatment effect
reg y treatment //this is the standard ols result

//run ritest to find which hypotheses for the treatment effect in [-1,1] can[not] be rejected
tempfile gridsearch
postfile pf TE pval using `gridsearch'
forval i=-1(0.05)1 {
	qui ritest treatment (_b[treatment]/_se[treatment]), reps(500) null(y `i') seed(123): reg y treatment //run ritest for the ols reg with the studentized treatment effect
	mat pval = r(p)
	post pf (`i') (pval[1,1])
}
postclose pf

//show results to illustrate confidence intervals
use `gridsearch', clear
tw line pval TE , yline(0.05)
