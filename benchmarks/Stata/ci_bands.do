//generate mock data
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
