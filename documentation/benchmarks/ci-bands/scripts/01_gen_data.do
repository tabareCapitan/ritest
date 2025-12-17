/* Following the example provided in https://github.com/simonheb/ritest */

version 17.0

cd "~/projects/ritest/documentation/benchmarks/ci-bands/"


set seed 123

clear

set obs 100

gen treatment = _n>_N/2 //half are treated

gen y = 0.3*treatment + rnormal() //there's a treatment effect

reg y treatment //this is the standard ols result


export delimited using "data/ci_bands.csv", replace
