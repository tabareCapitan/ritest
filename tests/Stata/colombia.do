// edited by tabarecapitan.com
// downloaded 2025-06-03


**** Example 3: Iacovone and McKenzie on supply chain shortening - clustered randomization, with 63 clusters

cd "D:/Dropbox/T/projects/ritest/tests/data/"
	
use "clusterColombia.dta", clear

count

export delimited using "colombia.csv", replace // for Python tests


areg dayscorab b_treat b_dayscorab miss_b_dayscorab round2 round3, cluster(b_block) a(b_pair)

timer clear 	
timer on 1  	

ritest b_treat _b[b_treat], nodots reps(5000) cluster(b_block) strata(b_pair) seed(546): areg dayscorab b_treat b_dayscorab miss_b_dayscorab round2 round3, cluster(b_block) a(b_pair)

timer off 1  	
timer list 1    // 313.3710 (5.2238 minutes) for 5000 replications
