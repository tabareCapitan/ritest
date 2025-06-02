cd D:/Dropbox/T/projects/ritest/tests

import delimited "ritest_testdata.csv", clear



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
ritest treatment _b[treatment], reps(100) seed(23) nodots right: ///
										regress y_constant treatment



// heterogeneous treatment ✅
ritest treatment _b[treatment], reps(100) seed(23) nodots right: ///
										regress y_heterogeneous treatment
