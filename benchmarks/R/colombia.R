# install.package
# remotes::install_github("grantmcdermott/ritest")
# install.packages("fixest")

# Load required packages (add this if you're not in the vignette)
library(ritest)
library(fixest)

# Load the bundled Colombia dataset
data("colombia")

# Run fixed effects model with clustered SEs
co_est <- fixest::feols(
  dayscorab ~ b_treat + b_dayscorab + miss_b_dayscorab + round2 + round3 | b_pair,
  vcov = ~b_block,
  data = colombia
)

# Show coefficient table
# summary(co_est)
co_est

# Start timer
tic <- Sys.time()

# Run randomization inference
co_ri = ritest(co_est, 'b_treat', cluster='b_block', strata='b_pair', reps=5e3, seed=546L)

# Stop timer
toc <- Sys.time() - tic
print(toc)  # 26.03 s

# Print test result
co_ri
# print(co_ri)

# Plot null distribution
plot(co_ri, type = "hist", highlight = "fill")
