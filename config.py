"""Default configuration for ritest."""

DEFAULTS = {
    "reps": 100,           # number of permutations
    "seed": 23,            # for the one and only, MJJ
    "alpha": 0.05,         # significance level for p-values
    "ci_range": 3.0,       # grid search range for coef CI (in se-units of coef)
    "ci_step": 0.005,      # grid resolution
    "ci_method": "cp",     # 'cp' = Clopper-Pearson, 'normal' = Wald
    "n_jobs": -1           # parallel jobs (-1 = all cores)
}
