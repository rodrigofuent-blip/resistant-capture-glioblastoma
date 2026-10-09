# Supplementary Table S1: projected-stationarity residual

The optimal-control solver uses the exact discrete gradient of the RK4-discretized objective. For an interval-constant control vector `u`, step size `h` and treatment bounds `[0, umax]`, the projected-stationarity residual is

`max(abs(clip(u - grad(J)/(2*h*w_u), 0, umax) - u))`.

The Supplementary Table S1 residuals are evaluated at the **returned control candidate** using this expression. Each archived NPZ file contains the returned vector (`u`) and its trajectory (`y`). The solver summary files `data/control/partial_resistance_exact.csv` and `data/control/partial_resistance_refined_exact.csv` report the associated `best_residual`. These are the source values used for Table S1.

The NPZ array `hist_resid` instead records the projected residual evaluated *before* each relaxed control update. The last iteration-history residual and the residual evaluated at the returned candidate are different evaluations and can differ numerically. Both are preserved in `results/reference_tables/reported_tables/supp_partial_resistant_sensitivity.csv` with separate column headings.

The reported stationarity tolerance is `1e-5`. All eight displayed Table S1 cases satisfy this criterion at their returned candidates. Reproduce the table with `python scripts/run_reproducibility.py tables`; the Table S1 check reads the resulting candidate-based residuals.
