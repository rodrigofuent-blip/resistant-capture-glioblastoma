# Parameter guide

## Event-localized ODE model

The baseline model vector is defined in:

`src/event_grids/event_core.py`

The entries of `BASE` are, in order:

`rS, rD, rR, dS, dD, dR, alpha0, alpha1, beta, eta, K, N_on, N_off`.

For single-trajectory exploration, use `run_custom_trajectory.py`; it accepts these parameters as command-line options and therefore does not require source-code edits.

## Grid experiments

`src/reproduction/grids/reproduce_all_grids.py` defines the reported grid configurations:

- 50x50 initial-condition grid: fixed `D0=0.005`, `T=1000`, `N_on=0.35`, `N_off=0.20`;
- 25x25 `(alpha1, Theta)` grid: fixed `(S0,D0,R0)=(0.20,0.005,0)`, `T=1000`, `N_on=0.35`, `N_off=0.20`;
- 60x60 regime grid: fixed `(S0,D0,R0)=(0.50,0,0.02)`, `T=1500`, `N_on=0.55`, `N_off=0.45`.

The grid axes are stored as CSV files in the `data/event_grids/` directory. To explore another domain, copy the driver to a separate working directory and replace the axis definitions and/or fixed values. New output files should be written to a new `--outdir`.

## Optimal control

The optimal-control parameters are defined in:

`src/control/code/control_dynamics.py`

- `P0`: model parameter vector;
- `IC`: initial condition;
- `W['A']`, `W['B']`, `W['C']`: objective-weight vectors.

The order of `P0` is documented directly above the array in the source file. The order of the weight vector is also documented there. The solver `solve_exact(...)` accepts mesh size, iteration budget, stationarity tolerance, relaxation factor, resistant-sensitivity ratio, safety penalty and safety threshold as arguments.

For alternative scientific scenarios, make a copy of `control_dynamics.py` or construct a separate driver with a modified `P0`, `IC` or weight vector. This keeps the reported calculation unchanged while allowing direct sensitivity experiments.

## Global sensitivity / PRCC

The portable workflow is:

`src/reproduction/event_prcc/recompute_lhs_event_portable.py`

The registered LHS input matrix is:

`data/event_prcc_inputs/lhs_samples.csv`

The script reads the 15 sampled predictors listed in `PARAMS` and evaluates the event-localized model for all rows. To investigate different ranges or a different design, create a new CSV with the same column names and point `SAMPLE` to that file in a copied working version of the script. Keep the original input file unchanged when reproducing the reported PRCCs.

## Numerical tolerances

Reported event-localized production runs use maximum RK4 step `0.1` and event localization by bisection to a bracket width of `1e-11` model-time units. The independent five-case DOP853 validation uses `rtol=1e-11`, `atol=1e-13`, `max_step=2`, with an additional `max_step=0.25` comparison. Optimal-control independent state/cost checks use the tolerances defined in `control_dynamics.py`.
