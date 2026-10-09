# Numerical reproduction and output handling

Reference files under `data/` and `results/` are inputs to independent numerical comparisons. The command-line workflows write their outputs to the Git-ignored `work/` directory and do not overwrite the packaged reference arrays.

Finite-time adaptive-therapy figures use event-localized switching. The node-switched data are retained for time-discretization comparisons, not for the primary regime classification. Global-sensitivity PRCC coefficients are conditional on parameter sets with treatment activation. Discrete control calculations provide numerical candidates, not certificates of global optimality.

The `smoke` mode checks a selected set of grid cells, tabulated outcomes, and packaged file integrity. `grid-full`, `prcc-full` and `control-full` rerun the corresponding numerical experiments. The successful completion of a quick check does not imply completion of full-grid or multistart optimization verification. Numerical tolerances and algorithm parameters are documented in the model code and `docs/PARAMETER_GUIDE.md`.
