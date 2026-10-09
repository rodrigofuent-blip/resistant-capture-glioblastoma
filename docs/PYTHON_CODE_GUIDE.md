# Python code guide

The table below describes every Python file included in this package.
| File | Purpose |
|---|---|
| `src/event_grids/event_core.py` | Core S-D-R ODE implementation with RK4 integration, event-localized adaptive switching, node-switched comparison and continuous-MTD simulation. |
| `src/control/code/check_baseline.py` | Checks baseline optimal-control quantities and consistency of the packaged control configuration. |
| `src/control/code/discrete_gradient_check.py` | Independent reverse-mode/automatic-differentiation check of the exact discrete control gradient. |
| `src/control/code/control_discrete_adjoint.py` | Exact discrete adjoint and projected-gradient solver for interval-constant controls using the same RK4 discretization as the forward model. |
| `src/control/code/control_dynamics.py` | Optimal-control model equations, objective weights, forward RK4 integration, adjoint utilities, projected controls and DOP853 state/cost evaluation. |
| `src/control/code/control_model.py` | Unified optimal-control workflow used to generate and summarize A/B/C control candidates and robustness cases. |
| `src/control/code/run_exact.py` | Runs the exact-discrete baseline/multistart optimal-control calculations and writes NPZ/CSV outputs. |
| `src/control/code/run_exact_tail.py` | Continues selected exact-discrete optimization runs from a computed initial control. |
| `src/control/code/run_partial_resistance.py` | Runs optimal-control sensitivity calculations for nonzero resistant-cell treatment sensitivity. |
| `src/control/code/test_discrete_adjoint.py` | Numerical test suite for the exact discrete adjoint/gradient implementation. |
| `src/control/code/validate_selected_exact.py` | Independent DOP853 validation of selected exact-discrete control trajectories and objective values. |
| `src/sensitivity/plot_prcc.py` | Creates PRCC heatmap, aggregate ranking and output-specific tornado plots from packaged PRCC coefficient files. |
| `src/reproduction/control/plot_control_from_reference_npz.py` | Reconstructs the optimal-control multipanel figure from packaged NPZ trajectories. |
| `src/reproduction/event_prcc/core/event_core.py` | Core S-D-R ODE implementation with RK4 integration, event-localized adaptive switching, node-switched comparison and continuous-MTD simulation. |
| `src/reproduction/event_prcc/recompute_lhs_event_portable.py` | Portable 2000-run LHS event-localized recomputation and PRCC analysis using the registered input matrix. |
| `src/reproduction/grids/plot_grids_from_reference_csv.py` | Replots grid diagnostics directly from packaged grid CSV matrices. |
| `src/reproduction/grids/reproduce_all_grids.py` | Recomputes representative or complete 50x50, 25x25 and 60x60 event-localized grids and compares them with reference matrices. |
| `scripts/paths.py` | Central locations of source code, numerical references, and runtime outputs. |
| `scripts/output_safety.py` | Prevents documented reproduction drivers from writing into packaged reference-output directories. |
| `docs/verify_manifests.py` | Verifies SHA-256, file size, and inventory completeness for the package and numerical subdirectories. |
| `scripts/reproduce_control.py` | Runs the reported A/B/C controls, Scenario-A continuation/refinement, partial-resistance cases and soft burden-safety cases. |
| `scripts/reproduce_event_validation.py` | Runs the five event-localized validation points and compares the production RK4/event solver with an independent DOP853 event implementation. |
| `scripts/reproduce_protocol_comparator_table.py` | Recomputes the protocol/candidate comparison table and performs an independent DOP853 comparator check. |
| `scripts/reproduce_reported_summary_tables.py` | Regenerates summary tables from event-grid, PRCC and exact-discrete control outputs. |
| `scripts/reproduce_temporal_convergence.py` | Regenerates the representative node-step temporal-convergence table and the event-localized reference row. |
| `scripts/run_custom_trajectory.py` | Runs a single adaptive, continuous-treatment or untreated trajectory with command-line parameter overrides and records the complete parameter set. |
| `scripts/verify_supp_table_S1.py` | Checks eight Supplementary Table S1 rows against candidate-based projected residuals. |
| `scripts/run_reproducibility.py` | Top-level command dispatcher for quick checks and complete reproduction modes. |
| `scripts/verify_reference_data.py` | Checks dimensions and key invariants of the packaged numerical reference files. |
