# Correspondence with manuscript figures and tables

The companion Main and Supplementary Information manuscripts are submitted separately through the journal. They are not included in this computational archive. This index gives the manuscript numbering and the associated reference output; filenames in the numerical directories retain the experimental identifiers used by the executable code.

## Main figures

| Figure | Reference graphic | Reproducibility data |
|---|---|---|
| 1 — model structure | `results/figures/main/Fig01_model_structure.png` | Model equations and parameters: `docs/PARAMETER_GUIDE.md` and Python model implementations. Conceptual schematic, not a numerical plot. |
| 2 — initial-condition landscape | `results/figures/main/Fig02_initial_condition_landscapes.pdf` | `grid50_event_*.csv`; `grid-full` (50×50). |
| 3 — finite-time regime map | `results/figures/main/Fig03_finite_time_regime_map.png` | `grid60_event_*.csv`; `grid-full` (60×60). |
| 4 — activation-conditioned PRCC heatmap | `results/figures/main/Fig04_activation_conditioned_PRCC_heatmap.pdf` | `prcc_event_15_predictors_activation_subset.csv`; `prcc-full` and `prcc-figures`. |
| 5 — mechanistic decomposition | `results/figures/main/Fig05_mechanistic_decomposition.pdf` | Conceptual diagram rendered as a vector crop from page 26 of the Main PDF. Its original editable drawing source is not part of this repository. |
| 6 — open-loop control candidates | `results/figures/main/Fig06_open_loop_control_candidates.pdf` | `data/control/exact_discrete_summary.csv`, `data/control/exact_discrete_tail_summary.csv`, `control-quick`. |

## Supplementary figures

| Figure | Reference graphic | Reproducibility data |
|---|---|---|
| S1 | `results/figures/supplementary/FigS01_PRCC_terminal_resistant_fraction.pdf` | Event-localized activation-conditioned PRCC data. |
| S2 | `results/figures/supplementary/FigS02_PRCC_cumulative_exposure.pdf` | Same PRCC output, selected observable `A_u`. |
| S3 | `results/figures/supplementary/FigS03_PRCC_cumulative_resistant_burden.pdf` | Same PRCC output, selected observable `A_R`. |
| S4 | `results/figures/supplementary/FigS04_event_localized_25x25_diagnostics.pdf` | `grid25_event_*.csv`; event-localized 25×25 sweep. |
| S5 | `results/figures/supplementary/FigS05_cumulative_burden.png` | `grid60_event_A_R.csv` and `grid60_event_A_N.csv`. |
| S6 | `results/figures/supplementary/FigS06_tradeoff_structure.png` | Event-localized 60×60 grid outputs and operational regime labels. |
| S7 | `results/figures/supplementary/FigS07_mean_absolute_PRCC.pdf` | `prcc_event_15_predictors_activation_subset.csv`; `prcc-figures`. |

## Manuscript tables

| Manuscript table(s) | Machine-readable reference |
|---|---|
| Main Table 1 (baseline model parameters) | `docs/PARAMETER_GUIDE.md`; model code definitions. |
| Main Table 2 (cycle-stratified) | `results/reference_tables/reported_tables/main_cycle_stratified_summary.csv`. |
| Main Table 3 (adaptive vs continuous-pressure) | `results/reference_tables/reported_tables/main_AT_vs_MTD_summary.csv`. |
| Main Table 4 (25×25 parametric summary) | `results/reference_tables/reported_tables/main_parametric_25x25_summary.csv`. |
| Main Table 5 (control objective weights) | `docs/PARAMETER_GUIDE.md` and the optimization scenario definitions. |
| Supplement Table S1 (partial resistance) | `results/reference_tables/reported_tables/supp_partial_resistant_sensitivity.csv` and `docs/TABLE_S1_RESIDUAL_PROVENANCE.md`. |
| Supplement Table S2 (soft safety) | `results/reference_tables/reported_tables/supp_soft_safety_summary.csv`. |
| Supplement Table S3 (LHS ranges and seed) | `docs/PARAMETER_GUIDE.md` and LHS-generation code. |
| Supplement Table S4 (temporal switching) | `results/reference_tables/temporal_switching_table.csv`. |
| Supplement Table S5 (numerical protocol) | `docs/PARAMETER_GUIDE.md`. |
| Supplement Table S6 (risk threshold sensitivity) | `results/reference_tables/reported_tables/supp_threshold_sensitivity.csv`. |
| Supplement Table S7 (parameter provenance) | `docs/PARAMETER_GUIDE.md`; parameter provenance discussion in the Supplementary Information. |
| Supplement Table S8 (protocol comparators) | `results/reference_tables/protocol_control_candidate_outcomes.csv`. |
| Supplement Table S9 (stationarity) | `data/control/` and `data/control_checks/refinement_summary.csv`. |

Parameter interpretation, event-localization, sensitivity-conditioning criteria, and limits on optimality are defined by the manuscripts. CSV precision is typically higher than the displayed manuscript tables. The computational record does not represent patient-calibrated treatments.
