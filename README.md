# Finite-time resistant capture: computational reproducibility

**Associated article:** *Finite-Time Resistant Capture and Resistance-Aware Control in a Reduced Phenotypic Glioblastoma Model*  
**Authors:** Rodrigo Fuentes Axtell and Juan Belmonte-Beitia

The repository contains code, parameter definitions, numerical inputs, reference outputs, figures and procedures used to reproduce the finite-time adaptive-therapy and resistance-aware control calculations. The accompanying research article and Supplementary Information are not included in this computational repository. Time and treatment intensity are nondimensional; the calculations do not represent patient-calibrated temozolomide schedules.

## Repository contents

```text
paper1-reproducibility/
├── data/              Numerical reference inputs, trajectories and comparison arrays
├── src/               Model, solver, sensitivity and plotting implementations
├── scripts/           Reproduction commands and reference checks
├── results/
│   ├── figures/       Figures 1–6 and S1–S7
│   ├── reference_tables/  Machine-readable numerical summaries
│   └── diagnostics/   Supplementary computational checks
├── docs/              Figure and table map, parameters, and integrity manifests
├── requirements.txt   Python dependencies
├── CITATION.cff        Software citation metadata
└── README.md
```

All workflows are provided as Python scripts; no notebook runtime is required. Newly generated output is written to `work/`, which is excluded from version control.

## Installation

A Python environment with compatible wheels for the pinned dependencies is required. Tested environment: Python 3.13.5, CPU installation of PyTorch.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Initial checks

```bash
python scripts/run_reproducibility.py preflight
python scripts/run_reproducibility.py smoke
```

`preflight` checks the package inventories, SHA-256 digests, grid sizes, LHS records and control trajectory arrays. `smoke` regenerates numerical summaries and compares selected grid cells. The latter does not rerun the optimal-control optimizations.

## Numerical reproduction

```bash
python scripts/run_reproducibility.py grid-full
python scripts/run_reproducibility.py prcc-full
python scripts/run_reproducibility.py prcc-figures
python scripts/run_reproducibility.py control-quick
python scripts/run_reproducibility.py control-full
python scripts/run_reproducibility.py tables
python scripts/run_reproducibility.py temporal
python scripts/run_reproducibility.py event-validation
python scripts/run_reproducibility.py protocol-table
python scripts/run_reproducibility.py gradient-check
python scripts/run_reproducibility.py control-dop853-check
python scripts/run_reproducibility.py all-full
```

`grid-full` recomputes 50×50, 25×25 and 60×60 event-localized grids. `prcc-full` reruns the 2,000-input LHS ensemble and PRCC calculations on the activation-containing subset (1,628 realizations). `control-full` reruns the control multistart and sensitivity configurations. Full grid and control runs can be computationally intensive. `all-full` executes the full scripted suite. The `work/` output directory is never used as a source of archived manuscript results.

Supplementary Table S1 uses the projected-stationarity residual calculated at the returned candidate. See `docs/TABLE_S1_RESIDUAL_PROVENANCE.md` for its definition and relationship to the iteration-history residual.

## Figures and tables

`docs/MANUSCRIPT_CROSS_REFERENCE.md` links the six main figures, seven supplementary figures, and reported tables to their numerical sources. `docs/FIGURE_MAP.csv` and `docs/OUTPUT_MAP.csv` provide machine-readable cross-references. Main Figure 5 is a conceptual diagram and is distributed as a vector PDF corresponding to the typeset article illustration; its editable drawing source is not part of the computational repository.

## Software citation

`CITATION.cff` contains author and article metadata. A release DOI, when assigned by an archival repository, identifies the specific reproducible software snapshot. The repository does not claim a DOI before one is assigned.
