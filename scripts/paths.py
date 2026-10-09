"""Repository paths for independent numerical reproduction."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GRID_DATA = ROOT / "data/event_grids"
CONTROL_DATA = ROOT / "data/control"
SENSITIVITY_DATA = ROOT / "data/sensitivity"
CONTROL_CODE = ROOT / "src/control/code"
CONTROL_CHECKS = ROOT / "data/control_checks"
GRID_CODE = ROOT / "src/event_grids"
REPRODUCE = ROOT / "src/reproduction"
PRCC_INPUTS = ROOT / "data/event_prcc_inputs"
REF = ROOT / "results/reference_tables"
WORK = ROOT / "work"
