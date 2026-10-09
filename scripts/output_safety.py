"""Protect the repository's immutable inputs and reference figures against accidental overwrite."""
from pathlib import Path
from paths import ROOT
IMMUTABLE = [ROOT / "data", ROOT / "results/figures", ROOT / "results/reference_tables", ROOT / "results/diagnostics", ROOT / "src", ROOT / "docs"]
def assert_writable_output(path):
    target=Path(path).resolve()
    for folder in IMMUTABLE:
        protected=folder.resolve()
        if target==protected or protected in target.parents:
            raise ValueError(f"Refusing to write into immutable repository folder: {protected}")
    return target
