#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_prcc.py

Reproducibility script for the global-sensitivity PRCC figures used in:

- main article:
  * prcc_global_PRCC_heatmap.png
  * prcc_aggregate_PRCC_ranking.png

- supplementary material:
  * prcc_tornado_phi_R_T.png
  * prcc_tornado_A_u.png
  * prcc_tornado_A_R.png

Inputs
------
A directory containing:
    prcc_rankings.csv
    lhs_results.csv          [kept for provenance/checking]
    lhs_samples.csv          [kept for provenance/checking]
    run_metadata.json        [kept for provenance/checking]

The required plotting input is prcc_rankings.csv with columns:
    output, parameter, PRCC, abs_PRCC

Usage
-----
python plot_prcc.py \
    --input-dir data/sensitivity/plot_input \
    --output-dir work/prcc_figures

This script generates named PRCC diagnostics and reference figures. It reads the PRCC coefficient files and
replots the final PRCC results from the archived CSV output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PARAM_LABELS = {
    "rS": r"$r_S$",
    "rD": r"$r_D$",
    "rR": r"$r_R$",
    "dS": r"$d_S$",
    "dD": r"$d_D$",
    "dD_ratio": r"$d_D/d_S$",
    "dR": r"$d_R$",
    "alpha0": r"$\alpha_0$",
    "alpha1": r"$\alpha_1$",
    "beta": r"$\beta$",
    "eta": r"$\eta$",
    "Theta": r"$\Theta$",
    "N_on": r"$N_{\mathrm{on}}$",
    "N_off": r"$N_{\mathrm{off}}$",
    "hwidth": r"$\Delta_h$",
    "S0": r"$S_0$",
    "D0": r"$D_0$",
    "R0": r"$R_0$",
    "T": r"$T$",
}

OUTPUT_LABELS = {
    "A_N": r"$A_N$",
    "N_max": r"$N_{\max}$",
    "N_T": r"$N(T)$",
    "A_R": r"$A_R$",
    "phi_R_T": r"$\phi_R(T)$",
    "A_u": r"$A_u$",
    "n_cycles": r"$n_{\mathrm{cycles}}$",
    "first_on": r"$t_{\mathrm{first\,on}}$",
}


DEFAULT_OUTPUT_ORDER = [
    "A_N",
    "N_max",
    "N_T",
    "A_R",
    "phi_R_T",
    "A_u",
    "n_cycles",
    "first_on",
]

DEFAULT_PARAMETER_ORDER = [
    "T",
    "R0",
    "S0",
    "rS",
    "rR",
    "N_on",
    "alpha1",
    "dS",
    "dR",
    "hwidth",
    "Theta",
    "eta",
    "beta",
    "alpha0",
    "dD_ratio",
    "rD",
]


def _label_parameter(name: str) -> str:
    return PARAM_LABELS.get(name, name)


def _label_output(name: str) -> str:
    return OUTPUT_LABELS.get(name, name)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save both PNG and PDF with the exact stem used by LaTeX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_prcc(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "prcc_rankings.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    required = {"output", "parameter", "PRCC", "abs_PRCC"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["output"] = df["output"].astype(str)
    df["parameter"] = df["parameter"].astype(str)
    df["PRCC"] = pd.to_numeric(df["PRCC"], errors="raise")
    df["abs_PRCC"] = pd.to_numeric(df["abs_PRCC"], errors="raise")
    return df


def make_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    outputs = [x for x in DEFAULT_OUTPUT_ORDER if x in set(df["output"])]
    if not outputs:
        outputs = sorted(df["output"].unique())

    # Parameters are ordered by aggregated absolute PRCC, with a preferred ordering
    # for the highest-level manuscript narrative.
    agg = (
        df.groupby("parameter", as_index=False)["abs_PRCC"]
        .mean()
        .sort_values("abs_PRCC", ascending=False)
    )
    seen = set()
    params = []
    for p in DEFAULT_PARAMETER_ORDER:
        if p in set(df["parameter"]):
            params.append(p)
            seen.add(p)
    for p in agg["parameter"]:
        if p not in seen:
            params.append(p)
            seen.add(p)

    mat = (
        df.pivot(index="parameter", columns="output", values="PRCC")
        .reindex(index=params, columns=outputs)
        .fillna(0.0)
    )

    fig_h = max(6.2, 0.34 * len(params) + 1.8)
    fig_w = max(8.5, 0.9 * len(outputs) + 2.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat.values, aspect="auto", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(outputs)))
    ax.set_xticklabels([_label_output(o) for o in outputs], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels([_label_parameter(p) for p in params])

    ax.set_title("Global PRCC sensitivity map")
    ax.set_xlabel("Finite-time output")
    ax.set_ylabel("Input parameter")

    # Annotate entries; use compact formatting for readability.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PRCC")
    fig.tight_layout()

    _save_figure(fig, output_dir, "prcc_global_PRCC_heatmap")


def make_aggregate_ranking(df: pd.DataFrame, output_dir: Path, top_n: int = 12) -> None:
    agg = (
        df.groupby("parameter", as_index=False)["abs_PRCC"]
        .mean()
        .sort_values("abs_PRCC", ascending=False)
        .head(top_n)
        .sort_values("abs_PRCC", ascending=True)
    )

    fig_h = max(4.6, 0.38 * len(agg) + 1.2)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    ax.barh([_label_parameter(p) for p in agg["parameter"]], agg["abs_PRCC"])
    ax.set_xlabel("Mean absolute PRCC across outputs")
    ax.set_title("Aggregated global sensitivity ranking")

    for y, val in enumerate(agg["abs_PRCC"]):
        ax.text(val + 0.01, y, f"{val:.2f}", va="center", fontsize=8)

    xmax = min(1.05, max(0.15, float(agg["abs_PRCC"].max()) + 0.12))
    ax.set_xlim(0, xmax)
    fig.tight_layout()

    _save_figure(fig, output_dir, "prcc_aggregate_PRCC_ranking")


def make_tornado(
    df: pd.DataFrame,
    output_name: str,
    output_dir: Path,
    stem: str,
    top_n: int = 12,
) -> None:
    sub = df[df["output"] == output_name].copy()
    if sub.empty:
        raise ValueError(f"No PRCC rows found for output={output_name!r}")

    sub = sub.sort_values("abs_PRCC", ascending=False).head(top_n)
    # Plot from smallest to largest so strongest appears at top.
    sub = sub.sort_values("PRCC", ascending=True)

    fig_h = max(4.6, 0.38 * len(sub) + 1.2)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    y_labels = [_label_parameter(p) for p in sub["parameter"]]
    ax.barh(y_labels, sub["PRCC"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("PRCC")
    ax.set_title(f"Output-specific PRCC ranking for {_label_output(output_name)}")

    for y, val in enumerate(sub["PRCC"]):
        if val >= 0:
            ax.text(val + 0.025, y, f"{val:.2f}", va="center", fontsize=8)
        else:
            ax.text(val - 0.025, y, f"{val:.2f}", va="center", ha="right", fontsize=8)

    fig.tight_layout()
    _save_figure(fig, output_dir, stem)


def write_manifest(input_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    manifest = {
        "input_dir": str(input_dir),
        "n_prcc_rows": int(len(df)),
        "outputs": sorted(df["output"].unique().tolist()),
        "parameters": sorted(df["parameter"].unique().tolist()),
        "generated_files": sorted(p.name for p in output_dir.glob("prcc_*")),
    }

    metadata_path = input_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            manifest["run_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            manifest["run_metadata_error"] = str(exc)

    (output_dir / "prcc_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing prcc_rankings.csv and associated global-sensitivity outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where prcc_* files will be written.",
    )
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    df = load_prcc(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_heatmap(df, args.output_dir)
    make_aggregate_ranking(df, args.output_dir, top_n=args.top_n)
    make_tornado(df, "phi_R_T", args.output_dir, "prcc_tornado_phi_R_T", top_n=args.top_n)
    make_tornado(df, "A_u", args.output_dir, "prcc_tornado_A_u", top_n=args.top_n)
    make_tornado(df, "A_R", args.output_dir, "prcc_tornado_A_R", top_n=args.top_n)
    write_manifest(args.input_dir, args.output_dir, df)

    print("PRCC figures written to:", args.output_dir)
    for p in sorted(args.output_dir.glob("prcc_*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
