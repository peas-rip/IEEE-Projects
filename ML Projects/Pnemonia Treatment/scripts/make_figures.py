#!/usr/bin/env python3
"""Generate the report figures from ``results/reproduction_run.json``.

The figures are deliberately built from *aggregate* artifacts (confusion
matrices and headline metrics) because the original Colab run did not persist
per-image predictions. Rerun the evaluation scripts with ``--json`` to refresh
``results/reproduction_run.json``, then rerun this script.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --input results/reproduction_run.json --outdir docs/figures

Writes (PNG, 200 dpi):
    fig_pipeline_confusion.png     3-class end-to-end confusion matrix
    fig_stage1_confusion.png       Normal vs Pneumonia confusion matrix
    fig_stage2_confusion.png       Bacterial vs Viral confusion matrix (routed set)
    fig_member_accuracy.png        Per-member accuracy vs. paper targets, both stages
    fig_ablations.png              End-to-end accuracy under each ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DPI = 200
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"

# Colour-blind-safe accents
BLUE = "#2563eb"
AMBER = "#d97706"
RED = "#dc2626"
GREEN = "#059669"
GREY = "#9ca3af"


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def _save(fig, outdir: Path, name: str) -> Path:
    path = outdir / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[figure] wrote {path}")
    return path


def confusion_figure(
    cm,
    class_names,
    title: str,
    subtitle: str,
    cmap: str,
    filename: str,
    outdir: Path,
):
    """Heatmap of a confusion matrix annotated with counts and row-normalised share."""
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    row_totals = cm.sum(axis=1, keepdims=True)
    row_share = np.divide(cm, row_totals, out=np.zeros_like(cm), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(1.9 * n + 2.0, 1.7 * n + 2.1))
    image = ax.imshow(row_share, cmap=cmap, vmin=0.0, vmax=1.0)

    for i in range(n):
        for j in range(n):
            share = row_share[i, j]
            ax.text(
                j, i,
                f"{int(cm[i, j])}\n{share * 100:.1f}%",
                ha="center", va="center",
                fontsize=10,
                fontweight="bold" if i == j else "normal",
                color="white" if share > 0.55 else INK,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, pad=26)
    ax.text(
        0.5, 1.015, subtitle,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, color=MUTED,
    )
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of true class (row-normalised)", fontsize=9, color=MUTED)
    colorbar.ax.tick_params(labelsize=8)

    return _save(fig, outdir, filename)


def member_accuracy_figure(results: dict, outdir: Path):
    """Two panels: Stage 1 and Stage 2 member/ensemble accuracy vs. paper target."""
    stage1 = results["stage1"]
    stage2 = results["stage2"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"wspace": 0.32})

    panels = [
        (
            axes[0],
            stage1,
            "Stage 1 — Normal vs Pneumonia",
            results["paper_targets"]["stage1_accuracy"],
            f"Paper ensemble: {results['paper_targets']['stage1_accuracy'] * 100:.2f}%",
            "member weight",
        ),
        (
            axes[1],
            stage2,
            "Stage 2 — Bacterial vs Viral",
            results["paper_targets"]["stage2_accuracy"],
            f"Paper ensemble: {results['paper_targets']['stage2_accuracy'] * 100:.2f}%",
            "member weight",
        ),
    ]

    for ax, stage, title, paper_acc, paper_label, _ in panels:
        members = stage["members"]
        labels = [m["name"] for m in members] + ["ensemble"]
        member_values = [m["accuracy"] * 100 for m in members]
        values = member_values + [stage["ensemble"]["accuracy"] * 100]

        # Flag collapsed members (accuracy more than 12 points below the best member).
        best = max(member_values)
        colours = [RED if v < best - 12 else GREY for v in member_values] + [BLUE]

        bars = ax.bar(range(len(labels)), values, color=colours, width=0.62, zorder=3)
        ax.axhline(paper_acc * 100, color=AMBER, linestyle="--", linewidth=1.4, zorder=2)
        ax.text(
            -0.45, paper_acc * 100 + 1.4, paper_label,
            ha="left", va="bottom", fontsize=8.5, color=AMBER, zorder=4,
        )

        # Value labels sit inside the bar so they can never collide with the
        # dashed paper line or the top of the axes.
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value - 2.5,
                f"{value:.1f}", ha="center", va="top", fontsize=9.5,
                fontweight="bold", color="white", zorder=4,
            )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title, pad=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.text(
        0.5, -0.06,
        "Red bars: collapsed member (>12 points below the best member in that stage).  "
        "Amber dashes: accuracy reported by Godbole et al. (2024) on their own split.",
        ha="center", fontsize=9, color=MUTED,
    )

    return _save(fig, outdir, "fig_member_accuracy.png")


def ablation_figure(results: dict, outdir: Path):
    """Horizontal bars of end-to-end accuracy for each ablation."""
    ablations = results["ablations"]["results"]
    baseline = next(a for a in ablations if a["name"] == "baseline")["accuracy"]

    display = {
        "baseline": "Paper config\n(Stage 1 5/4/3/1, Stage 2 equal)",
        "drop_inception_stage1": "Drop Inception-v3\nfrom Stage 1",
        "stage2_weights_2_3_5": "Stage 2 weights\n2 / 3 / 5",
        "both": "Both",
    }

    names = [display.get(a["name"], a["name"]) for a in ablations]
    values = [a["accuracy"] * 100 for a in ablations]
    deltas = [a["delta_vs_baseline"] * 100 for a in ablations]
    colours = [GREY] + [GREEN if d > 0 else RED for d in deltas[1:]]

    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    positions = np.arange(len(names))[::-1]
    bars = ax.barh(positions, values, color=colours, height=0.56, zorder=3)

    for bar, value, delta in zip(bars, values, deltas):
        label = f"{value:.2f}%" if delta == 0 else f"{value:.2f}%   ({delta:+.2f})"
        ax.text(
            value + 0.4, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha="left", fontsize=9.5, color=INK,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(70, 90)
    ax.set_xlabel("End-to-end 3-class accuracy (%)")
    ax.set_title("Ablations — no retraining, forward passes only", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.text(
        0.5, -0.16,
        f"Baseline is {baseline * 100:.2f}%. The x-axis starts at 70% so the deltas stay visible. "
        "The Stage 2 2/3/5 weights were fitted on this test set, so their gain is optimistic "
        "by roughly 0.5-1.0 points.",
        ha="center", fontsize=8.5, color=MUTED,
    )

    return _save(fig, outdir, "fig_ablations.png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="results/reproduction_run.json",
                        help="Results JSON to plot (default: results/reproduction_run.json)")
    parser.add_argument("--outdir", default="docs/figures",
                        help="Output directory for PNGs (default: docs/figures)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"ERROR: results file not found: {args.input}", file=sys.stderr)
        return 1

    results = load_results(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _style()

    pipeline = results["pipeline"]
    confusion_figure(
        pipeline["confusion_matrix"],
        pipeline["class_order"],
        "End-to-end pipeline: 3-class confusion matrix",
        f"{pipeline['test_set']['num_images']} test images  ·  "
        f"accuracy {pipeline['accuracy'] * 100:.2f}%  ·  AUC {pipeline['auc_ovr']:.4f}",
        "Blues",
        "fig_pipeline_confusion.png",
        outdir,
    )

    stage1 = results["stage1"]
    confusion_figure(
        stage1["ensemble"]["confusion_matrix"],
        stage1["class_order"],
        "Stage 1: Normal vs Pneumonia",
        f"Weighted ensemble (5/4/3/1)  ·  "
        f"accuracy {stage1['ensemble']['accuracy'] * 100:.2f}%  ·  "
        f"recall {stage1['ensemble']['recall'] * 100:.2f}%  ·  "
        f"precision {stage1['ensemble']['precision'] * 100:.2f}%",
        "Oranges",
        "fig_stage1_confusion.png",
        outdir,
    )

    stage2 = results["pipeline"]["stage2_breakdown"]
    confusion_figure(
        stage2["confusion_matrix"],
        results["stage2"]["class_order"],
        "Stage 2: Bacterial vs Viral",
        f"Routed pneumonia images only ({stage2['n_routed']} of {stage2['n_pneumonia']})  ·  "
        f"accuracy {stage2['accuracy_operating_set'] * 100:.2f}%",
        "Greens",
        "fig_stage2_confusion.png",
        outdir,
    )

    member_accuracy_figure(results, outdir)
    ablation_figure(results, outdir)

    print(f"\nDone. {len(list(outdir.glob('*.png')))} figures in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
