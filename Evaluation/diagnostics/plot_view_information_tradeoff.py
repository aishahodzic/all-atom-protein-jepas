#!/usr/bin/env python3
"""Plot objective-specific view agreement from the paired-view diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from Evaluation.latent_viz.thesis_plot_style import apply_thesis_style


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Evaluation" / "diagnostics" / "results" / "view_information_tradeoff"
OUTPUT = ROOT / "iclr2027" / "figures" / "paired-view-objective-tradeoff.png"

MODELS = {
    "Per-protein objective": (
        RESULTS / "cropped-protein-raw10k.json",
        "#3478B8",
        "o",
    ),
    "Per-residue objective": (
        RESULTS / "cropped-residue-raw10k.json",
        "#E07A2D",
        "s",
    ),
}
STAGES = ["block12_768d", "post_layernorm_768d", "final_mean_8d"]
STAGE_LABELS = ["Block 12\n768D", "Post-LN\n768D", "Final latent\n8D"]


def load_series(path: Path) -> tuple[list[float], list[float]]:
    with path.open() as handle:
        payload = json.load(handle)
    protein_margin = [
        100.0
        * payload["stages"][stage]["protein_view_agreement"][
            "standardized_cosine_margin"
        ]
        for stage in STAGES
    ]
    residue_retrieval = [
        100.0
        * payload["stages"][stage]["same_residue_query_overlap"][
            "within_protein_retrieval_top1"
        ]
        for stage in STAGES
    ]
    return protein_margin, residue_retrieval


def main() -> None:
    apply_thesis_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    x = range(len(STAGES))

    for label, (path, color, marker) in MODELS.items():
        protein_margin, residue_retrieval = load_series(path)
        for panel_index, (ax, values) in enumerate(
            zip(axes, (protein_margin, residue_retrieval))
        ):
            ax.plot(
                x,
                values,
                color=color,
                marker=marker,
                markersize=7,
                linewidth=2.4,
                label=label,
            )
            if panel_index == 0:
                label_offset = -14 if label.startswith("Per-protein") else 10
            else:
                label_offset = 10 if label.startswith("Per-protein") else -15
            for index, value in enumerate(values):
                ax.annotate(
                    f"{value:.1f}",
                    (index, value),
                    xytext=(0, label_offset),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=color,
                )

    axes[0].set_title("A  Protein-level view agreement", loc="left")
    axes[0].set_ylabel("Standardized cosine margin (×100)")
    axes[0].set_ylim(48, 97)

    axes[1].set_title("B  Residue correspondence", loc="left")
    axes[1].set_ylabel("Same-residue retrieval, top-1 (%)")
    axes[1].set_ylim(18, 92)

    for ax in axes:
        ax.set_xticks(list(x), STAGE_LABELS)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
