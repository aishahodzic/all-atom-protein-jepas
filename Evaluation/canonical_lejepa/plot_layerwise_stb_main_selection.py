#!/usr/bin/env python3
"""Render the compact main-paper view of uncropped LeJEPA's layerwise STB result."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import plot_layerwise_stb_all_blocks as full


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "iclr2027/figures/main/"
    "canonical_lejepa_uncropped_selected_layerwise_stb.png"
)

# BindShake has one original benchmark split. For tasks with structural
# holdouts, use the same superfamily split rather than selecting by effect size.
PANELS = (
    ("AUROC", "BindShake", "Test"),
    ("AUROC", "CatBio", "SF"),
    ("AUROC", "Rep", "SF"),
    ("AUROC", "BindBio", "SF"),
    ("Spearman", "FlexRMSF", "SF"),
    ("Spearman", "FlexBFactor", "SF"),
    ("Spearman", "FlexNEQ", "SF"),
    ("Macro-F1", "Homo", "Family"),
)

CURVE_COLOR = "#7851A9"
FINAL_COLOR = "#0072B2"
DISPLAY_BLOCKS = tuple(range(1, 13))
FINAL_X = 14.1
REFERENCE_REGION_START = 12.75
X_MAX = 14.8
LATIN_MODERN_DIR = Path(
    "/mnt/data/slurm-storage/aishod/.TinyTeX/texmf-dist/fonts/opentype/public/lm"
)


def main() -> None:
    scores = full.load_scores()
    final_8d = full.load_final_8d_reference()

    for filename in (
        "lmroman10-regular.otf",
        "lmroman10-bold.otf",
        "lmroman10-italic.otf",
        "lmroman10-bolditalic.otf",
    ):
        font_manager.fontManager.addfont(LATIN_MODERN_DIR / filename)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Computer Modern Roman",
                "DejaVu Serif",
            ],
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    fig, axes_grid = plt.subplots(2, 4, figsize=(7.15, 4.75), sharex=False)
    axes = axes_grid.ravel()
    ylabels = {
        0: "AUROC",
        4: r"Spearman's $\rho$",
        7: "Macro-F1",
    }
    for panel_index, (ax, (metric, task, split)) in enumerate(zip(axes, PANELS)):
        values = [
            scores[(metric, task, split, block)][0] * 100
            for block in DISPLAY_BLOCKS
        ]
        final_value = final_8d[(metric, task, split)]
        ax.plot(
            DISPLAY_BLOCKS,
            values,
            color=CURVE_COLOR,
            linewidth=1.15,
            marker="o",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.axvspan(
            REFERENCE_REGION_START,
            X_MAX,
            color="#F2F2F2",
            zorder=0,
        )
        ax.axvline(
            REFERENCE_REGION_START,
            color="#999999",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
        ax.hlines(
            final_value,
            0.7,
            FINAL_X,
            color=FINAL_COLOR,
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            zorder=2,
        )
        ax.scatter(
            [FINAL_X],
            [final_value],
            color=FINAL_COLOR,
            marker="o",
            s=28,
            edgecolor="black",
            linewidth=0.65,
            zorder=4,
        )

        best_index = max(range(len(values)), key=values.__getitem__)
        best_block = DISPLAY_BLOCKS[best_index]
        best_value = values[best_index]
        ax.scatter(
            [best_block],
            [best_value],
            marker="*",
            s=72,
            color=CURVE_COLOR,
            edgecolor="black",
            linewidth=0.45,
            zorder=4,
        )
        finite = values + [final_value]
        spread = max(finite) - min(finite)
        pad = max(1.5, 0.13 * spread)
        ax.set_ylim(min(finite) - pad, max(finite) + pad)
        ax.set_xlim(0.7, X_MAX)
        ax.set_xticks((2, 4, 6, 8, 10, 12, FINAL_X))
        ax.set_xticklabels(("2", "4", "6", "8", "10", "12", "Final\n8D"))
        ax.yaxis.set_major_locator(
            MultipleLocator(
                10 if metric == "Spearman" or task == "CatBio" else 5
            )
        )
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9.0, pad=2)
        ax.get_xticklabels()[-1].set_fontsize(7.0)
        ax.get_xticklabels()[-1].set_color("black")
        ax.tick_params(axis="y", labelsize=9.4)
        # The PNG is scaled to the paper column width. 10.6 pt here renders at
        # approximately 8 pt in the compiled paper.
        ax.set_title(task, fontsize=10.6, fontweight="bold", pad=4)
        ax.set_ylabel(ylabels.get(panel_index, ""), fontsize=9.8, labelpad=2)
        ax.set_xlabel("Block", fontsize=9.0, labelpad=2)
        ax.xaxis.set_label_coords((6.5 - 0.7) / (X_MAX - 0.7), -0.18)

    legend = (
        Line2D(
            [0], [0], color=CURVE_COLOR, marker="o", linewidth=1.15,
            markersize=4, label="Frozen intermediate representation",
        ),
        Line2D(
            [0], [0], color=FINAL_COLOR, marker="o", linestyle="None",
            markeredgecolor="black", markeredgewidth=0.65,
            markersize=4.2, label="Final 8D representation",
        ),
        Line2D(
            [0], [0], color="#555555", marker="*", linestyle="None",
            markersize=8, label="Best block",
        ),
    )
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        # Two rendered points larger than the previous legend after TeX scaling.
        fontsize=11.6,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.3,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.13,
        top=0.81,
        wspace=0.32,
        hspace=0.58,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.png")
    fig.savefig(temporary, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
