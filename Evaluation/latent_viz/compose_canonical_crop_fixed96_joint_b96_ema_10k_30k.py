"""Compose publication-friendly EMA-30k atlas and EMA 10k/30k comparison."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
for path in (
    REPO,
    REPO / "Evaluation/linear_probing",
    REPO / "Evaluation/knn_purity",
    HERE,
):
    sys.path.insert(0, str(path))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array  # noqa: E402
from run_tsne import (  # noqa: E402
    class_indices,
    coordinate_path,
    scatter_panel,
    view_labels_colors,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


IDS = {
    "EMA-10k": (
        "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
        "ema-step10000"
    ),
    "EMA-30k": (
        "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
        "ema-step30000"
    ),
}
N_POINTS = 15_000
PERPLEXITY = 30.0
ATLAS = HERE / "figures/canonical_crop_fixed96_joint_b96_ema30k_tsne_atlas.png"
COMPARISON = HERE / (
    "figures/canonical_crop_fixed96_joint_b96_ema10k_vs_ema30k_tsne_comparison.png"
)
RAW_EMA_COMPARISON = HERE / (
    "figures/canonical_crop_fixed96_joint_b96_raw_vs_ema_10k_30k_aa_tsne.png"
)
RAW_COMPARISON = HERE / (
    "figures/canonical_crop_fixed96_joint_b96_raw10k_vs_raw30k_tsne_comparison.png"
)


def categorical_legend(ax, view: str, *, fontsize: float = 6.5) -> None:
    labels, colors = view_labels_colors(view)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=4, label=label)
        for label, color in zip(labels, colors)
    ]
    columns = 4 if view == "aa20" else 1
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=columns,
        fontsize=fontsize,
        frameon=False,
        columnspacing=0.7,
        handletextpad=0.25,
    )


def continuous_panel(ax, embedding, values, title, cmap, vmin, vmax, label) -> None:
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=values, cmap=cmap,
        vmin=vmin, vmax=vmax, s=2.2, alpha=0.65, linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.015)
    colorbar.set_label(label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def main() -> None:
    labels = load_labels()
    caches = {name: load_encoder_latents(checkpoint_id) for name, checkpoint_id in IDS.items()}
    protein_ids = common_protein_ids(caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    aa = selection.aa[sampled]
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate([
        np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
        for protein_id, indices in selection.keys
    ])[sampled]
    normalized_position = (residue_index + 1.0) / lengths
    embeddings = {
        name: np.load(
            coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED),
            allow_pickle=False,
        )["embedding"]
        for name, checkpoint_id in IDS.items()
    }

    # All EMA-30k colorings in one coordinate-identical atlas.
    embedding = embeddings["EMA-30k"]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.1))
    for ax, view, title in zip(
        axes[0],
        ("aa20", "charge", "aromatic", "size"),
        ("Amino-acid identity", "Charge", "Aromaticity", "Side-chain size"),
    ):
        scatter_panel(ax, embedding, aa, view=view, annotate_aa=(view == "aa20"))
        ax.set_title(title, fontsize=10)
        categorical_legend(ax, view)
    continuous_panel(
        axes[1, 0], embedding, residue_index + 1, "Absolute sequence index",
        "turbo", 1, 256, "residue index i",
    )
    continuous_panel(
        axes[1, 1], embedding, normalized_position, "Normalized sequence position",
        "viridis", 0, 1, "i/L",
    )
    continuous_panel(
        axes[1, 2], embedding, lengths, "Source chain length",
        "magma", 50, 256, "chain length L",
    )
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.04, 0.72,
        "Canonical LeJEPA\nfixed global/local crops: 96/48\n"
        "protein loss + joint-view SIGReg\nglobal batch: 96\ntraining step: 30k\nweights: EMA\n\n"
        "All panels use identical coordinates\nand the same 15,000 residues.",
        transform=axes[1, 3].transAxes, va="top", fontsize=11, linespacing=1.4,
    )
    fig.suptitle("EMA-30k residue-latent t-SNE atlas", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.6)
    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ATLAS, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Matched biochemical comparison: rows are steps, columns are colorings.
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.3))
    views = ("aa20", "charge", "aromatic", "size")
    titles = ("Amino-acid identity", "Charge", "Aromaticity", "Side-chain size")
    for row, name in enumerate(IDS):
        for column, (view, title) in enumerate(zip(views, titles)):
            ax = axes[row, column]
            scatter_panel(ax, embeddings[name], aa, view=view, annotate_aa=False)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if column == 0:
                ax.set_ylabel(name, fontsize=12, weight="bold")
            categorical_legend(ax, view)
    fig.suptitle(
        "EMA-10k versus EMA-30k: matched residue-chemistry organization",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.7)
    fig.savefig(COMPARISON, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Direct raw-versus-EMA comparison at both training steps.
    raw_ids = {
        "10k": (
            "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
            "raw-step10000"
        ),
        "30k": (
            "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
            "raw-step30000"
        ),
    }
    raw_embeddings = {
        step: np.load(
            coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED),
            allow_pickle=False,
        )["embedding"]
        for step, checkpoint_id in raw_ids.items()
    }

    # Matched raw 10k/30k biochemical comparison in one figure.
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.3))
    for row, step in enumerate(("10k", "30k")):
        for column, (view, title) in enumerate(zip(views, titles)):
            ax = axes[row, column]
            scatter_panel(ax, raw_embeddings[step], aa, view=view, annotate_aa=False)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if column == 0:
                ax.set_ylabel(f"Raw-{step}", fontsize=12, weight="bold")
            categorical_legend(ax, view)
    fig.suptitle(
        "Raw-10k versus raw-30k: matched residue-chemistry organization",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.7)
    fig.savefig(RAW_COMPARISON, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.2))
    for row, step in enumerate(("10k", "30k")):
        for column, (weight_name, embedding) in enumerate((
            ("EMA", embeddings[f"EMA-{step}"]),
            ("Raw", raw_embeddings[step]),
        )):
            ax = axes[row, column]
            scatter_panel(ax, embedding, aa, view="aa20", annotate_aa=False)
            ax.set_title(f"{weight_name} — {step}", fontsize=12)
            if column == 0:
                ax.set_ylabel(f"Training step {step}", fontsize=11, weight="bold")
            categorical_legend(ax, "aa20", fontsize=6.2)
    fig.suptitle(
        "Raw versus EMA encoder: amino-acid organization",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.8)
    fig.savefig(RAW_EMA_COMPARISON, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ATLAS}")
    print(f"wrote {COMPARISON}")
    print(f"wrote {RAW_EMA_COMPARISON}")
    print(f"wrote {RAW_COMPARISON}")


if __name__ == "__main__":
    main()
