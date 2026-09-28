"""Combine all A/B EMA-20k t-SNE colorings into one matched atlas."""

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
from run_tsne import coordinate_path, scatter_panel, view_labels_colors  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


MODELS = {
    "A: iterative EMA (20k)": (
        "canonical-lejepa-residue-iterative-protein-centered-sigreg-"
        "ema-step20000"
    ),
    "B: visible-only EMA (20k)": (
        "canonical-lejepa-residue-visible-only-protein-centered-sigreg-"
        "ema-step20000"
    ),
}
N_POINTS = 15_000
PERPLEXITY = 30.0
OUTPUT = HERE / "figures/canonical_lejepa_residue_ab_ema20k_tsne_atlas.png"


def clean_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def legend(ax, view: str) -> None:
    labels, colors = view_labels_colors(view)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=3.5, label=label)
        for label, color in zip(labels, colors)
    ]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22),
        ncol=5 if view == "aa20" else 1, fontsize=5.4, frameon=False,
        columnspacing=0.55, handletextpad=0.2,
    )


def secondary_structure(ax, embedding, values) -> None:
    for index, (name, color) in enumerate((
        ("Helix", "#d62728"), ("Sheet", "#1f77b4"), ("Loop", "#7f7f7f")
    )):
        members = values == index
        ax.scatter(
            embedding[members, 0], embedding[members, 1], s=2.2,
            alpha=0.58, linewidths=0, c=color, label=name, rasterized=True,
        )
    clean_axes(ax)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3,
        fontsize=6, frameon=False, markerscale=1.8,
    )


def continuous(ax, embedding, values, cmap, vmin, vmax, label) -> None:
    image = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=values, cmap=cmap,
        vmin=vmin, vmax=vmax, s=2.2, alpha=0.68, linewidths=0,
        rasterized=True,
    )
    clean_axes(ax)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.015)
    bar.set_label(label, fontsize=6.5)
    bar.ax.tick_params(labelsize=5.5)


def main() -> None:
    labels = load_labels()
    caches = {name: load_encoder_latents(checkpoint_id) for name, checkpoint_id in MODELS.items()}
    protein_ids = common_protein_ids(caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    aa = selection.aa[sampled]
    ss = selection.ss[sampled]
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
        for name, checkpoint_id in MODELS.items()
    }

    fig, axes = plt.subplots(4, 4, figsize=(16.2, 14.6))
    column_titles = (
        "Amino-acid identity", "Secondary structure", "Charge", "Aromaticity",
    )
    lower_titles = (
        "Side-chain size", "Absolute sequence index", "Normalized sequence position",
        "Source chain length",
    )
    for model_index, (model_name, embedding) in enumerate(embeddings.items()):
        top = model_index * 2
        bottom = top + 1
        scatter_panel(axes[top, 0], embedding, aa, view="aa20", annotate_aa=False)
        legend(axes[top, 0], "aa20")
        secondary_structure(axes[top, 1], embedding, ss)
        for column, view in ((2, "charge"), (3, "aromatic")):
            scatter_panel(axes[top, column], embedding, aa, view=view, annotate_aa=False)
            legend(axes[top, column], view)
        scatter_panel(axes[bottom, 0], embedding, aa, view="size", annotate_aa=False)
        legend(axes[bottom, 0], "size")
        continuous(
            axes[bottom, 1], embedding, residue_index + 1, "turbo", 1, 256,
            "residue index i",
        )
        continuous(
            axes[bottom, 2], embedding, normalized_position, "viridis", 0, 1,
            "i/L",
        )
        continuous(
            axes[bottom, 3], embedding, lengths, "magma", 50, 256,
            "chain length L",
        )
        for column, title in enumerate(column_titles):
            axes[top, column].set_title(title, fontsize=10)
        for column, title in enumerate(lower_titles):
            axes[bottom, column].set_title(title, fontsize=10)
        axes[top, 0].text(
            -0.14, 0.5, model_name, transform=axes[top, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12, weight="bold",
        )

    fig.suptitle(
        "Canonical residue LeJEPA A/B — matched EMA-20k t-SNE atlas",
        fontsize=17,
    )
    fig.tight_layout(rect=(0.025, 0, 1, 0.965), h_pad=2.9, w_pad=1.25)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
