"""Matched JEPA-D64 rotation comparison with amino-acid labels on clusters."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
sys.path[:0] = [str(REPO), str(LINEAR), str(KNN), str(HERE)]

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from run_tsne import (  # noqa: E402
    annotate_aa_locations,
    coordinate_path,
    scatter_panel,
    view_labels_colors,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = (
    "jepa-d64-10k-baseline",
    "jepa-d64-10k-rot",
)
TITLES = {
    "jepa-d64-10k-baseline": "JEPA-r70 D64 (no rotations), 10k",
    "jepa-d64-10k-rot": "JEPA-r70 D64 (+ independent rotations), 10k",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
OUTPUT = HERE / "figures" / "workshop" / (
    "tsne_aa20_jepa-r70-d64-no-rotation-v-rotation_labels-on-clusters.png"
)


def main() -> None:
    labels = load_labels()
    latent_caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in CHECKPOINTS
    }
    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    aa = selection.aa[sampled]
    aa_labels, colors = view_labels_colors("aa20")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.65), squeeze=False)
    axes = axes.ravel()
    for ax, checkpoint_id in zip(axes, CHECKPOINTS):
        path = coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED)
        if not path.is_file():
            raise FileNotFoundError(path)
        embedding = np.load(path, allow_pickle=False)["embedding"]
        scatter_panel(ax, embedding, aa, view="aa20", annotate_aa=False)
        annotate_aa_locations(ax, embedding, aa, ["black"] * len(aa_labels))
        ax.set_title(TITLES[checkpoint_id], fontsize=9)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=color,
            markersize=6, label=label,
        )
        for label, color in zip(aa_labels, colors)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=10,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("t-SNE of per-residue latents — amino-acid identity", fontsize=13)
    fig.tight_layout(rect=[0, 0.075, 1, 0.95])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
