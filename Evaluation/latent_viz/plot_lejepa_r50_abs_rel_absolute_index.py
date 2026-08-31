"""Compare 100k LeJEPA-r50 absolute and relative positions by chain index.

The plot reuses the locked 15,000-residue sample and cached t-SNE coordinates
from the matched amino-acid panel.  It only changes the coloring and labels:
points are colored by their one-based absolute residue index within each chain.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
sys.path[:0] = [str(REPO), str(LINEAR), str(KNN), str(HERE)]

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array, representative_point  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = (
    "le-jepa-rand-mask-50",
    "le-jepa-r50-relpos-step100000",
)
TITLES = {
    "le-jepa-rand-mask-50": "LeJEPA-r50 absolute positions (100k)",
    "le-jepa-r50-relpos-step100000": "LeJEPA-r50 relative positions (100k)",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
COORD_DIR = HERE / "results" / "tsne_coordinates"
OUTPUT = HERE / "figures" / "workshop" / (
    "tsne_lejepa-r50_absolute-v-relative_absolute-index_100k.png"
)


def coordinate_path(checkpoint_id: str) -> pathlib.Path:
    return COORD_DIR / f"{checkpoint_id}_n15000_perp30p0_seed42_cosine.npz"


def main() -> None:
    labels = load_labels()
    per_protein = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in CHECKPOINTS
    }
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    one_based_index = local_index_array(selection)[sampled] + 1

    norm = Normalize(
        vmin=float(one_based_index.min()),
        vmax=float(one_based_index.max()),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.3))
    scatter = None
    for ax, checkpoint_id in zip(axes, CHECKPOINTS):
        path = coordinate_path(checkpoint_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        embedding = np.load(path, allow_pickle=False)["embedding"]
        if len(embedding) != len(sampled):
            raise ValueError(
                f"{checkpoint_id}: {len(embedding)} coordinates != "
                f"{len(sampled)} sampled residues"
            )

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=one_based_index,
            cmap="turbo",
            norm=norm,
            s=5,
            alpha=0.68,
            linewidths=0,
            rasterized=True,
        )
        for index1 in np.unique(one_based_index):
            if index1 != 1 and index1 % 10 != 0:
                continue
            mask = one_based_index == index1
            if mask.sum() < 3:
                continue
            x, y = representative_point(embedding[mask])
            annotation = ax.text(
                x,
                y,
                str(int(index1)),
                fontsize=6,
                weight="bold",
                ha="center",
                va="center",
            )
            annotation.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground="white"),
                path_effects.Normal(),
            ])
        ax.set_title(TITLES[checkpoint_id], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(right=0.91, top=0.92, wspace=0.04)
    colorbar_ax = fig.add_axes([0.93, 0.14, 0.014, 0.70])
    fig.colorbar(
        scatter,
        cax=colorbar_ax,
        label="absolute residue index within chain",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
