"""Stack the workshop position and rotation t-SNE comparisons.

Panel a compares LeJEPA absolute versus relative predictor positions, colored
by one-based chain index. Panel b compares JEPA-r70 D64 without versus with
independent target rotations, colored by amino-acid identity. The plot reuses
the locked 15,000-residue samples and existing t-SNE coordinate caches.
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
from run_tsne import (  # noqa: E402
    annotate_aa_locations,
    coordinate_path,
    scatter_panel,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


POSITION_CHECKPOINTS = (
    "le-jepa-rand-mask-50",
    "le-jepa-r50-relpos-step100000",
)
POSITION_TITLES = {
    "le-jepa-rand-mask-50": "LeJEPA-r50 absolute positions (100k)",
    "le-jepa-r50-relpos-step100000": "LeJEPA-r50 relative positions (100k)",
}
ROTATION_CHECKPOINTS = (
    "jepa-r70-d64-step60000",
    "jepa-r70-d64-rot-step60000",
)
ROTATION_TITLES = {
    "jepa-r70-d64-step60000": "JEPA-r70 D64 (60k)",
    "jepa-r70-d64-rot-step60000": "JEPA-r70 D64 with rotations (60k)",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
POSITION_COORD_DIR = HERE / "results" / "tsne_coordinates"
OUTPUT = HERE / "figures" / "workshop" / "main" / (
    "tsne_lejepa-positions_and_jepa-rotations_stacked.png"
)


def position_coordinate_path(checkpoint_id: str) -> pathlib.Path:
    return POSITION_COORD_DIR / (
        f"{checkpoint_id}_n15000_perp30p0_seed42_cosine.npz"
    )


def locked_selection(checkpoint_ids, labels):
    caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in checkpoint_ids
    }
    protein_ids = common_protein_ids(caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    return selection, sampled


def main() -> None:
    labels = load_labels()
    position_selection, position_sampled = locked_selection(
        POSITION_CHECKPOINTS, labels
    )
    one_based_index = local_index_array(position_selection)[position_sampled] + 1
    rotation_selection, rotation_sampled = locked_selection(
        ROTATION_CHECKPOINTS, labels
    )
    aa = rotation_selection.aa[rotation_sampled]

    norm = Normalize(
        vmin=float(one_based_index.min()),
        vmax=float(one_based_index.max()),
    )
    fig = plt.figure(figsize=(18.8, 4.7))
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=(1, 1, 0.22, 1, 1),
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.91,
        wspace=0.13,
    )
    position_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    spacer_ax = fig.add_subplot(grid[0, 2])
    spacer_ax.axis("off")
    rotation_axes = (fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[0, 4]))

    scatter = None
    for ax, checkpoint_id in zip(position_axes, POSITION_CHECKPOINTS):
        path = position_coordinate_path(checkpoint_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        embedding = np.load(path, allow_pickle=False)["embedding"]
        if len(embedding) != len(position_sampled):
            raise ValueError(
                f"{checkpoint_id}: {len(embedding)} coordinates != "
                f"{len(position_sampled)} sampled residues"
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
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar_ax = spacer_ax.inset_axes([0.03, 0.13, 0.16, 0.72])
    fig.colorbar(
        scatter,
        cax=colorbar_ax,
        label="absolute residue index within chain",
    )

    for ax, checkpoint_id in zip(rotation_axes, ROTATION_CHECKPOINTS):
        path = coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED)
        if not path.is_file():
            raise FileNotFoundError(path)
        embedding = np.load(path, allow_pickle=False)["embedding"]
        if len(embedding) != len(rotation_sampled):
            raise ValueError(
                f"{checkpoint_id}: {len(embedding)} coordinates != "
                f"{len(rotation_sampled)} sampled residues"
            )
        scatter_panel(ax, embedding, aa, view="aa20", annotate_aa=False)
        for collection in ax.collections:
            collection.set_alpha(0.92)
            collection.set_sizes([4.6])
        if checkpoint_id == "jepa-r70-d64-rot-step60000":
            annotate_aa_locations(ax, embedding, aa, ["black"] * 20)
        ax.set_aspect("equal", adjustable="box")

    title_y = 0.925
    for ax, checkpoint_id in zip(position_axes, POSITION_CHECKPOINTS):
        bounds = ax.get_position()
        fig.text(
            (bounds.x0 + bounds.x1) / 2,
            title_y,
            POSITION_TITLES[checkpoint_id],
            ha="center",
            va="bottom",
            fontsize=12,
        )
    for ax, checkpoint_id in zip(rotation_axes, ROTATION_CHECKPOINTS):
        bounds = ax.get_position()
        fig.text(
            (bounds.x0 + bounds.x1) / 2,
            title_y,
            ROTATION_TITLES[checkpoint_id],
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Subfigure labels mark the start of each horizontal two-panel group.
    fig.text(0.006, 0.965, "a)", fontsize=17, weight="bold", va="top")
    rotation_left = rotation_axes[0].get_position().x0
    fig.text(
        rotation_left - 0.025,
        0.965,
        "b)",
        fontsize=17,
        weight="bold",
        va="top",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
