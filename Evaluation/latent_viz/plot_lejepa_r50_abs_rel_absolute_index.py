"""Compare 100k JEPA+SIGReg-r50 absolute and relative positions by chain index.

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
from run_tsne import load_or_fit  # noqa: E402
from thesis_plot_style import BLACK, apply_thesis_style  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = (
    "le-jepa-rand-mask-50",
    "le-jepa-r50-relpos-step100000",
)
TITLES = {
    "le-jepa-rand-mask-50": "JEPA+SIGReg-r50 absolute positions",
    "le-jepa-r50-relpos-step100000": "JEPA+SIGReg-r50 relative positions",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
COORD_DIR = HERE / "results" / "tsne_coordinates"
FILENAME = "tsne_lejepa-r50_absolute-v-relative_absolute-index_100k.png"
OUTPUTS = (
    HERE / "figures-extended" / "workshop" / "main" / FILENAME,
    REPO / "thesis-latex" / "figures" / "results" / FILENAME,
)


def coordinate_path(checkpoint_id: str) -> pathlib.Path:
    return COORD_DIR / f"{checkpoint_id}_n15000_perp30p0_seed42_cosine.npz"


def main() -> None:
    lm_family = apply_thesis_style()
    labels = load_labels()
    latent_paths = {
        checkpoint_id: LINEAR / "latents" / f"{checkpoint_id}.pt"
        for checkpoint_id in CHECKPOINTS
    }
    have_latents = all(path.exists() for path in latent_paths.values())
    per_protein = (
        {
            checkpoint_id: load_encoder_latents(checkpoint_id)
            for checkpoint_id in CHECKPOINTS
        }
        if have_latents else {}
    )
    protein_ids = common_protein_ids(per_protein) if have_latents else sorted(labels)
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
        # Reuse verified coordinates, or reconstruct a missing cache from the
        # same frozen latents, physical residue sample and historical settings.
        if have_latents:
            X = selection.build_X(per_protein[checkpoint_id])[sampled]
            embedding, _, _ = load_or_fit(
                checkpoint_id, X, perplexity=PERPLEXITY, seed=SEED,
                max_iter=1000, force=False,
            )
        else:
            cached = np.load(coordinate_path(checkpoint_id), allow_pickle=False)
            embedding = cached["embedding"]
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
                fontsize=7.5,
                color=BLACK,
                family=lm_family,
                weight=600,
                ha="center",
                va="center",
            )
            annotation.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground="white"),
                path_effects.Normal(),
            ])
        ax.set_title(TITLES[checkpoint_id], fontsize=19, color=BLACK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(right=0.91, top=0.92, wspace=0.04)
    colorbar_ax = fig.add_axes([0.93, 0.14, 0.014, 0.70])
    colorbar = fig.colorbar(
        scatter,
        cax=colorbar_ax,
        label="absolute residue index within chain",
    )
    colorbar.set_label(
        "absolute residue index within chain", fontsize=15, color=BLACK
    )
    colorbar.ax.tick_params(labelsize=12, colors=BLACK)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
