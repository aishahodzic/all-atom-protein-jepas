"""Compare absolute-position encoding with the I-JEPA/Meta-style predictor.

Uses existing frozen residue-latent and t-SNE coordinate caches only.  It
compares matched old/new-predictor random-mask runs and the additional 100k
mask-token-predictor experiments.  Metrics are reported both in the original
latent space (cosine k-NN, implemented as Euclidean k-NN after L2
normalization) and in the cached 2-D t-SNE map.
"""

import csv
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "knn_purity"))
sys.path.insert(0, str(HERE))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import (  # noqa: E402
    knn_exact_index_stats,
    local_index_array,
    representative_point,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = (
    "jepa-rand-mask-50",
    "jepa-new-predictor",
    "le-jepa-rand-mask-50",
    "le-jepa-new-predictor",
    "le-jepa-rand70-d64",
    "le-jepa-spatial3d-d64",
    "le-jepa-chem-exp1",
    "le-jepa-chem-exp3",
)

LABELS = {
    "jepa-rand-mask-50": "JEPA-50\nold cross-attention predictor",
    "jepa-new-predictor": "JEPA-50\nMeta mask-token predictor",
    "le-jepa-rand-mask-50": "LeJEPA-50\nold cross-attention predictor",
    "le-jepa-new-predictor": "LeJEPA NEW PREDICTOR + EMA (random-mask 50%)\nMeta mask-token predictor; EMA is a confound",
    "le-jepa-rand70-d64": "LeJEPA-70 D64\nold cross-attention predictor",
    "le-jepa-spatial3d-d64": "LeJEPA spatial D64\nMeta mask-token predictor",
    "le-jepa-chem-exp1": "LeJEPA chem exp1\nMeta mask-token predictor",
    "le-jepa-chem-exp3": "LeJEPA chem exp3\nMeta mask-token predictor",
}

PREDICTOR = {
    "jepa-rand-mask-50": "cross_attention",
    "jepa-new-predictor": "mask_token_transformer",
    "le-jepa-rand-mask-50": "cross_attention",
    "le-jepa-new-predictor": "mask_token_transformer",
    "le-jepa-rand70-d64": "cross_attention",
    "le-jepa-spatial3d-d64": "mask_token_transformer",
    "le-jepa-chem-exp1": "mask_token_transformer",
    "le-jepa-chem-exp3": "mask_token_transformer",
}

COORD_DIR = HERE / "results" / "tsne_coordinates"
RESULT_PATH = HERE / "results" / "meta_predictor_absolute_position_summary.csv"
FIG_PATH = HERE / "figures" / "tsne_absolute_index" / "meta_predictor_absolute_position_comparison.png"
N_POINTS = 15_000


def coordinate_path(checkpoint_id: str) -> pathlib.Path:
    return COORD_DIR / f"{checkpoint_id}_n15000_perp30p0_seed42_cosine.npz"


def normalized_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def run() -> None:
    labels = load_labels()
    per_protein = {checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in CHECKPOINTS}
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    residue_index = local_index_array(selection)[sampled]

    embeddings = {}
    rows = []
    for checkpoint_id in CHECKPOINTS:
        path = coordinate_path(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(path)
        embedding = np.load(path, allow_pickle=False)["embedding"]
        embeddings[checkpoint_id] = embedding

        latent = selection.build_X(per_protein[checkpoint_id])[sampled]
        latent_stats = knn_exact_index_stats(normalized_rows(latent), residue_index)
        tsne_stats = knn_exact_index_stats(embedding, residue_index)
        rows.append({
            "checkpoint": checkpoint_id,
            "predictor": PREDICTOR[checkpoint_id],
            "latent_dim": latent.shape[1],
            "sampled_residues": len(sampled),
            "sampled_proteins": len(np.unique(selection.protein_idx[sampled])),
            "original_latent_exact_index_purity_at_10": latent_stats["exact_index_purity_at_10"],
            "original_latent_nearest_neighbor_exact_index": latent_stats["nearest_neighbor_exact_index"],
            "tsne_exact_index_purity_at_10": tsne_stats["exact_index_purity_at_10"],
            "tsne_nearest_neighbor_exact_index": tsne_stats["nearest_neighbor_exact_index"],
            "tsne_points_with_at_least_8_of_10_same_index": tsne_stats[
                "points_with_at_least_8_of_10_same_index"
            ],
        })

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    one_based = residue_index + 1
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    scatter = None
    for ax, checkpoint_id, row in zip(axes.flat, CHECKPOINTS, rows):
        embedding = embeddings[checkpoint_id]
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=one_based, cmap="turbo",
            vmin=1, vmax=251, s=4, alpha=0.68, linewidths=0, rasterized=True,
        )
        # Label residue 1 and every twentieth residue in the compact grid.
        for index0 in np.unique(residue_index):
            index1 = int(index0) + 1
            if index1 != 1 and index1 % 20 != 0:
                continue
            mask = residue_index == index0
            if mask.sum() < 3:
                continue
            x, y = representative_point(embedding[mask])
            text = ax.text(x, y, str(index1), fontsize=5.5, weight="bold", ha="center", va="center")
            text.set_path_effects([
                path_effects.Stroke(linewidth=1.4, foreground="white"),
                path_effects.Normal(),
            ])
        ax.set_title(
            f"{LABELS[checkpoint_id]}\n"
            f"position purity@10: latent={row['original_latent_exact_index_purity_at_10']:.3f}, "
            f"t-SNE={row['tsne_exact_index_purity_at_10']:.3f}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Absolute-position encoding: old cross-attention versus Meta/I-JEPA-style mask-token predictor\n"
        "Same 15,000 physical residues in every panel; labels are one-based absolute chain indices",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.03, top=0.87, wspace=0.04, hspace=0.24)
    colorbar_ax = fig.add_axes([0.945, 0.13, 0.012, 0.68])
    fig.colorbar(scatter, cax=colorbar_ax, label="absolute residue index within chain (one-based)")
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {RESULT_PATH}")
    print(f"wrote {FIG_PATH}")
    for row in rows:
        print(
            f"{row['checkpoint']:24s} latent={row['original_latent_exact_index_purity_at_10']:.4f} "
            f"tsne={row['tsne_exact_index_purity_at_10']:.4f}"
        )


if __name__ == "__main__":
    run()
