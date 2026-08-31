"""Matched held-out analysis of the LeJEPA relative-position intervention.

The final online/student encoder is extracted on the existing frozen latent
pool.  The script then fits the standard 15k-point cosine t-SNE and compares
absolute-index k-NN purity against the original LeJEPA rand50 checkpoint on
the exact same proteins and physical residue positions.
"""

import csv
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import (  # noqa: E402
    knn_exact_index_stats,
    local_index_array,
    representative_point,
)
from run_tsne import coordinate_path, run as run_tsne  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CONTROL = "le-jepa-rand-mask-50"
INTERVENTION = "le-jepa-rand50-relative-pos-10k"
_CHECKPOINT_SUFFIX = pathlib.Path(
    "aishod/runs/le_jepa_rand50_relative_positions_10k/"
    "store/le_jepa_rand50_relative_positions_10k/checkpoints/"
    "chk_epoch=00000001_step=000000010000.ckpt"
)
CHECKPOINT_CANDIDATES = [
    pathlib.Path("/mnt/hopper-raid") / _CHECKPOINT_SUFFIX,
    pathlib.Path("/mnt/raid") / _CHECKPOINT_SUFFIX,
]
CHECKPOINT = next(
    (candidate for candidate in CHECKPOINT_CANDIDATES if candidate.exists()),
    CHECKPOINT_CANDIDATES[0],
)
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
RESULT = HERE / "results" / "lejepa_relative_position_shortcut_summary.csv"
FIGURE = (
    HERE / "figures" / "tsne_absolute_index" /
    "lejepa_relative_position_shortcut_comparison.png"
)


def register_intervention() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    entry = (str(CHECKPOINT), "jepa")
    checkpoints.CHECKPOINTS[INTERVENTION] = entry
    checkpoints.ALL_CHECKPOINTS[INTERVENTION] = entry


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def same_protein_neighbour_fraction(
    values: np.ndarray, protein_index: np.ndarray, normalize: bool
) -> float:
    values = normalized_rows(values) if normalize else np.asarray(values)
    _, neighbours = cKDTree(values).query(values, k=11, workers=-1)
    neighbours = neighbours[:, 1:]
    return float((protein_index[neighbours] == protein_index[:, None]).mean())


def label_indices(ax, embedding: np.ndarray, residue_index: np.ndarray) -> None:
    for index0 in np.unique(residue_index):
        index1 = int(index0) + 1
        if index1 != 1 and index1 % 20 != 0:
            continue
        mask = residue_index == index0
        if mask.sum() < 3:
            continue
        x, y = representative_point(embedding[mask])
        text = ax.text(
            x, y, str(index1), fontsize=6, weight="bold",
            ha="center", va="center",
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=1.4, foreground="white"),
            path_effects.Normal(),
        ])


def analyze() -> None:
    register_intervention()

    # Resume-safe: extract() reuses the cache if it already exists.
    extract(checkpoint_ids=[INTERVENTION], device="cuda")
    run_tsne(
        checkpoint_ids=[CONTROL, INTERVENTION],
        n_points=N_POINTS,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
        output_tag="lejepa-relative-position",
    )

    labels = load_labels()
    per_protein = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in (CONTROL, INTERVENTION)
    }
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    residue_index = local_index_array(selection)[sampled]
    protein_index = selection.protein_idx[sampled]
    protein_counts = np.bincount(protein_index)
    random_same_protein = float(
        (protein_counts * (protein_counts - 1)).sum() /
        (len(protein_index) * (len(protein_index) - 1))
    )

    embeddings = {}
    rows = []
    for checkpoint_id in (CONTROL, INTERVENTION):
        latent = selection.build_X(per_protein[checkpoint_id])[sampled]
        coord = coordinate_path(
            checkpoint_id, N_POINTS, perplexity=PERPLEXITY, seed=SEED
        )
        embedding = np.load(coord, allow_pickle=False)["embedding"]
        embeddings[checkpoint_id] = embedding
        latent_stats = knn_exact_index_stats(
            normalized_rows(latent), residue_index
        )
        tsne_stats = knn_exact_index_stats(embedding, residue_index)
        rows.append({
            "checkpoint": checkpoint_id,
            "checkpoint_path": str(
                CHECKPOINT if checkpoint_id == INTERVENTION else
                checkpoints.CKPT_ROOT / checkpoints.CHECKPOINTS[CONTROL][0]
            ),
            "sampled_proteins": len(protein_ids),
            "sampled_residues": len(sampled),
            "latent_dim": latent.shape[1],
            "random_pair_same_protein_probability": random_same_protein,
            "original_latent_exact_index_purity_at_10": latent_stats[
                "exact_index_purity_at_10"
            ],
            "original_latent_nearest_neighbor_exact_index": latent_stats[
                "nearest_neighbor_exact_index"
            ],
            "original_latent_same_protein_fraction_at_10":
                same_protein_neighbour_fraction(
                    latent, protein_index, normalize=True
                ),
            "tsne_exact_index_purity_at_10": tsne_stats[
                "exact_index_purity_at_10"
            ],
            "tsne_nearest_neighbor_exact_index": tsne_stats[
                "nearest_neighbor_exact_index"
            ],
            "tsne_same_protein_fraction_at_10":
                same_protein_neighbour_fraction(
                    embedding, protein_index, normalize=False
                ),
            "tsne_points_with_at_least_8_of_10_same_index": tsne_stats[
                "points_with_at_least_8_of_10_same_index"
            ],
        })

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels_by_checkpoint = {
        CONTROL: "Original LeJEPA rand50 (100k)",
        INTERVENTION: "Relative-position LeJEPA rand50 (10k)",
    }
    one_based = residue_index + 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    scatter = None
    for ax, row in zip(axes, rows):
        checkpoint_id = row["checkpoint"]
        embedding = embeddings[checkpoint_id]
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=one_based, cmap="turbo",
            vmin=1, vmax=251, s=5, alpha=0.70, linewidths=0,
            rasterized=True,
        )
        label_indices(ax, embedding, residue_index)
        ax.set_title(
            f"{labels_by_checkpoint[checkpoint_id]}\n"
            "absolute-index purity@10: "
            f"latent={row['original_latent_exact_index_purity_at_10']:.3f}, "
            f"t-SNE={row['tsne_exact_index_purity_at_10']:.3f}",
            fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Does relative predictor positioning remove LeJEPA's absolute-index shortcut?\n"
        "Same held-out proteins and same 15,000 physical residues; numbers are one-based chain indices",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.03, top=0.84, wspace=0.04)
    cax = fig.add_axes([0.93, 0.14, 0.015, 0.62])
    fig.colorbar(scatter, cax=cax, label="absolute residue index within chain")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {RESULT}")
    print(f"wrote {FIGURE}")
    for row in rows:
        print(
            f"{row['checkpoint']}: "
            f"latent position purity@10={row['original_latent_exact_index_purity_at_10']:.6f}; "
            f"t-SNE position purity@10={row['tsne_exact_index_purity_at_10']:.6f}"
        )


if __name__ == "__main__":
    analyze()
