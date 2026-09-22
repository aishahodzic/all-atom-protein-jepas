"""Show whether LeJEPA-rand50 absolute-index worms mix source proteins.

This diagnostic reuses the locked 15,000-point t-SNE coordinates and residue
sample.  It performs no encoder forward and does not refit t-SNE.
"""

import csv
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_hex
import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "knn_purity"))
sys.path.insert(0, str(HERE))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array, representative_point  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT = "le-jepa-rand-mask-50"
N_POINTS = 15_000
K = 10
COORD_PATH = (
    HERE / "results" / "tsne_coordinates" /
    f"{CHECKPOINT}_n15000_perp30p0_seed42_cosine.npz"
)
FIG_PATH = (
    HERE / "figures" / "tsne_absolute_index" /
    "le-jepa-rand-mask-50_protein_colored_index_labeled.png"
)
POINTS_PATH = (
    HERE / "results" / "le-jepa-rand-mask-50_tsne_point_provenance.csv"
)
SUMMARY_PATH = (
    HERE / "results" / "le-jepa-rand-mask-50_protein_worm_summary.csv"
)


def stable_protein_colors(n_proteins: int, seed: int = SEED) -> np.ndarray:
    """Deterministic pseudo-random categorical colors, one per source protein."""
    rng = np.random.default_rng(seed)
    hues = (np.arange(n_proteins, dtype=np.float64) * 0.618033988749895) % 1.0
    rng.shuffle(hues)
    saturation = 0.62 + 0.30 * rng.random(n_proteins)
    value = 0.66 + 0.30 * rng.random(n_proteins)
    return hsv_to_rgb(np.column_stack((hues, saturation, value)))


def label_index_centres(ax, embedding: np.ndarray, residue_index: np.ndarray) -> None:
    for index0 in np.unique(residue_index):
        mask = residue_index == index0
        if mask.sum() < 3:
            continue
        x, y = representative_point(embedding[mask])
        label = ax.text(
            x, y, str(int(index0) + 1), ha="center", va="center",
            fontsize=4.8, color="black", weight="bold", zorder=5,
        )
        label.set_path_effects([
            path_effects.Stroke(linewidth=1.35, foreground="white"),
            path_effects.Normal(),
        ])


def neighbour_statistics(
    embedding: np.ndarray, residue_index: np.ndarray, protein_index: np.ndarray
) -> dict:
    _, neighbours = cKDTree(embedding).query(embedding, k=K + 1, workers=-1)
    neighbours = neighbours[:, 1:]
    exact_position = residue_index[neighbours] == residue_index[:, None]
    same_protein = protein_index[neighbours] == protein_index[:, None]
    return {
        "position_purity_at_10": float(exact_position.mean()),
        "same_protein_fraction_at_10": float(same_protein.mean()),
        "nearest_neighbour_same_position": float(exact_position[:, 0].mean()),
        "nearest_neighbour_same_protein": float(same_protein[:, 0].mean()),
    }


def run() -> None:
    labels = load_labels()
    per_protein = {CHECKPOINT: load_encoder_latents(CHECKPOINT)}
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)

    embedding = np.load(COORD_PATH, allow_pickle=False)["embedding"]
    if len(embedding) != len(sampled):
        raise ValueError(f"{len(embedding)} coordinates != {len(sampled)} sampled residues")

    residue_index = local_index_array(selection)[sampled]
    protein_index = selection.protein_idx[sampled]
    source_protein = np.asarray(protein_ids, dtype=object)[protein_index]
    palette = stable_protein_colors(len(protein_ids))
    point_colors = palette[protein_index]
    stats = neighbour_statistics(embedding, residue_index, protein_index)

    observed_proteins, observed_counts = np.unique(protein_index, return_counts=True)
    top_order = np.argsort(observed_counts)[::-1][:14]
    highlighted = observed_proteins[top_order]
    highlighted_set = set(highlighted.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(26, 12.5))

    axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=point_colors,
        s=8, alpha=0.78, linewidths=0, rasterized=True,
    )
    label_index_centres(axes[0], embedding, residue_index)
    axes[0].set_title(
        "All 15,000 residues colored by source protein\n"
        "Numbers are one-based absolute residue indices",
        fontsize=14,
    )

    axes[1].scatter(
        embedding[:, 0], embedding[:, 1], c="#d8d8d8",
        s=5, alpha=0.20, linewidths=0, rasterized=True,
    )
    for p_int in highlighted:
        mask = protein_index == p_int
        order = np.argsort(residue_index[mask])
        coords = embedding[mask][order]
        indices = residue_index[mask][order] + 1
        color = palette[p_int]
        axes[1].scatter(
            coords[:, 0], coords[:, 1], color=color, s=30, alpha=0.95,
            edgecolors="white", linewidths=0.35,
            label=f"{protein_ids[p_int]} ({len(coords)} residues)", zorder=4,
        )
        for (x, y), index1 in zip(coords, indices):
            text = axes[1].text(
                x, y, str(int(index1)), fontsize=5.5, color="black",
                ha="center", va="center", weight="bold", zorder=5,
            )
            text.set_path_effects([
                path_effects.Stroke(linewidth=1.25, foreground="white"),
                path_effects.Normal(),
            ])
    axes[1].set_title(
        "Fourteen most-sampled proteins highlighted\n"
        "Labels beside their dots are those residues' absolute indices",
        fontsize=14,
    )
    axes[1].legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        fontsize=8, markerscale=0.8,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "LeJEPA random masking 50%: absolute-position worms are mixtures of proteins\n"
        f"position purity@{K}={stats['position_purity_at_10']:.3f}; "
        f"same-protein neighbours@{K}="
        f"{100 * stats['same_protein_fraction_at_10']:.4f}%",
        fontsize=17,
    )
    fig.subplots_adjust(left=0.015, right=0.82, bottom=0.02, top=0.91, wspace=0.025)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POINTS_PATH, "w", newline="") as handle:
        fields = [
            "sample_row", "protein_id", "protein_index", "absolute_index_zero_based",
            "residue_number_one_based", "tsne_x", "tsne_y", "protein_color_hex",
            "highlighted_in_right_panel",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in range(len(sampled)):
            p_int = int(protein_index[row])
            writer.writerow({
                "sample_row": row,
                "protein_id": source_protein[row],
                "protein_index": p_int,
                "absolute_index_zero_based": int(residue_index[row]),
                "residue_number_one_based": int(residue_index[row]) + 1,
                "tsne_x": float(embedding[row, 0]),
                "tsne_y": float(embedding[row, 1]),
                "protein_color_hex": to_hex(palette[p_int]),
                "highlighted_in_right_panel": p_int in highlighted_set,
            })

    position_counts = []
    for index0 in np.unique(residue_index):
        mask = residue_index == index0
        position_counts.append({
            "absolute_index_zero_based": int(index0),
            "residue_number_one_based": int(index0) + 1,
            "sampled_points": int(mask.sum()),
            "unique_source_proteins": int(np.unique(protein_index[mask]).size),
        })

    with open(SUMMARY_PATH, "w", newline="") as handle:
        fields = list(position_counts[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(position_counts)

    print(f"wrote {FIG_PATH}")
    print(f"wrote {POINTS_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"sampled proteins: {np.unique(protein_index).size}")
    for name, value in stats.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    run()
