"""Label the LeJEPA t-SNE "worms" by absolute residue index.

This reuses the exact cached 15,000-point t-SNE coordinates produced by
``run_tsne.py``.  No encoder forward and no t-SNE refit is performed.  The
same seeded residue sample is reconstructed, and each point is assigned its
index within its source chain.  Labels in the figures are one-based (residue
1 is the first residue); the machine-readable CSV includes both conventions.

The local-purity statistic asks: among each plotted point's ten nearest
neighbours in the 2-D t-SNE map, what fraction have the exact same absolute
residue index?  High purity is direct quantitative evidence that a visible
strand is an absolute-position strand, rather than one protein or one amino
acid class.
"""

import argparse
import csv
import pathlib
import sys
from typing import Dict, Iterable, Tuple

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
KNN_DIR = REPO_ROOT / "Evaluation" / "knn_purity"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LINEAR_PROBING_DIR))
sys.path.insert(0, str(KNN_DIR))
sys.path.insert(0, str(HERE))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = (
    "le-jepa-mutation",
    "le-jepa-rand-mask-50",
    "le-jepa-rand-mask-70",
)
SHORT_NAMES = {
    "le-jepa-mutation": "LeJEPA mutation",
    "le-jepa-rand-mask-50": "LeJEPA random masking 50%",
    "le-jepa-rand-mask-70": "LeJEPA random masking 70%",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
K = 10
COORD_DIR = HERE / "results" / "tsne_coordinates"
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures" / "tsne_absolute_index"


def coordinate_path(checkpoint_id: str) -> pathlib.Path:
    return COORD_DIR / f"{checkpoint_id}_n15000_perp30p0_seed42_cosine.npz"


def local_index_array(selection: CombinedSelection) -> np.ndarray:
    """Absolute zero-based chain index aligned with CombinedSelection rows."""
    return np.concatenate([idx for _, idx in selection.keys]).astype(np.int64)


def protein_id_array(selection: CombinedSelection, protein_ids) -> np.ndarray:
    """Source protein string aligned with CombinedSelection rows."""
    chunks = []
    for pid, idx in selection.keys:
        chunks.append(np.repeat(pid, len(idx)))
    return np.concatenate(chunks)


def knn_exact_index_stats(embedding: np.ndarray, residue_index: np.ndarray, k: int = K) -> Dict[str, float]:
    # Query k+1 because the closest point is the query itself.
    _, neighbours = cKDTree(embedding).query(embedding, k=k + 1, workers=-1)
    neighbours = neighbours[:, 1:]
    matches = residue_index[neighbours] == residue_index[:, None]
    per_point = matches.mean(axis=1)
    return {
        "exact_index_purity_at_10": float(per_point.mean()),
        "nearest_neighbor_exact_index": float(matches[:, 0].mean()),
        "points_with_at_least_8_of_10_same_index": float((matches.sum(axis=1) >= 8).mean()),
    }


def representative_point(embedding: np.ndarray) -> Tuple[float, float]:
    """A real point nearest the component-wise median, robust to curved tails."""
    center = np.median(embedding, axis=0)
    chosen = np.argmin(np.square(embedding - center).sum(axis=1))
    return float(embedding[chosen, 0]), float(embedding[chosen, 1])


def render_index_atlas(
    checkpoint_id: str,
    embedding: np.ndarray,
    residue_index: np.ndarray,
    counts: Dict[int, int],
) -> pathlib.Path:
    """Large, all-index atlas: every sufficiently represented strand is labeled."""
    fig, ax = plt.subplots(figsize=(20, 17))
    one_based = residue_index + 1
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=one_based, cmap="turbo",
        s=7, alpha=0.72, linewidths=0, rasterized=True,
    )

    for index0 in sorted(counts):
        mask = residue_index == index0
        if mask.sum() < 3:
            continue
        x, y = representative_point(embedding[mask])
        label = ax.text(
            x, y, str(index0 + 1), ha="center", va="center", fontsize=5.0,
            color="black", weight="bold", zorder=4,
        )
        label.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground="white"),
            path_effects.Normal(),
        ])

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("absolute residue index within chain (one-based)")
    ax.set_title(
        f"{SHORT_NAMES[checkpoint_id]}: exact absolute-index atlas\n"
        "Each number labels the representative point of that residue-index strand",
        fontsize=15,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{checkpoint_id}_absolute_residue_index_all_labeled.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def render_comparison(
    embeddings: Dict[str, np.ndarray], residue_index: np.ndarray, every: int = 10
) -> pathlib.Path:
    """Compact three-model view, labeling every Nth one-based residue index."""
    fig, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(18, 6.4))
    one_based = residue_index + 1
    scatter = None
    for ax, checkpoint_id in zip(axes, CHECKPOINTS):
        embedding = embeddings[checkpoint_id]
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=one_based, cmap="turbo",
            s=4, alpha=0.65, linewidths=0, rasterized=True,
        )
        for index0 in np.unique(residue_index):
            # Show residue 1 and then 10,20,... in this compact overview.
            index1 = int(index0) + 1
            if index1 != 1 and index1 % every != 0:
                continue
            mask = residue_index == index0
            if mask.sum() < 3:
                continue
            x, y = representative_point(embedding[mask])
            label = ax.text(x, y, str(index1), fontsize=6, weight="bold", ha="center", va="center")
            label.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground="white"),
                path_effects.Normal(),
            ])
        ax.set_title(SHORT_NAMES[checkpoint_id])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        "LeJEPA t-SNE worms encode absolute chain index\n"
        f"Numbers are one-based residue indices (compact view labels residue 1 and every {every}th residue)",
        fontsize=14,
    )
    fig.subplots_adjust(right=0.92, top=0.86, wspace=0.03)
    colorbar_ax = fig.add_axes([0.935, 0.15, 0.012, 0.65])
    fig.colorbar(scatter, cax=colorbar_ax, label="absolute residue index (one-based)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "lejepa_absolute_residue_index_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def run(checkpoints: Iterable[str] = CHECKPOINTS) -> None:
    checkpoints = tuple(checkpoints)
    if checkpoints != CHECKPOINTS:
        raise ValueError(f"This locked comparison expects {CHECKPOINTS}, got {checkpoints}")

    labels = load_labels()
    per_protein = {checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in checkpoints}
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    residue_index = local_index_array(selection)[sampled]
    source_protein = protein_id_array(selection, protein_ids)[sampled]

    embeddings: Dict[str, np.ndarray] = {}
    summary_rows = []
    center_rows = []
    for checkpoint_id in checkpoints:
        coord_path = coordinate_path(checkpoint_id)
        if not coord_path.exists():
            raise FileNotFoundError(f"Missing locked t-SNE coordinates: {coord_path}")
        embedding = np.load(coord_path, allow_pickle=False)["embedding"]
        if len(embedding) != len(sampled):
            raise ValueError(f"{checkpoint_id}: {len(embedding)} coordinates != {len(sampled)} sampled residues")
        embeddings[checkpoint_id] = embedding

        counts = {int(i): int((residue_index == i).sum()) for i in np.unique(residue_index)}
        stats = knn_exact_index_stats(embedding, residue_index, k=K)
        summary_rows.append({
            "checkpoint": checkpoint_id,
            "sampled_residues": len(sampled),
            "sampled_proteins": len(np.unique(source_protein)),
            "observed_absolute_indices": len(counts),
            "minimum_index_zero_based": int(residue_index.min()),
            "maximum_index_zero_based": int(residue_index.max()),
            **stats,
        })
        for index0, count in counts.items():
            mask = residue_index == index0
            x, y = representative_point(embedding[mask])
            center_rows.append({
                "checkpoint": checkpoint_id,
                "residue_index_zero_based": index0,
                "residue_number_one_based": index0 + 1,
                "sampled_points": count,
                "representative_tsne_x": x,
                "representative_tsne_y": y,
            })
        path = render_index_atlas(checkpoint_id, embedding, residue_index, counts)
        print(f"wrote {path}")

    comparison_path = render_comparison(embeddings, residue_index)
    print(f"wrote {comparison_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "tsne_worm_absolute_index_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    centers_path = RESULTS_DIR / "tsne_worm_absolute_index_centers.csv"
    with open(centers_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(center_rows[0]))
        writer.writeheader()
        writer.writerows(center_rows)
    print(f"wrote {summary_path}")
    print(f"wrote {centers_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
