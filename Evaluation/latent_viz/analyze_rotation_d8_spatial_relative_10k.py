"""Matched 10k latent audit for D8 rotation-JEPA and spatial-relative LeJEPA."""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
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
from run_tsne import (  # noqa: E402
    coordinate_path,
    run as run_tsne,
    scatter_panel,
    view_labels_colors,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


RUN_ROOT = pathlib.Path("/mnt/hopper-raid/aishod/runs")
NEW_CHECKPOINTS = {
    "jepa-r70-d8-rot-10k": RUN_ROOT / (
        "jepa_rand70_d8_independent_rotations_100k/store/"
        "jepa_rand70_d8_independent_rotations_100k/checkpoints/"
        "chk_epoch=00000001_step=000000010000.ckpt"
    ),
    "le-jepa-sp3d-relpos-d8-10k": RUN_ROOT / (
        "le_jepa_spatial3d_relative_positions_d8_100k/store/"
        "le_jepa_spatial3d_relative_positions_d8_100k/checkpoints/"
        "chk_epoch=00000001_step=000000010000.ckpt"
    ),
}
REFERENCE_CHECKPOINTS = {
    "jepa-d64-10k-rot": RUN_ROOT / (
        "jepa_rand70_d64_independent_rotations_10k/store/"
        "jepa_rand70_d64_independent_rotations_10k/checkpoints/"
        "chk_epoch=00000001_step=000000010000.ckpt"
    ),
    "le-jepa-rand50-relative-pos-10k": RUN_ROOT / (
        "le_jepa_rand50_relative_positions_10k/store/"
        "le_jepa_rand50_relative_positions_10k/checkpoints/"
        "chk_epoch=00000001_step=000000010000.ckpt"
    ),
}
IDS = (
    "ca-cond",
    "jepa-d64-10k-rot",
    "jepa-r70-d8-rot-10k",
    "le-jepa-rand50-relative-pos-10k",
    "le-jepa-sp3d-relpos-d8-10k",
)
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
K = 10
RESULT = HERE / "results/rotation_d8_spatial_relative_10k.csv"
FIGURE = HERE / "figures/tsne_10k/rotation_d8_spatial_relative_10k.png"

NAMES = {
    "ca-cond": "CA-cond AE (100k reference)",
    "jepa-d64-10k-rot": "JEPA rand70 D64 + rotations (10k)",
    "jepa-r70-d8-rot-10k": "JEPA rand70 D8 + rotations (10k)",
    "le-jepa-rand50-relative-pos-10k": "LeJEPA rand50 D8 + relative pos. (10k)",
    "le-jepa-sp3d-relpos-d8-10k": "LeJEPA spatial-3D D8 + relative pos. (10k)",
}


def register() -> None:
    for checkpoint_id, path in REFERENCE_CHECKPOINTS.items():
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry
    for checkpoint_id, path in NEW_CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values.astype(np.float64), rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / np.maximum(eigenvalues.sum(), 1e-12)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def external_neighbours(values: np.ndarray, protein_index: np.ndarray, k: int = K) -> np.ndarray:
    values = normalized(values)
    _, candidates = cKDTree(values).query(values, k=64, workers=-1)
    result = np.empty((len(values), k), dtype=np.int64)
    for row, candidate_row in enumerate(candidates):
        external = candidate_row[protein_index[candidate_row] != protein_index[row]]
        if len(external) < k:
            raise RuntimeError("Neighbour buffer was insufficient")
        result[row] = external[:k]
    return result


def purity(labels: np.ndarray, neighbours: np.ndarray) -> float:
    return float((labels[neighbours] == labels[:, None]).mean())


def same_protein_fraction(values: np.ndarray, protein_index: np.ndarray) -> float:
    _, neighbours = cKDTree(values).query(values, k=K + 1, workers=-1)
    return float((protein_index[neighbours[:, 1:]] == protein_index[:, None]).mean())


def aligned_property(selection: CombinedSelection, labels: dict, key: str, valid_key: str):
    values, valid = [], []
    for protein_id, indices in selection.keys:
        label = labels[protein_id]
        values.append(np.asarray(label[key])[indices])
        valid.append(np.asarray(label[valid_key])[indices])
    return np.concatenate(values), np.concatenate(valid)


def neighbour_spearman(target: np.ndarray, valid: np.ndarray, neighbours: np.ndarray) -> tuple[float, int]:
    usable = valid & np.all(valid[neighbours], axis=1) & np.isfinite(target)
    prediction = target[neighbours[usable]].mean(axis=1)
    return float(spearmanr(target[usable], prediction).statistic), int(usable.sum())


def annotate_indices(ax, embedding: np.ndarray, residue_index: np.ndarray) -> None:
    for index0 in np.unique(residue_index):
        index1 = int(index0) + 1
        if index1 != 1 and index1 % 25 != 0:
            continue
        mask = residue_index == index0
        if mask.sum() < 3:
            continue
        x, y = representative_point(embedding[mask])
        text = ax.text(x, y, str(index1), fontsize=5, weight="bold", ha="center", va="center")
        text.set_path_effects([
            path_effects.Stroke(linewidth=1.2, foreground="white"),
            path_effects.Normal(),
        ])


def main() -> None:
    register()
    extract(checkpoint_ids=list(NEW_CHECKPOINTS), device="cuda")
    run_tsne(
        checkpoint_ids=list(IDS), n_points=N_POINTS, perplexity=PERPLEXITY,
        seed=SEED, max_iter=MAX_ITER, force=False,
        output_tag="rotation-d8-spatial-relative-10k",
    )

    labels = load_labels()
    latent_caches = {checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in IDS}
    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    aa = selection.aa[sampled]
    ss = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    residue_index = local_index_array(selection)[sampled]

    lengths = np.concatenate([
        np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
        for protein_id, indices in selection.keys
    ])[sampled]
    norm_position = residue_index / np.maximum(lengths - 1.0, 1.0)
    log_length = np.log(lengths)
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]

    rows = []
    embeddings = {}
    for checkpoint_id in IDS:
        latent = selection.build_X(latent_caches[checkpoint_id])[sampled].astype(np.float32)
        latent_norm = normalized(latent)
        neighbours = external_neighbours(latent, protein_index)
        position_stats = knn_exact_index_stats(latent_norm, residue_index)
        embedding = np.load(
            coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED), allow_pickle=False
        )["embedding"]
        embeddings[checkpoint_id] = embedding
        tsne_position_stats = knn_exact_index_stats(embedding, residue_index)
        position_rho, position_n = neighbour_spearman(
            norm_position, np.ones(N_POINTS, dtype=bool), neighbours
        )
        length_rho, length_n = neighbour_spearman(
            log_length, np.ones(N_POINTS, dtype=bool), neighbours
        )
        rsa_rho, rsa_n = neighbour_spearman(rsa, rsa_valid, neighbours)
        rows.append({
            "checkpoint": checkpoint_id,
            "checkpoint_path": str(
                NEW_CHECKPOINTS.get(
                    checkpoint_id,
                    REFERENCE_CHECKPOINTS.get(checkpoint_id, "cached reference"),
                )
            ),
            "proteins": len(protein_ids),
            "sampled_residues": len(sampled),
            "latent_dim": latent.shape[1],
            "effective_rank": effective_rank(latent),
            "aa_purity_at_10_cross_protein": purity(aa, neighbours),
            "ss_purity_at_10_cross_protein": purity(ss, neighbours),
            "absolute_index_purity_at_10": position_stats["exact_index_purity_at_10"],
            "tsne_absolute_index_purity_at_10": tsne_position_stats["exact_index_purity_at_10"],
            "same_protein_fraction_at_10": same_protein_fraction(latent_norm, protein_index),
            "tsne_same_protein_fraction_at_10": same_protein_fraction(embedding, protein_index),
            "normalized_position_knn10_spearman": position_rho,
            "normalized_position_n": position_n,
            "log_chain_length_knn10_spearman": length_rho,
            "log_chain_length_n": length_n,
            "rsa_knn10_spearman": rsa_rho,
            "rsa_n": rsa_n,
        })

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aa_labels, aa_colors = view_labels_colors("aa20")
    fig, axes = plt.subplots(2, len(IDS), figsize=(4.1 * len(IDS), 8.2))
    scatter = None
    for column, row in enumerate(rows):
        checkpoint_id = row["checkpoint"]
        embedding = embeddings[checkpoint_id]
        scatter_panel(axes[0, column], embedding, aa, view="aa20", annotate_aa=True)
        axes[0, column].set_title(
            f"{NAMES[checkpoint_id]}\n"
            f"rank={row['effective_rank']:.2f}/{row['latent_dim']}; "
            f"AA@10={row['aa_purity_at_10_cross_protein']:.3f}", fontsize=9
        )
        scatter = axes[1, column].scatter(
            embedding[:, 0], embedding[:, 1], c=residue_index + 1, cmap="turbo",
            vmin=1, vmax=251, s=3, alpha=0.68, linewidths=0, rasterized=True,
        )
        annotate_indices(axes[1, column], embedding, residue_index)
        axes[1, column].set_title(
            "absolute chain index\n"
            f"latent exact-index@10={row['absolute_index_purity_at_10']:.3f}", fontsize=9
        )
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
        axes[1, column].set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=label)
        for label, color in zip(aa_labels, aa_colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, frameon=False)
    fig.suptitle(
        "Matched frozen-encoder t-SNE at 10k — same 15,000 held-out physical residues",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.02, right=0.965, bottom=0.075, top=0.90, wspace=0.05, hspace=0.17)
    cax = fig.add_axes([0.975, 0.12, 0.008, 0.33])
    fig.colorbar(scatter, cax=cax, label="absolute residue index")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=210, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {RESULT}")
    print(f"wrote {FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
