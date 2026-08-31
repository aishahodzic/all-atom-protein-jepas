"""Matched milestone analysis for relative-position LeJEPA at 10k and 30k."""

import csv
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, knn_purity, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import knn_exact_index_stats, local_index_array, representative_point  # noqa: E402
from run_tsne import (  # noqa: E402
    coordinate_path,
    run as run_tsne,
    scatter_panel,
    view_labels_colors,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CONTROL = "le-jepa-rand-mask-50"
REL10 = "le-jepa-rand50-relative-pos-10k"
REL30 = "le-jepa-rand50-relative-pos-30k"
RUN_ROOT = pathlib.Path("/mnt/hopper-raid/aishod/runs/le_jepa_rand50_relative_positions_10k")
CHECKPOINT_DIR = RUN_ROOT / "store/le_jepa_rand50_relative_positions_10k/checkpoints"
CKPT10 = CHECKPOINT_DIR / "chk_epoch=00000001_step=000000010000.ckpt"
CKPT30 = CHECKPOINT_DIR / "chk_epoch=00000005_step=000000030000.ckpt"
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
IDS = (CONTROL, REL10, REL30)
RESULT = HERE / "results/lejepa_relative_position_milestones_10k_30k.csv"
POSITION_FIGURE = HERE / "figures/tsne_absolute_index/lejepa_relative_position_10k_30k.png"
AA_FIGURE = HERE / "figures/tsne_aa20_lejepa_relative_position_10k_30k.png"


def register() -> None:
    for checkpoint_id, checkpoint in ((REL10, CKPT10), (REL30, CKPT30)):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        entry = (str(checkpoint), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def same_protein_fraction(values: np.ndarray, protein_index: np.ndarray) -> float:
    _, neighbours = cKDTree(values).query(values, k=11, workers=-1)
    return float((protein_index[neighbours[:, 1:]] == protein_index[:, None]).mean())


def effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / np.maximum(eigenvalues.sum(), 1e-12)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def label_indices(ax, embedding: np.ndarray, residue_index: np.ndarray) -> None:
    for index0 in np.unique(residue_index):
        index1 = int(index0) + 1
        if index1 != 1 and index1 % 20 != 0:
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


def analyze() -> None:
    register()
    extract(checkpoint_ids=[REL30], device="cuda")
    run_tsne(
        checkpoint_ids=list(IDS), n_points=N_POINTS, perplexity=PERPLEXITY,
        seed=SEED, max_iter=MAX_ITER, force=False,
        output_tag="lejepa-relative-position-10k-30k",
    )

    labels = load_labels()
    per_protein = {checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in IDS}
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    aa = selection.aa[sampled]
    ss = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    residue_index = local_index_array(selection)[sampled]
    query_indices = np.arange(len(sampled))

    rows = []
    embeddings = {}
    for checkpoint_id in IDS:
        latent = selection.build_X(per_protein[checkpoint_id])[sampled]
        normalized = normalized_rows(latent)
        embedding = np.load(
            coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED),
            allow_pickle=False,
        )["embedding"]
        embeddings[checkpoint_id] = embedding
        pos = knn_exact_index_stats(normalized, residue_index)
        tsne_pos = knn_exact_index_stats(embedding, residue_index)
        aa_purity = knn_purity(
            torch.from_numpy(latent), aa, protein_index, query_indices,
            k_values=(10,), device="cuda", batch_size=512,
        )[10]
        ss_purity = knn_purity(
            torch.from_numpy(latent), ss, protein_index, query_indices,
            k_values=(10,), device="cuda", batch_size=512,
        )[10]
        rows.append({
            "checkpoint": checkpoint_id,
            "training_steps": 100000 if checkpoint_id == CONTROL else (10000 if checkpoint_id == REL10 else 30000),
            "proteins": len(protein_ids),
            "sampled_residues": len(sampled),
            "latent_dim": latent.shape[1],
            "effective_rank": effective_rank(latent),
            "latent_absolute_index_purity_at_10": pos["exact_index_purity_at_10"],
            "latent_nearest_neighbor_exact_index": pos["nearest_neighbor_exact_index"],
            "latent_aa_purity_at_10": aa_purity,
            "latent_ss_purity_at_10": ss_purity,
            "latent_same_protein_fraction_at_10": same_protein_fraction(normalized, protein_index),
            "tsne_absolute_index_purity_at_10": tsne_pos["exact_index_purity_at_10"],
            "tsne_nearest_neighbor_exact_index": tsne_pos["nearest_neighbor_exact_index"],
        })

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = {
        CONTROL: "Original absolute positions (100k)",
        REL10: "Relative positions (10k)",
        REL30: "Relative positions (30k)",
    }
    one_based = residue_index + 1
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    scatter = None
    for ax, row in zip(axes, rows):
        checkpoint_id = row["checkpoint"]
        embedding = embeddings[checkpoint_id]
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=one_based, cmap="turbo",
            vmin=1, vmax=251, s=4, alpha=0.70, linewidths=0, rasterized=True,
        )
        label_indices(ax, embedding, residue_index)
        ax.set_title(
            f"{names[checkpoint_id]}\nabsolute-index purity@10 = "
            f"{row['latent_absolute_index_purity_at_10']:.3f}", fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Relative-position LeJEPA milestone analysis — same 15,000 physical residues")
    fig.subplots_adjust(left=0.01, right=0.93, bottom=0.03, top=0.86, wspace=0.03)
    cax = fig.add_axes([0.945, 0.15, 0.012, 0.60])
    fig.colorbar(scatter, cax=cax, label="absolute residue index within chain")
    POSITION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSITION_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    aa_labels, aa_colors = view_labels_colors("aa20")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
    for ax, row in zip(axes, rows):
        checkpoint_id = row["checkpoint"]
        scatter_panel(ax, embeddings[checkpoint_id], aa, view="aa20", annotate_aa=True)
        ax.set_title(
            f"{names[checkpoint_id]}\n8D AA purity@10 = {row['latent_aa_purity_at_10']:.3f}",
            fontsize=11,
        )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=label)
        for label, color in zip(aa_labels, aa_colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, frameon=False)
    fig.suptitle("Relative-position LeJEPA residue latents colored by amino-acid identity")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    fig.savefig(AA_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {RESULT}")
    print(f"wrote {POSITION_FIGURE}")
    print(f"wrote {AA_FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    analyze()
