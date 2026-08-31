"""Matched latent-manifold analysis for the JEPA-D64 10k 2x2 intervention grid."""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from run_tsne import coordinate_path, run as run_tsne  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


RUN_ROOT = pathlib.Path("/mnt/hopper-raid/aishod/runs")
CHECKPOINTS = {
    "jepa-d64-10k-baseline": RUN_ROOT / "jepa_rand70_dim64/store/jepa_rand70_dim64/checkpoints/chk_epoch=00000001_step=000000010000.ckpt",
    "jepa-d64-10k-sigreg": RUN_ROOT / "jepa_rand70_d64_sigreg_10k/store/jepa_rand70_d64_sigreg_10k/checkpoints/chk_epoch=00000001_step=000000010000.ckpt",
    "jepa-d64-10k-rot": RUN_ROOT / "jepa_rand70_d64_independent_rotations_10k/store/jepa_rand70_d64_independent_rotations_10k/checkpoints/chk_epoch=00000001_step=000000010000.ckpt",
    "jepa-d64-10k-sigreg-rot": RUN_ROOT / "jepa_rand70_d64_sigreg_independent_rotations_10k/store/jepa_rand70_d64_sigreg_independent_rotations_10k/checkpoints/chk_epoch=00000001_step=000000010000.ckpt",
}
LABELS = {
    "jepa-d64-10k-baseline": "Baseline",
    "jepa-d64-10k-sigreg": "SIGReg",
    "jepa-d64-10k-rot": "Independent rotations",
    "jepa-d64-10k-sigreg-rot": "SIGReg + rotations",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
RESULT = HERE / "results" / "jepa_d64_2x2_10k_manifold.csv"
FIGURE = HERE / "figures" / "tsne_absolute_index" / "jepa_d64_2x2_10k_manifold.png"


def register() -> None:
    for checkpoint_id, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def effective_rank(values: np.ndarray) -> float:
    centered = values.astype(np.float64) - values.mean(0, keepdims=True)
    variance = np.linalg.svd(centered, full_matrices=False, compute_uv=False) ** 2
    probabilities = variance / variance.sum()
    return float(np.exp(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()))


def run() -> None:
    register()
    checkpoint_ids = list(CHECKPOINTS)
    extract(checkpoint_ids=checkpoint_ids, device="cuda")
    run_tsne(
        checkpoint_ids=checkpoint_ids,
        n_points=N_POINTS,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
        output_tag="jepa-d64-2x2-10k",
        show_suptitle=False,
    )

    labels = load_labels()
    per_checkpoint = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in checkpoint_ids
    }
    protein_ids = common_protein_ids(per_checkpoint)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    protein_index = selection.protein_idx[sampled]

    rows = []
    panel_data = []
    for checkpoint_id in checkpoint_ids:
        latent = selection.build_X(per_checkpoint[checkpoint_id])[sampled].astype(np.float32)
        pca = PCA(n_components=8, random_state=SEED).fit(latent)
        pc = pca.transform(latent)

        # Weighted protein-centroid variance divided by total residue variance.
        global_mean = latent.mean(0)
        between_ss = 0.0
        for protein in np.unique(protein_index):
            members = latent[protein_index == protein]
            between_ss += len(members) * float(((members.mean(0) - global_mean) ** 2).sum())
        total_ss = float(((latent - global_mean) ** 2).sum())
        between_fraction = between_ss / total_ss

        coord = coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED)
        embedding = np.load(coord, allow_pickle=False)["embedding"]
        _, neighbours = cKDTree(embedding).query(embedding, k=11, workers=-1)
        neighbours = neighbours[:, 1:]
        same_protein = float((protein_index[neighbours] == protein_index[:, None]).mean())

        row = {
            "checkpoint": checkpoint_id,
            "checkpoint_path": str(CHECKPOINTS[checkpoint_id]),
            "sampled_proteins": len(protein_ids),
            "sampled_residues": len(sampled),
            "effective_rank": effective_rank(latent),
            "pca_pc1_fraction": float(pca.explained_variance_ratio_[0]),
            "pca_pc1_pc2_fraction": float(pca.explained_variance_ratio_[:2].sum()),
            "pca_first8_fraction": float(pca.explained_variance_ratio_[:8].sum()),
            "protein_centroid_variance_fraction": between_fraction,
            "tsne_same_protein_fraction_at_10": same_protein,
        }
        rows.append(row)
        panel_data.append((checkpoint_id, embedding, pc[:, 0], row))

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    for column, (checkpoint_id, embedding, pc1, row) in enumerate(panel_data):
        axes[0, column].scatter(
            embedding[:, 0], embedding[:, 1], c=pc1, cmap="viridis",
            s=3, alpha=0.65, linewidths=0, rasterized=True,
        )
        axes[1, column].scatter(
            embedding[:, 0], embedding[:, 1], c=protein_index, cmap="gist_ncar",
            s=3, alpha=0.65, linewidths=0, rasterized=True,
        )
        axes[0, column].set_title(
            f"{LABELS[checkpoint_id]}\n"
            f"rank={row['effective_rank']:.2f}/64; PC1+2={row['pca_pc1_pc2_fraction']:.1%}"
        )
        axes[1, column].set_title(
            f"colored by protein\ncentroid variance={row['protein_centroid_variance_fraction']:.1%}; "
            f"same-protein@10={row['tsne_same_protein_fraction_at_10']:.2f}"
        )
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal", adjustable="box")
    axes[0, 0].set_ylabel("t-SNE colored by latent PC1")
    axes[1, 0].set_ylabel("t-SNE colored by protein ID")
    fig.suptitle("JEPA rand70 D64 at 10k: matched 2x2 latent manifolds", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {RESULT}")
    print(f"wrote {FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a CUDA node for latent extraction")
    run()
