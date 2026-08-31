"""Separate the real JEPA-D64 latent geometry from the t-SNE drawing.

Uses only the existing frozen-latent and t-SNE caches.  The key comparison
decomposes each residue score along the two dominant latent PCs into a
protein-centroid term and a within-protein residual.
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "knn_purity"))
sys.path.insert(0, str(HERE))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT = "jepa-rand70-d64"
N_POINTS = 15_000
COORD_PATH = (
    HERE / "results" / "tsne_coordinates" /
    "jepa-rand70-d64_n15000_perp30p0_seed42_cosine.npz"
)
FIG_PATH = (
    HERE / "figures" / "tsne_absolute_index" /
    "jepa_rand70_d64_manifold_forensics.png"
)


def run() -> None:
    labels = load_labels()
    per_protein = load_encoder_latents(CHECKPOINT)
    protein_ids = common_protein_ids({CHECKPOINT: per_protein})
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    latent_all = selection.build_X(per_protein).astype(np.float32)
    latent = latent_all[sampled]
    tsne = np.load(COORD_PATH, allow_pickle=False)["embedding"]

    pca = PCA(n_components=8, random_state=SEED).fit(latent)
    pc = pca.transform(latent)[:, :2]

    # Project each complete protein centroid through the same PCA basis.
    centroid_pc = np.zeros((len(protein_ids), 2), dtype=np.float32)
    length = np.zeros(len(protein_ids), dtype=np.float32)
    helix_fraction = np.zeros(len(protein_ids), dtype=np.float32)
    for p_int, pid in enumerate(protein_ids):
        centroid_pc[p_int] = pca.transform(
            per_protein[pid].float().mean(0, keepdim=True).numpy()
        )[0, :2]
        length[p_int] = len(per_protein[pid])
        valid = labels[pid]["ss_valid"]
        ss = labels[pid]["ss"][valid]
        helix_fraction[p_int] = np.mean(ss == 0) if len(ss) else np.nan

    protein_int = selection.protein_idx[sampled]
    point_centroid_pc = centroid_pc[protein_int]
    within_pc = pc - point_centroid_pc

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    panels = [
        (pc, pc[:, 0], "Latent PCA itself\n(no t-SNE)"),
        (tsne, pc[:, 0], "t-SNE colored by residue PC1"),
        (tsne, point_centroid_pc[:, 0], "t-SNE colored by protein-centroid PC1"),
        (tsne, within_pc[:, 0], "t-SNE colored by within-protein PC1 residual"),
        (tsne, pc[:, 1], "t-SNE colored by residue PC2"),
        (tsne, point_centroid_pc[:, 1], "t-SNE colored by protein-centroid PC2"),
        (tsne, length[protein_int], "t-SNE colored by chain length"),
        (tsne, helix_fraction[protein_int], "t-SNE colored by protein helix fraction"),
    ]
    for ax, (coords, values, title) in zip(axes.flat, panels):
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=values, cmap="viridis",
            s=4, alpha=0.72, linewidths=0, rasterized=True,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(
        "JEPA-70 D64: the wave is dominated by a protein-level, near-2D code",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"sample PCA variance: {pca.explained_variance_ratio_[:8]}")
    print(f"wrote {FIG_PATH}")


if __name__ == "__main__":
    run()
