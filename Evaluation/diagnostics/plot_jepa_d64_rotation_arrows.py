"""Show how rigidly rotating identical proteins moves JEPA-D64 latents.

The cached held-out residue cloud defines a fixed PCA plane.  We then encode a
small fixed protein panel at identity and three 90-degree rotations and draw
centroid-displacement arrows in that same plane.  Unlike a t-SNE recoloring,
this is a controlled intervention: protein identity and structure are held
fixed and only the global coordinate frame changes.
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Evaluation" / "diagnostics"))
sys.path.insert(0, str(REPO / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO / "Evaluation" / "knn_purity"))
sys.path.insert(0, str(REPO / "Evaluation" / "latent_viz"))

import rotation_latent_gauge as rotation_test  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT_ID = "jepa-rand70-d64"
MODEL_ID = "jepa_rand70_d64_100k_online"
OUT = (
    REPO / "Evaluation" / "latent_viz" / "figures" /
    "tsne_absolute_index" / "jepa_rand70_d64_rotation_arrows_pca.png"
)


def run() -> None:
    cached = load_encoder_latents(CHECKPOINT_ID)
    labels = load_labels()
    protein_ids = common_protein_ids({CHECKPOINT_ID: cached})
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=15_000, seed=SEED)
    latent_sample = selection.build_X(cached)[sampled].astype(np.float32)
    pca = PCA(n_components=2, random_state=SEED).fit(latent_sample)
    background = pca.transform(latent_sample)

    encoder = rotation_test.load_encoder(rotation_test.CHECKPOINTS[MODEL_ID])
    ids = rotation_test.TRAIN_IDS + rotation_test.TEST_IDS
    values = rotation_test.encode_all(encoder, ids)
    projected = {
        rotation: pca.transform(
            torch.stack([values[rotation][pid].mean(0) for pid in ids]).numpy()
        )
        for rotation in rotation_test.ROTATIONS
    }

    # Scale movement by the natural spread of all held-out protein centroids.
    all_centroids = pca.transform(
        np.stack([cached[pid].float().mean(0).numpy() for pid in protein_ids])
    )
    centroid_sd = all_centroids.std(0)

    rotations = [
        ("rz90", "90° around z", "#d62728"),
        ("rx90", "90° around x", "#1f77b4"),
        ("rz90_rx90", "90° z, then 90° x", "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), sharex=True, sharey=True)
    identity = projected["identity"]
    for ax, (rotation, label, color) in zip(axes, rotations):
        moved = projected[rotation]
        delta = moved - identity
        normalized_rms = np.sqrt(np.mean(delta**2, axis=0)) / centroid_sd
        ax.scatter(
            background[:, 0], background[:, 1], s=2, c="#bdbdbd",
            alpha=0.16, linewidths=0, rasterized=True,
        )
        for start, end in zip(identity, moved):
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.52, lw=0.9),
            )
        ax.scatter(identity[:, 0], identity[:, 1], s=15, c="black", label="original centroid", zorder=3)
        ax.scatter(moved[:, 0], moved[:, 1], s=18, c=color, label="rotated centroid", zorder=3)
        ax.set_title(
            f"{label}\nRMS movement: {normalized_rms[0]:.2f}× PC1, "
            f"{normalized_rms[1]:.2f}× PC2 centroid SD",
            fontsize=10,
        )
        ax.set_xlabel("Original latent PC1")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("Original latent PC2")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "JEPA-70 D64: identical proteins move through the latent PCA plane when globally rotated\n"
        "Gray = original held-out residue cloud; arrows change only the input coordinate frame",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
