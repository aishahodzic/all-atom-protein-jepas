"""Diagnose what organizes the JEPA-70 D64 t-SNE wave.

Recolors the already-computed t-SNE coordinates; no t-SNE or encoder forward
is repeated.  Quantitative local-neighbour statistics compare the visible
2-D manifold with residue position, protein identity/length, AA, SS, and the
leading axes of the original 64-D latent space.
"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "knn_purity"))
sys.path.insert(0, str(HERE))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array  # noqa: E402
from viz import AA_LETTERS, CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT = "jepa-rand70-d64"
N_POINTS = 15_000
COORD_PATH = (
    HERE / "results" / "tsne_coordinates" /
    "jepa-rand70-d64_n15000_perp30p0_seed42_cosine.npz"
)
FIG_PATH = HERE / "figures" / "tsne_absolute_index" / "jepa_rand70_d64_wave_diagnostic.png"
RESULT_PATH = HERE / "results" / "jepa_rand70_d64_wave_diagnostic.csv"


def mean_neighbour_delta(values: np.ndarray, neighbours: np.ndarray) -> float:
    scale = float(np.ptp(values))
    if scale == 0:
        return 0.0
    return float(np.abs(values[neighbours] - values[:, None]).mean() / scale)


def categorical_purity(values: np.ndarray, neighbours: np.ndarray) -> float:
    return float((values[neighbours] == values[:, None]).mean())


def run() -> None:
    labels = load_labels()
    per_protein = {CHECKPOINT: load_encoder_latents(CHECKPOINT)}
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    embedding = np.load(COORD_PATH, allow_pickle=False)["embedding"]
    latent = selection.build_X(per_protein[CHECKPOINT])[sampled].astype(np.float32)

    absolute_index = local_index_array(selection)[sampled]
    protein_int = selection.protein_idx[sampled]
    aa = selection.aa[sampled]
    ss = selection.ss[sampled]
    lengths = {
        p_int: len(per_protein[CHECKPOINT][pid])
        for p_int, pid in enumerate(protein_ids)
    }
    chain_length = np.array([lengths[p] for p in protein_int], dtype=np.int64)
    relative_position = absolute_index / np.maximum(chain_length - 1, 1)

    pca = PCA(n_components=8, random_state=SEED)
    latent_pcs = pca.fit_transform(latent)
    _, neighbours = cKDTree(embedding).query(embedding, k=11, workers=-1)
    neighbours = neighbours[:, 1:]

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(embedding))
    shuffled_neighbours = shuffled[neighbours]
    rows = [
        {
            "quantity": "absolute_index",
            "local_same_or_similarity": categorical_purity(absolute_index, neighbours),
            "shuffled_baseline": categorical_purity(absolute_index, shuffled_neighbours),
            "local_normalized_absolute_delta": mean_neighbour_delta(absolute_index, neighbours),
        },
        {
            "quantity": "relative_position",
            "local_same_or_similarity": 1.0 - mean_neighbour_delta(relative_position, neighbours),
            "shuffled_baseline": 1.0 - mean_neighbour_delta(relative_position, shuffled_neighbours),
            "local_normalized_absolute_delta": mean_neighbour_delta(relative_position, neighbours),
        },
        {
            "quantity": "chain_length",
            "local_same_or_similarity": 1.0 - mean_neighbour_delta(chain_length, neighbours),
            "shuffled_baseline": 1.0 - mean_neighbour_delta(chain_length, shuffled_neighbours),
            "local_normalized_absolute_delta": mean_neighbour_delta(chain_length, neighbours),
        },
        {
            "quantity": "amino_acid",
            "local_same_or_similarity": categorical_purity(aa, neighbours),
            "shuffled_baseline": categorical_purity(aa, shuffled_neighbours),
            "local_normalized_absolute_delta": "",
        },
        {
            "quantity": "secondary_structure",
            "local_same_or_similarity": categorical_purity(ss, neighbours),
            "shuffled_baseline": categorical_purity(ss, shuffled_neighbours),
            "local_normalized_absolute_delta": "",
        },
        {
            "quantity": "same_protein",
            "local_same_or_similarity": categorical_purity(protein_int, neighbours),
            "shuffled_baseline": categorical_purity(protein_int, shuffled_neighbours),
            "local_normalized_absolute_delta": "",
        },
    ]
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    panels = [
        (latent_pcs[:, 0], "viridis", f"Original latent PC1\n{pca.explained_variance_ratio_[0]:.1%} variance"),
        (latent_pcs[:, 1], "viridis", f"Original latent PC2\n{pca.explained_variance_ratio_[1]:.1%} variance"),
        (absolute_index + 1, "turbo", "Absolute residue index (one-based)"),
        (relative_position, "viridis", "Relative chain position (0=N, 1=C)"),
        (chain_length, "viridis", "Source-chain length"),
        (aa, "tab20", "Amino-acid identity"),
        (ss, "Set1", "Secondary structure (3 classes)"),
        (protein_int, "gist_ncar", "Protein ID (color itself is arbitrary)"),
    ]
    for ax, (values, cmap, title) in zip(axes.flat, panels):
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=values, cmap=cmap,
            s=4, alpha=0.7, linewidths=0, rasterized=True,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if title != "Protein ID (color itself is arbitrary)":
            fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(
        "JEPA-70 D64: what organizes the wavy t-SNE manifold?\n"
        "Same coordinates in every panel; only point color changes",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"PCA variance first 8: {pca.explained_variance_ratio_}")
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {FIG_PATH}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    run()
