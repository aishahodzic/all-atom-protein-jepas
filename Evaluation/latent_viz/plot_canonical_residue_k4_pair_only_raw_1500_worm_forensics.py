"""Recolour the fixed raw-step1500 pair-only t-SNE to explain its worms.

This script performs no encoder forward and does not refit t-SNE.  Every
panel uses the identical frozen 15,000-residue sample and coordinates.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
for path in (
    REPO,
    HERE,
    REPO / "Evaluation/linear_probing",
    REPO / "Evaluation/knn_purity",
):
    sys.path.insert(0, str(path))

from analyze_current_four_health import aligned_property  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT = "canonical-lejepa-residue-k4-pair-only-raw-step1500"
N_POINTS = 15_000
COORD_PATH = HERE / (
    "results/tsne_coordinates/"
    f"{CHECKPOINT}_n15000_perp30p0_seed42_cosine.npz"
)
GRID_FIGURE = HERE / (
    "figures/canonical_lejepa/"
    "tsne_canonical_residue_k4_pair_only_raw_1500_recolored-grid.png"
)
PROTEIN_FIGURE = HERE / (
    "figures/canonical_lejepa/"
    "tsne_canonical_residue_k4_pair_only_raw_1500_source-proteins.png"
)
RESULT = HERE / (
    "results/canonical_residue_k4_pair_only_raw_1500_worm_forensics.json"
)

AA_LETTERS = np.asarray(list("ARNDCQEGHILKMFPSTWYV"))
AA_COLORS = np.asarray([plt.get_cmap("tab20")(i / 20) for i in range(20)])
AA_GROUP_NAMES = ("hydrophobic", "polar", "charged", "Gly/Pro")
AA_GROUP_COLORS = ("#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")
AA_GROUP_OF_LETTER = {
    **{letter: 0 for letter in "AVLIMFWY"},
    **{letter: 1 for letter in "STNQC"},
    **{letter: 2 for letter in "DEKRH"},
    **{letter: 3 for letter in "GP"},
}
SS_NAMES = ("helix", "sheet", "loop")
SS_COLORS = ("#d62728", "#1f77b4", "#7f7f7f")


def stable_protein_colors(n_proteins: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = (np.arange(n_proteins, dtype=np.float64) * 0.618033988749895) % 1.0
    rng.shuffle(hues)
    saturation = 0.62 + 0.30 * rng.random(n_proteins)
    value = 0.66 + 0.30 * rng.random(n_proteins)
    return hsv_to_rgb(np.column_stack((hues, saturation, value)))


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def categorical_panel(
    ax,
    embedding: np.ndarray,
    values: np.ndarray,
    names,
    colors,
    title: str,
) -> None:
    for value, (name, color) in enumerate(zip(names, colors)):
        mask = values == value
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=2.4, alpha=0.64, linewidths=0, c=[color],
            label=name, rasterized=True,
        )
    clean_axis(ax)
    ax.set_title(title, fontsize=10)
    ax.legend(
        frameon=False, fontsize=5.3, markerscale=2.0,
        ncol=4 if len(names) > 4 else len(names),
        loc="lower center", bbox_to_anchor=(0.5, -0.13),
        columnspacing=0.65, handletextpad=0.25,
    )


def continuous_panel(
    ax,
    embedding: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
    *,
    valid: np.ndarray | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if valid is None:
        valid = np.isfinite(values)
    invalid = ~valid
    if invalid.any():
        ax.scatter(
            embedding[invalid, 0], embedding[invalid, 1],
            s=2.0, alpha=0.25, linewidths=0, c="#d0d0d0", rasterized=True,
        )
    image = ax.scatter(
        embedding[valid, 0], embedding[valid, 1],
        c=values[valid], cmap=cmap, vmin=vmin, vmax=vmax,
        s=2.4, alpha=0.72, linewidths=0, rasterized=True,
    )
    clean_axis(ax)
    ax.set_title(title, fontsize=10)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.038, pad=0.012)
    colorbar.set_label(label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def neighbour_mean_spearman(values: np.ndarray, neighbours: np.ndarray) -> float:
    return float(spearmanr(values, values[neighbours].mean(axis=1)).statistic)


def main() -> None:
    if not COORD_PATH.is_file():
        raise FileNotFoundError(COORD_PATH)
    labels = load_labels()
    per_protein = load_encoder_latents(CHECKPOINT)
    protein_ids = common_protein_ids({CHECKPOINT: per_protein})
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    embedding = np.load(COORD_PATH, allow_pickle=False)["embedding"]
    if len(embedding) != len(sampled):
        raise ValueError(f"{len(embedding)} coordinates != {len(sampled)} residues")

    latent = selection.build_X(per_protein)[sampled].astype(np.float32)
    amino_acid = selection.aa[sampled]
    aa_letter = AA_LETTERS[amino_acid]
    aa_group = np.asarray(
        [AA_GROUP_OF_LETTER[letter] for letter in aa_letter], dtype=np.int64
    )
    secondary_structure = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate(
        [
            np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    normalized_position = (residue_index + 1.0) / lengths
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    bfactor, bfactor_valid = aligned_property(
        selection, labels, "bfactor_z", "bfactor_valid"
    )
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]
    bfactor, bfactor_valid = bfactor[sampled], bfactor_valid[sampled]

    # These collapsed latents differ at only ~1e-4 around O(1) means.  PCA in
    # float32 suffers catastrophic cancellation and reports a spurious rank-1
    # spectrum, so all centering/covariance work here is explicitly float64.
    latent64 = latent.astype(np.float64)
    pca = PCA(n_components=8, random_state=SEED).fit(latent64)
    principal = pca.transform(latent64)
    centroid_pc = np.zeros((len(protein_ids), 2), dtype=np.float32)
    for protein_int, protein_id in enumerate(protein_ids):
        centroid = (
            per_protein[protein_id].double().mean(0, keepdim=True).numpy()
        )
        centroid_pc[protein_int] = pca.transform(centroid)[0, :2]
    point_centroid_pc = centroid_pc[protein_index]
    within_pc = principal[:, :2] - point_centroid_pc
    latent_norm = np.linalg.norm(latent, axis=1)
    protein_palette = stable_protein_colors(len(protein_ids))

    fig, axes = plt.subplots(4, 4, figsize=(18.5, 17.5))
    categorical_panel(
        axes[0, 0], embedding, amino_acid, AA_LETTERS, AA_COLORS,
        "amino-acid identity",
    )
    categorical_panel(
        axes[0, 1], embedding, aa_group, AA_GROUP_NAMES, AA_GROUP_COLORS,
        "amino-acid biochemical class",
    )
    charge = np.full(len(aa_letter), 2, dtype=np.int64)
    charge[np.isin(aa_letter, list("RKH"))] = 0
    charge[np.isin(aa_letter, list("DE"))] = 1
    categorical_panel(
        axes[0, 2], embedding, charge,
        ("basic", "acidic", "neutral"), ("#2166ac", "#b2182b", "#bdbdbd"),
        "residue charge",
    )
    categorical_panel(
        axes[0, 3], embedding, secondary_structure, SS_NAMES, SS_COLORS,
        "secondary structure",
    )

    continuous_panel(
        axes[1, 0], embedding, rsa, "relative solvent accessibility", "RSA",
        valid=rsa_valid, cmap="viridis", vmin=0.0, vmax=1.0,
    )
    bfactor_limit = float(np.nanpercentile(np.abs(bfactor[bfactor_valid]), 98))
    continuous_panel(
        axes[1, 1], embedding, bfactor, "normalized B-factor", "B-factor z",
        valid=bfactor_valid, cmap="coolwarm", vmin=-bfactor_limit,
        vmax=bfactor_limit,
    )
    continuous_panel(
        axes[1, 2], embedding, residue_index + 1.0,
        "absolute sequence index", "residue i", cmap="turbo",
        vmin=1.0, vmax=float((residue_index + 1.0).max()),
    )
    continuous_panel(
        axes[1, 3], embedding, normalized_position,
        "normalized sequence position", "i/L", cmap="viridis",
        vmin=0.0, vmax=1.0,
    )

    continuous_panel(
        axes[2, 0], embedding, lengths, "source chain length", "L",
        cmap="magma", vmin=float(lengths.min()), vmax=float(lengths.max()),
    )
    axes[2, 1].scatter(
        embedding[:, 0], embedding[:, 1], c=protein_palette[protein_index],
        s=2.4, alpha=0.72, linewidths=0, rasterized=True,
    )
    clean_axis(axes[2, 1])
    axes[2, 1].set_title("source protein (stable random colours)", fontsize=10)
    continuous_panel(
        axes[2, 2], embedding, latent_norm, "latent-vector norm", "||z||",
        cmap="cividis",
    )
    pc1_limit = float(np.percentile(np.abs(principal[:, 0]), 98))
    continuous_panel(
        axes[2, 3], embedding, principal[:, 0],
        "residue latent PC1", "PC1", cmap="coolwarm",
        vmin=-pc1_limit, vmax=pc1_limit,
    )

    pc2_limit = float(np.percentile(np.abs(principal[:, 1]), 98))
    continuous_panel(
        axes[3, 0], embedding, principal[:, 1],
        "residue latent PC2", "PC2", cmap="coolwarm",
        vmin=-pc2_limit, vmax=pc2_limit,
    )
    centroid1_limit = float(np.percentile(np.abs(point_centroid_pc[:, 0]), 98))
    continuous_panel(
        axes[3, 1], embedding, point_centroid_pc[:, 0],
        "source-protein centroid PC1", "centroid PC1", cmap="coolwarm",
        vmin=-centroid1_limit, vmax=centroid1_limit,
    )
    within1_limit = float(np.percentile(np.abs(within_pc[:, 0]), 98))
    continuous_panel(
        axes[3, 2], embedding, within_pc[:, 0],
        "within-protein PC1 residual", "residual PC1", cmap="coolwarm",
        vmin=-within1_limit, vmax=within1_limit,
    )
    centroid2_limit = float(np.percentile(np.abs(point_centroid_pc[:, 1]), 98))
    continuous_panel(
        axes[3, 3], embedding, point_centroid_pc[:, 1],
        "source-protein centroid PC2", "centroid PC2", cmap="coolwarm",
        vmin=-centroid2_limit, vmax=centroid2_limit,
    )
    fig.suptitle(
        "Pair-only residue K=4, raw step 1,500 — identical t-SNE recoloured",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.1)
    GRID_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRID_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    observed_proteins, observed_counts = np.unique(
        protein_index, return_counts=True
    )
    highlighted = observed_proteins[np.argsort(observed_counts)[::-1][:14]]
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
    axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=protein_palette[protein_index],
        s=5, alpha=0.76, linewidths=0, rasterized=True,
    )
    axes[0].set_title("All points coloured by their source protein")
    axes[1].scatter(
        embedding[:, 0], embedding[:, 1], c="#d8d8d8",
        s=3, alpha=0.18, linewidths=0, rasterized=True,
    )
    for protein_int in highlighted:
        mask = protein_index == protein_int
        order = np.argsort(residue_index[mask])
        coords = embedding[mask][order]
        indices = residue_index[mask][order] + 1
        color = protein_palette[protein_int]
        axes[1].scatter(
            coords[:, 0], coords[:, 1], color=color, s=28, alpha=0.95,
            edgecolors="white", linewidths=0.3,
            label=f"{protein_ids[protein_int]} ({len(coords)})", zorder=4,
        )
        for (x, y), index1 in zip(coords, indices):
            axes[1].text(
                x, y, str(int(index1)), fontsize=5.0, color="black",
                ha="center", va="center", zorder=5,
            )
    axes[1].set_title(
        "Fourteen most-sampled proteins highlighted\n"
        "point labels are one-based residue indices"
    )
    axes[1].legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=7,
    )
    for ax in axes:
        clean_axis(ax)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        "Are the apparent worms individual proteins?",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.015, right=0.83, bottom=0.02, top=0.91, wspace=0.03)
    fig.savefig(PROTEIN_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    _, tsne_neighbours = cKDTree(embedding).query(embedding, k=11, workers=-1)
    tsne_neighbours = tsne_neighbours[:, 1:]
    covariance_eigenvalues = np.maximum(
        np.linalg.eigvalsh(np.cov(latent64, rowvar=False)), 0.0
    )[::-1]
    covariance_probabilities = covariance_eigenvalues / np.maximum(
        covariance_eigenvalues.sum(), 1e-30
    )
    nonzero_probabilities = covariance_probabilities[
        covariance_probabilities > 0
    ]
    effective_rank = float(
        np.exp(
            -(
                nonzero_probabilities * np.log(nonzero_probabilities)
            ).sum()
        )
    )
    dimension_std = latent64.std(axis=0)
    rms_spread = float(np.sqrt(np.square(dimension_std).sum()))
    mean_norm = float(np.linalg.norm(latent64, axis=1).mean())
    stats = {
        "checkpoint": CHECKPOINT,
        "same_coordinates_as_chain_length_figure": True,
        "sampled_residues": len(sampled),
        "sampled_source_proteins": int(np.unique(protein_index).size),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "effective_rank_float64_covariance": effective_rank,
        "latent_dimension_std_min": float(dimension_std.min()),
        "latent_dimension_std_max": float(dimension_std.max()),
        "latent_mean_norm": mean_norm,
        "latent_rms_spread": rms_spread,
        "latent_relative_rms_spread": rms_spread / mean_norm,
        "tsne_neighbour_same_protein_fraction_at_10": float(
            (protein_index[tsne_neighbours] == protein_index[:, None]).mean()
        ),
        "tsne_neighbour_exact_aa_purity_at_10": float(
            (amino_acid[tsne_neighbours] == amino_acid[:, None]).mean()
        ),
        "tsne_neighbour_ss_purity_at_10": float(
            (
                secondary_structure[tsne_neighbours]
                == secondary_structure[:, None]
            ).mean()
        ),
        "tsne_neighbour_raw_index_spearman": neighbour_mean_spearman(
            residue_index, tsne_neighbours
        ),
        "tsne_neighbour_normalized_position_spearman": neighbour_mean_spearman(
            normalized_position, tsne_neighbours
        ),
        "tsne_neighbour_log_chain_length_spearman": neighbour_mean_spearman(
            np.log(lengths), tsne_neighbours
        ),
        "grid_figure": str(GRID_FIGURE),
        "source_protein_figure": str(PROTEIN_FIGURE),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
