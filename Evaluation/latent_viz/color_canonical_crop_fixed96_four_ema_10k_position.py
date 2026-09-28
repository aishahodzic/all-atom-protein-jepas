"""Color the matched four-run EMA-10k t-SNEs by position and chain length."""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

from analyze_current_four_health import (  # noqa: E402
    external_neighbours,
    neighbour_spearman,
)
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array  # noqa: E402
from run_tsne import coordinate_path  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


SPECS = {
    "canonical-lejepa-residue-cropped-fixed96-perview-sigreg-b96-ema-step10000": (
        "Residue loss / per-view SIGReg / B96"
    ),
    "canonical-lejepa-protein-cropped-fixed96-perview-sigreg-b96-ema-step10000": (
        "Protein loss / per-view SIGReg / B96"
    ),
    "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-ema-step10000": (
        "Protein loss / joint-view SIGReg / B96"
    ),
    "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b32-ema-step10000": (
        "Protein loss / joint-view SIGReg / B32"
    ),
}
N_POINTS = 15_000
PERPLEXITY = 30.0
FIGURE = HERE / (
    "figures/tsne_position_canonical-crop-fixed96-four-ema-10k_grid.png"
)
RESULT = HERE / (
    "results/tsne_position_metrics_canonical-crop-fixed96-four-ema-10k.csv"
)


def correlations(
    values: np.ndarray,
    protein_index: np.ndarray,
    residue_index: np.ndarray,
    normalized_position: np.ndarray,
    log_length: np.ndarray,
) -> dict[str, float]:
    neighbours = external_neighbours(values, protein_index)
    valid = np.ones(len(values), dtype=bool)
    raw_rho, _ = neighbour_spearman(residue_index, valid, neighbours)
    normalized_rho, _ = neighbour_spearman(normalized_position, valid, neighbours)
    length_rho, _ = neighbour_spearman(log_length, valid, neighbours)
    return {
        "raw_index_knn10_spearman": raw_rho,
        "normalized_position_knn10_spearman": normalized_rho,
        "log_chain_length_knn10_spearman": length_rho,
        "exact_index_at_10_cross_protein": float(
            (residue_index[neighbours] == residue_index[:, None]).mean()
        ),
    }


def main() -> None:
    checkpoint_ids = list(SPECS)
    labels = load_labels()
    caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in checkpoint_ids
    }
    protein_ids = common_protein_ids(caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    protein_index = selection.protein_idx[sampled]
    lengths = np.concatenate(
        [
            np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    normalized_position = (residue_index + 1.0) / lengths
    log_length = np.log(lengths)

    fig, axes = plt.subplots(
        len(checkpoint_ids), 3, figsize=(13.2, 3.7 * len(checkpoint_ids)), squeeze=False
    )
    rows: list[dict[str, object]] = []
    for row_index, checkpoint_id in enumerate(checkpoint_ids):
        latent = selection.build_X(caches[checkpoint_id])[sampled].astype(np.float32)
        embedding = np.load(
            coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED),
            allow_pickle=False,
        )["embedding"]
        latent_stats = correlations(
            latent, protein_index, residue_index, normalized_position, log_length
        )
        tsne_stats = correlations(
            embedding, protein_index, residue_index, normalized_position, log_length
        )
        rows.append(
            {
                "checkpoint": checkpoint_id,
                **{f"latent_{key}": value for key, value in latent_stats.items()},
                **{f"tsne_{key}": value for key, value in tsne_stats.items()},
            }
        )
        colors = (
            (residue_index + 1.0, "turbo", 1.0, 256.0, "absolute residue index i"),
            (normalized_position, "viridis", 0.0, 1.0, "normalized position i/L"),
            (lengths, "magma", 50.0, 256.0, "source chain length L"),
        )
        titles = (
            f"absolute index; latent rho={latent_stats['raw_index_knn10_spearman']:.3f}",
            f"normalized i/L; latent rho={latent_stats['normalized_position_knn10_spearman']:.3f}",
            f"chain length; latent rho={latent_stats['log_chain_length_knn10_spearman']:.3f}",
        )
        for column, ((color, cmap, vmin, vmax, label), title) in enumerate(
            zip(colors, titles)
        ):
            ax = axes[row_index, column]
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=color, cmap=cmap,
                vmin=vmin, vmax=vmax, s=2.7, alpha=0.72, linewidths=0,
                rasterized=True,
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.015)
            colorbar.set_label(label, fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
        axes[row_index, 0].set_ylabel(SPECS[checkpoint_id], fontsize=10)

    fig.suptitle(
        "Canonical crop 96/48 EMA-10k: positional-shortcut audit\n"
        "Identical 15,000 residues; identical precomputed cosine t-SNE coordinates",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {FIGURE}")
    print(f"wrote {RESULT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
