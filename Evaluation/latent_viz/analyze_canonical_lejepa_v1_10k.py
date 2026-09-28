"""Frozen-encoder t-SNE and latent-health audit for canonical LeJEPA V1 at 10k."""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)
from analyze_current_four_health import (  # noqa: E402
    annotate_indices,
    external_neighbours,
    purity,
    same_protein_fraction,
    spectrum_metrics,
)
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import knn_exact_index_stats, local_index_array  # noqa: E402
from run_tsne import (  # noqa: E402
    AA_THREE,
    coordinate_path,
    run as run_tsne,
    scatter_panel,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT_ID = "canonical-lejepa-v1-ema-step10000"
CHECKPOINT = (
    REPO
    / "store/canonical_lejepa_v1_100k/checkpoints/step-00010000.ckpt"
)
STAGED_CHECKPOINT = (
    REPO
    / "store/canonical_lejepa_v1_100k/evaluation/"
    "canonical-lejepa-v1-step10000-ema-staged.ckpt"
)
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
RESULT = HERE / "results/canonical_lejepa_v1_ema_10k_health.csv"
FIGURE = (
    HERE
    / "figures/canonical_lejepa/"
    "tsne_canonical_lejepa_v1_ema_10k_aa-and-absolute-index.png"
)


def register() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    STAGED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if not STAGED_CHECKPOINT.is_file():
        stage_structtokenbench_checkpoint(
            CHECKPOINT, STAGED_CHECKPOINT, source="ema"
        )
    entry = (str(STAGED_CHECKPOINT), "jepa")
    checkpoints.CHECKPOINTS[CHECKPOINT_ID] = entry
    checkpoints.ALL_CHECKPOINTS[CHECKPOINT_ID] = entry


def annotate_amino_acids(
    ax: plt.Axes, embedding: np.ndarray, amino_acids: np.ndarray
) -> None:
    for aa_index, name in enumerate(AA_THREE):
        members = np.flatnonzero(amino_acids == aa_index)
        if not len(members):
            continue
        points = embedding[members]
        center = np.median(points, axis=0)
        chosen = members[np.square(points - center).sum(axis=1).argmin()]
        label = ax.text(
            embedding[chosen, 0],
            embedding[chosen, 1],
            name,
            color="black",
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )


def main() -> None:
    register()
    extract(checkpoint_ids=[CHECKPOINT_ID], device="cuda")

    import run_tsne as run_tsne_module

    run_tsne_module.SHORT_LABEL[CHECKPOINT_ID] = (
        "Canonical LeJEPA V1 EMA (10k)"
    )
    run_tsne(
        checkpoint_ids=[CHECKPOINT_ID],
        n_points=N_POINTS,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
        output_tag="canonical-lejepa-v1-ema-10k",
        grid_ncols=1,
        include_purity=False,
        show_suptitle=False,
    )

    labels = load_labels()
    latent_cache = load_encoder_latents(CHECKPOINT_ID)
    protein_ids = common_protein_ids({CHECKPOINT_ID: latent_cache})
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    latent = selection.build_X(latent_cache)[sampled].astype(np.float32)
    amino_acids = selection.aa[sampled]
    secondary_structure = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    residue_index = local_index_array(selection)[sampled]
    latent_normalized = latent / np.maximum(
        np.linalg.norm(latent, axis=1, keepdims=True), 1e-12
    )
    neighbours = external_neighbours(latent, protein_index)
    position_stats = knn_exact_index_stats(latent_normalized, residue_index)
    embedding = np.load(
        coordinate_path(CHECKPOINT_ID, N_POINTS, PERPLEXITY, SEED),
        allow_pickle=False,
    )["embedding"]
    tsne_position_stats = knn_exact_index_stats(embedding, residue_index)

    row = {
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_path": str(CHECKPOINT),
        "encoder_source": "ema",
        "training_steps": 10_000,
        "proteins": len(protein_ids),
        "sampled_residues": len(sampled),
        "latent_dim": latent.shape[1],
        **spectrum_metrics(latent),
        "latent_mean_norm": float(np.linalg.norm(latent, axis=1).mean()),
        "aa_purity_at_10_cross_protein": purity(amino_acids, neighbours),
        "ss_purity_at_10_cross_protein": purity(
            secondary_structure, neighbours
        ),
        "absolute_index_purity_at_10": position_stats[
            "exact_index_purity_at_10"
        ],
        "nearest_neighbor_exact_index": position_stats[
            "nearest_neighbor_exact_index"
        ],
        "same_protein_fraction_at_10": same_protein_fraction(
            latent_normalized, protein_index
        ),
        "tsne_absolute_index_purity_at_10": tsne_position_stats[
            "exact_index_purity_at_10"
        ],
        "tsne_nearest_neighbor_exact_index": tsne_position_stats[
            "nearest_neighbor_exact_index"
        ],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    scatter_panel(
        axes[0], embedding, amino_acids, view="aa20", annotate_aa=False
    )
    annotate_amino_acids(axes[0], embedding, amino_acids)
    axes[0].set_title(
        "Canonical LeJEPA V1 EMA (10k) — amino acid\n"
        f"rank={row['effective_rank']:.2f}/8; "
        f"cross-protein AA@10={row['aa_purity_at_10_cross_protein']:.3f}",
        fontsize=10,
    )
    index_scatter = axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=residue_index + 1,
        cmap="turbo",
        vmin=1,
        vmax=251,
        s=3,
        alpha=0.68,
        linewidths=0,
        rasterized=True,
    )
    annotate_indices(axes[1], embedding, residue_index)
    axes[1].set_title(
        "Canonical LeJEPA V1 EMA (10k) — absolute chain index\n"
        f"latent exact-index@10={row['absolute_index_purity_at_10']:.3f}",
        fontsize=10,
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(index_scatter, ax=axes[1], fraction=0.045, pad=0.02)
    colorbar.set_label("absolute residue index within chain (one-based)")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {RESULT}")
    print(f"wrote {FIGURE}")
    print(row)


if __name__ == "__main__":
    main()
