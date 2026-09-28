"""Matched canonical-LeJEPA position, length, and residue-signal audit.

The default preserves the original 10k/30k comparison and its artifact names.
``--include-50k`` extends that exact frozen cohort and residue sample to 50k;
``--include-70k`` extends it through 70k (and therefore also includes 50k).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr


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
    aligned_property,
    external_neighbours,
    neighbour_spearman,
    purity,
    same_protein_fraction,
    spectrum_metrics,
)
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import knn_exact_index_stats, local_index_array  # noqa: E402
from run_tsne import coordinate_path, run as run_tsne, scatter_panel  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = {
    "canonical-lejepa-v1-ema-step10000": REPO
    / "store/canonical_lejepa_v1_100k/checkpoints/step-00010000.ckpt",
    "canonical-lejepa-v1-ema-step30000": REPO
    / "store/canonical_lejepa_v1_100k/checkpoints/step-00030000.ckpt",
}
CHECKPOINT_STEPS = {
    "canonical-lejepa-v1-ema-step10000": 10_000,
    "canonical-lejepa-v1-ema-step30000": 30_000,
}
STAGED = {
    checkpoint_id: REPO
    / "store/canonical_lejepa_v1_100k/evaluation"
    / f"canonical-lejepa-v1-step{step}-ema-staged.ckpt"
    for checkpoint_id, step in (
        ("canonical-lejepa-v1-ema-step10000", "10000"),
        ("canonical-lejepa-v1-ema-step30000", "30000"),
    )
}
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
K = 10
LENGTH_BIN_WIDTH = 16
EXPECTED_PROTEINS = 5_224
EXPECTED_RESIDUE_POOL = 748_402
EXPECTED_PHYSICAL_SAMPLE_SHA256 = (
    "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"
)
RESULT = HERE / "results/canonical_lejepa_v1_ema_10k_30k_position.csv"
MANIFEST = HERE / "results/canonical_lejepa_v1_ema_10k_30k_position.json"
POSITION_FIGURE = HERE / (
    "figures/canonical_lejepa/"
    "tsne_canonical_lejepa_v1_ema_10k-v-30k_position-and-length.png"
)
RESIDUE_FIGURE = HERE / (
    "figures/canonical_lejepa/"
    "tsne_canonical_lejepa_v1_ema_10k-v-30k_aa-and-ss.png"
)
ANALYSIS_LABEL = "Canonical LeJEPA V1"
OUTPUT_TAG = "canonical-lejepa-v1-ema-10k-v-30k"
MANIFEST_DESCRIPTION = (
    "Matched canonical LeJEPA V1 EMA 10k/30k position and length audit"
)
ENCODER_SOURCE = "ema"
ENCODER_LABEL = "EMA"
ROW_LABELS: dict[str, str] = {}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_staged_ema(
    checkpoint_path: pathlib.Path,
    staged_path: pathlib.Path,
    expected_step: int,
) -> None:
    """Require an EMA artifact validated/staged by the preceding FlexRMSF job."""
    if not staged_path.is_file():
        raise FileNotFoundError(
            f"required staged EMA checkpoint is absent: {staged_path}; "
            f"run this analysis after the {expected_step // 1000}k FlexRMSF job"
        )
    staged = torch.load(staged_path, map_location="cpu", weights_only=False)
    provenance = staged.get("canonical_lejepa_evaluation", {})
    expected = {
        "encoder_source": "ema",
        "source_global_step": expected_step,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"staged EMA provenance does not match {checkpoint_path}: {mismatches}"
        )


def register() -> None:
    for checkpoint_id, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        staged_path = STAGED[checkpoint_id]
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if CHECKPOINT_STEPS[checkpoint_id] in {50_000, 70_000}:
            validate_staged_ema(
                checkpoint_path, staged_path, CHECKPOINT_STEPS[checkpoint_id]
            )
        elif not staged_path.is_file():
            stage_structtokenbench_checkpoint(
                checkpoint_path, staged_path, source="ema"
            )
        entry = (str(staged_path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman correlation of x/y after linearly residualizing ranked control."""
    ranked_control = rankdata(control).astype(np.float64)
    design = np.column_stack((np.ones(len(control)), ranked_control))

    def residual(values: np.ndarray) -> np.ndarray:
        ranked = rankdata(values).astype(np.float64)
        beta, *_ = np.linalg.lstsq(design, ranked, rcond=None)
        return ranked - design @ beta

    return float(np.corrcoef(residual(x), residual(y))[0, 1])


def within_length_bin_neighbours(
    values: np.ndarray,
    protein_index: np.ndarray,
    lengths: np.ndarray,
    *,
    width: int = LENGTH_BIN_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-protein cosine kNN restricted to a narrow chain-length bin."""
    values = normalized(values)
    length_bins = ((lengths - 1) // width).astype(np.int64)
    result = np.full((len(values), K), -1, dtype=np.int64)
    usable = np.zeros(len(values), dtype=bool)
    for length_bin in np.unique(length_bins):
        members = np.flatnonzero(length_bins == length_bin)
        if len(members) <= K:
            continue
        query_k = min(len(members), 128)
        _, local_candidates = cKDTree(values[members]).query(
            values[members], k=query_k, workers=-1
        )
        if query_k == 1:
            local_candidates = local_candidates[:, None]
        for local_row, candidates in enumerate(local_candidates):
            row = members[local_row]
            candidates = members[np.atleast_1d(candidates)]
            external = candidates[protein_index[candidates] != protein_index[row]]
            if len(external) >= K:
                result[row] = external[:K]
                usable[row] = True
    return result, usable


def render_position_grid(
    embeddings: dict[str, np.ndarray],
    residue_index: np.ndarray,
    normalized_position: np.ndarray,
    lengths: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    metrics = {str(row["checkpoint"]): row for row in rows}
    fig, axes = plt.subplots(
        len(CHECKPOINTS), 3,
        figsize=(13.2, 4.0 * len(CHECKPOINTS)),
        squeeze=False,
    )
    color_specs = (
        (residue_index + 1, "turbo", 1.0, float((residue_index + 1).max()), "raw index i"),
        (normalized_position, "viridis", 0.0, 1.0, "normalized position i/L"),
        (lengths, "magma", float(lengths.min()), float(lengths.max()), "chain length L"),
    )
    for row_index, checkpoint_id in enumerate(CHECKPOINTS):
        embedding = embeddings[checkpoint_id]
        row = metrics[checkpoint_id]
        for column, (colors, cmap, vmin, vmax, label) in enumerate(color_specs):
            ax = axes[row_index, column]
            image = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap,
                vmin=vmin, vmax=vmax, s=2.8, alpha=0.72, linewidths=0,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column == 0:
                title = (
                    f"raw index; rho={float(row['raw_index_knn10_spearman']):.3f}; "
                    f"partial rho={float(row['raw_index_partial_length_spearman']):.3f}"
                )
            elif column == 1:
                title = (
                    "normalized i/L; "
                    f"rho={float(row['normalized_position_knn10_spearman']):.3f}"
                )
            else:
                title = (
                    "chain length; "
                    f"rho={float(row['log_chain_length_knn10_spearman']):.3f}"
                )
            ax.set_title(title, fontsize=9)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.015)
            colorbar.set_label(label, fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
        axes[row_index, 0].set_ylabel(
            ROW_LABELS.get(
                checkpoint_id,
                f"{ENCODER_LABEL} {CHECKPOINT_STEPS[checkpoint_id] // 1000}k",
            ),
            fontsize=11,
        )
    fig.suptitle(
        f"{ANALYSIS_LABEL}: matched residues and t-SNE settings",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    POSITION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSITION_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_residue_grid(
    embeddings: dict[str, np.ndarray],
    amino_acids: np.ndarray,
    secondary_structure: np.ndarray,
) -> None:
    fig, axes = plt.subplots(
        len(CHECKPOINTS), 2,
        figsize=(9.2, 4.0 * len(CHECKPOINTS)),
        squeeze=False,
    )
    ss_names = ("helix", "sheet", "loop")
    ss_colors = ("#d62728", "#1f77b4", "#7f7f7f")
    for row_index, checkpoint_id in enumerate(CHECKPOINTS):
        embedding = embeddings[checkpoint_id]
        scatter_panel(
            axes[row_index, 0], embedding, amino_acids,
            view="aa20", annotate_aa=False,
        )
        axes[row_index, 0].set_title("amino-acid identity", fontsize=9)
        for ss_index, (name, color) in enumerate(zip(ss_names, ss_colors)):
            members = secondary_structure == ss_index
            axes[row_index, 1].scatter(
                embedding[members, 0], embedding[members, 1],
                s=2.8, alpha=0.62, linewidths=0, c=color,
                label=name, rasterized=True,
            )
        axes[row_index, 1].set_xticks([])
        axes[row_index, 1].set_yticks([])
        for spine in axes[row_index, 1].spines.values():
            spine.set_visible(False)
        axes[row_index, 1].set_title("secondary structure", fontsize=9)
        axes[row_index, 1].legend(frameon=False, fontsize=7, markerscale=2)
        axes[row_index, 0].set_ylabel(
            ROW_LABELS.get(
                checkpoint_id,
                f"{ENCODER_LABEL} {CHECKPOINT_STEPS[checkpoint_id] // 1000}k",
            ),
            fontsize=11,
        )
    fig.suptitle(
        f"{ANALYSIS_LABEL}: does residue-level organization emerge?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    RESIDUE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESIDUE_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-50k",
        action="store_true",
        help=(
            "extend the original matched 10k/30k audit to the immutable 50k "
            "EMA checkpoint without overwriting 10k/30k artifacts"
        ),
    )
    parser.add_argument(
        "--include-70k",
        action="store_true",
        help=(
            "extend the matched audit through the immutable 70k EMA checkpoint "
            "(also includes 50k) without overwriting earlier artifacts"
        ),
    )
    parser.add_argument(
        "--residue-ab-20k",
        action="store_true",
        help="compare the matched protein-centered SIGReg residue A/B EMA checkpoints at 20k",
    )
    return parser.parse_args()


def configure_residue_ab_20k() -> None:
    global RESULT, MANIFEST, POSITION_FIGURE, RESIDUE_FIGURE
    global OUTPUT_TAG, MANIFEST_DESCRIPTION, ANALYSIS_LABEL

    CHECKPOINTS.clear()
    CHECKPOINT_STEPS.clear()
    STAGED.clear()
    runs = {
        "canonical-lejepa-residue-iterative-protein-centered-sigreg-ema-step20000": (
            "canonical_lejepa_residue_iterative_protein_centered_sigreg_2k",
            "canonical-lejepa-residue-iterative-protein-centered-sigreg-step20000-ema-staged.ckpt",
        ),
        "canonical-lejepa-residue-visible-only-protein-centered-sigreg-ema-step20000": (
            "canonical_lejepa_residue_visible_only_protein_centered_sigreg_2k",
            "canonical-lejepa-residue-visible-only-protein-centered-sigreg-step20000-ema-staged.ckpt",
        ),
    }
    for checkpoint_id, (run_name, stage_name) in runs.items():
        run_root = REPO / "store" / run_name
        CHECKPOINTS[checkpoint_id] = run_root / "checkpoints/step-00020000.ckpt"
        CHECKPOINT_STEPS[checkpoint_id] = 20_000
        STAGED[checkpoint_id] = run_root / "evaluation" / stage_name
    RESULT = HERE / "results/canonical_lejepa_residue_ab_protein_centered_sigreg_ema_20k_position.csv"
    MANIFEST = HERE / "results/canonical_lejepa_residue_ab_protein_centered_sigreg_ema_20k_position.json"
    POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_residue_ab_protein_centered_sigreg_ema_20k_position-and-length.png"
    )
    RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_residue_ab_protein_centered_sigreg_ema_20k_aa-and-ss.png"
    )
    OUTPUT_TAG = "canonical-lejepa-residue-ab-protein-centered-sigreg-ema-20k"
    ANALYSIS_LABEL = "Canonical residue LeJEPA A/B, protein-centered SIGReg"
    MANIFEST_DESCRIPTION = (
        "Matched canonical residue LeJEPA A/B protein-centered SIGReg EMA 20k "
        "position and length audit"
    )


def configure_50k_extension() -> None:
    global RESULT, MANIFEST, POSITION_FIGURE, RESIDUE_FIGURE
    global OUTPUT_TAG, MANIFEST_DESCRIPTION

    checkpoint_id = "canonical-lejepa-v1-ema-step50000"
    CHECKPOINTS[checkpoint_id] = (
        REPO / "store/canonical_lejepa_v1_100k/checkpoints/step-00050000.ckpt"
    )
    CHECKPOINT_STEPS[checkpoint_id] = 50_000
    STAGED[checkpoint_id] = (
        REPO
        / "store/canonical_lejepa_v1_100k/evaluation"
        / "canonical-lejepa-v1-step50000-ema-staged.ckpt"
    )
    RESULT = (
        HERE / "results/canonical_lejepa_v1_ema_10k_30k_50k_position.csv"
    )
    MANIFEST = (
        HERE / "results/canonical_lejepa_v1_ema_10k_30k_50k_position.json"
    )
    POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_v1_ema_10k-v-30k-v-50k_position-and-length.png"
    )
    RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_v1_ema_10k-v-30k-v-50k_aa-and-ss.png"
    )
    OUTPUT_TAG = "canonical-lejepa-v1-ema-10k-v-30k-v-50k"
    MANIFEST_DESCRIPTION = (
        "Matched canonical LeJEPA V1 EMA 10k/30k/50k position and length audit"
    )


def configure_70k_extension() -> None:
    global RESULT, MANIFEST, POSITION_FIGURE, RESIDUE_FIGURE
    global OUTPUT_TAG, MANIFEST_DESCRIPTION

    configure_50k_extension()
    checkpoint_id = "canonical-lejepa-v1-ema-step70000"
    CHECKPOINTS[checkpoint_id] = (
        REPO / "store/canonical_lejepa_v1_100k/checkpoints/step-00070000.ckpt"
    )
    CHECKPOINT_STEPS[checkpoint_id] = 70_000
    STAGED[checkpoint_id] = (
        REPO
        / "store/canonical_lejepa_v1_100k/evaluation"
        / "canonical-lejepa-v1-step70000-ema-staged.ckpt"
    )
    RESULT = (
        HERE / "results/canonical_lejepa_v1_ema_10k_30k_50k_70k_position.csv"
    )
    MANIFEST = (
        HERE / "results/canonical_lejepa_v1_ema_10k_30k_50k_70k_position.json"
    )
    POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_v1_ema_10k-v-30k-v-50k-v-70k_position-and-length.png"
    )
    RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_v1_ema_10k-v-30k-v-50k-v-70k_aa-and-ss.png"
    )
    OUTPUT_TAG = "canonical-lejepa-v1-ema-10k-v-30k-v-50k-v-70k"
    MANIFEST_DESCRIPTION = (
        "Matched canonical LeJEPA V1 EMA 10k/30k/50k/70k position and length audit"
    )


def main() -> None:
    args = parse_args()
    if args.residue_ab_20k:
        configure_residue_ab_20k()
    elif args.include_70k:
        configure_70k_extension()
    elif args.include_50k:
        configure_50k_extension()
    register()
    checkpoint_ids = list(CHECKPOINTS)
    extract(checkpoint_ids=checkpoint_ids, device="cuda")

    import run_tsne as run_tsne_module

    run_tsne_module.SHORT_LABEL.update(
        {
            "canonical-lejepa-v1-ema-step10000": "Canonical LeJEPA V1 EMA (10k)",
            "canonical-lejepa-v1-ema-step30000": "Canonical LeJEPA V1 EMA (30k)",
            "canonical-lejepa-v1-ema-step50000": "Canonical LeJEPA V1 EMA (50k)",
            "canonical-lejepa-v1-ema-step70000": "Canonical LeJEPA V1 EMA (70k)",
            "canonical-lejepa-residue-iterative-protein-centered-sigreg-ema-step20000": "A: iterative EMA (20k)",
            "canonical-lejepa-residue-visible-only-protein-centered-sigreg-ema-step20000": "B: visible-only EMA (20k)",
        }
    )
    run_tsne(
        checkpoint_ids=checkpoint_ids,
        n_points=N_POINTS,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
        output_tag=OUTPUT_TAG,
        grid_ncols=len(checkpoint_ids),
        include_purity=False,
        show_suptitle=False,
    )

    labels = load_labels()
    latent_caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in checkpoint_ids
    }
    reference_keys = set(latent_caches[checkpoint_ids[0]])
    for checkpoint_id in checkpoint_ids[1:]:
        if set(latent_caches[checkpoint_id]) != reference_keys:
            raise RuntimeError(
                f"{checkpoint_id} does not contain exactly the same protein IDs "
                "as the matched 10k reference"
            )
    for protein_id in sorted(reference_keys):
        lengths_by_checkpoint = {
            checkpoint_id: int(latent_caches[checkpoint_id][protein_id].shape[0])
            for checkpoint_id in checkpoint_ids
        }
        if len(set(lengths_by_checkpoint.values())) != 1:
            raise RuntimeError(
                f"latent residue count differs for {protein_id}: "
                f"{lengths_by_checkpoint}"
            )
    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, labels)
    if len(protein_ids) != EXPECTED_PROTEINS or selection.n != EXPECTED_RESIDUE_POOL:
        raise RuntimeError(
            "matched cohort changed: "
            f"proteins={len(protein_ids)} (expected {EXPECTED_PROTEINS}), "
            f"residue_pool={selection.n} (expected {EXPECTED_RESIDUE_POOL})"
        )
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    physical_sample = np.concatenate(
        [
            np.asarray(
                [f"{protein_id}:{int(index)}" for index in indices],
                dtype=object,
            )
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    physical_sample_sha256 = hashlib.sha256(
        "\n".join(physical_sample.tolist()).encode()
    ).hexdigest()
    if physical_sample_sha256 != EXPECTED_PHYSICAL_SAMPLE_SHA256:
        raise RuntimeError(
            "physical residue sample differs from the frozen 10k reference: "
            f"{physical_sample_sha256}"
        )
    amino_acids = selection.aa[sampled]
    secondary_structure = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate(
        [
            np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    # One-based i/L, exactly matching the declared diagnostic.
    normalized_position = (residue_index + 1.0) / lengths
    log_length = np.log(lengths)
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]

    rows: list[dict[str, object]] = []
    embeddings: dict[str, np.ndarray] = {}
    for checkpoint_id in checkpoint_ids:
        latent = selection.build_X(latent_caches[checkpoint_id])[sampled].astype(
            np.float32
        )
        neighbours = external_neighbours(latent, protein_index)
        exact_index_stats = knn_exact_index_stats(
            normalized(latent), residue_index
        )
        raw_prediction = residue_index[neighbours].mean(axis=1)
        within_bin_neighbours, within_bin_usable = within_length_bin_neighbours(
            latent, protein_index, lengths
        )
        embedding = np.load(
            coordinate_path(
                checkpoint_id, N_POINTS, PERPLEXITY, SEED
            ),
            allow_pickle=False,
        )["embedding"]
        embeddings[checkpoint_id] = embedding

        raw_rho, _ = neighbour_spearman(
            residue_index, np.ones(N_POINTS, dtype=bool), neighbours
        )
        normalized_rho, _ = neighbour_spearman(
            normalized_position, np.ones(N_POINTS, dtype=bool), neighbours
        )
        length_rho, _ = neighbour_spearman(
            log_length, np.ones(N_POINTS, dtype=bool), neighbours
        )
        rsa_rho, rsa_n = neighbour_spearman(rsa, rsa_valid, neighbours)
        within_raw_prediction = residue_index[
            within_bin_neighbours[within_bin_usable]
        ].mean(axis=1)
        within_raw_rho = spearmanr(
            residue_index[within_bin_usable], within_raw_prediction
        ).statistic
        row: dict[str, object] = {
            "checkpoint": checkpoint_id,
            "checkpoint_path": str(CHECKPOINTS[checkpoint_id]),
            "encoder_source": ENCODER_SOURCE,
            "training_steps": CHECKPOINT_STEPS[checkpoint_id],
            "proteins": len(protein_ids),
            "residue_pool": selection.n,
            "sampled_residues": len(sampled),
            "latent_dim": latent.shape[1],
            **spectrum_metrics(latent),
            "aa_purity_at_10_cross_protein": purity(amino_acids, neighbours),
            "ss_purity_at_10_cross_protein": purity(
                secondary_structure, neighbours
            ),
            "rsa_knn10_spearman": rsa_rho,
            "rsa_n": rsa_n,
            "raw_index_knn10_spearman": raw_rho,
            "normalized_position_knn10_spearman": normalized_rho,
            "log_chain_length_knn10_spearman": length_rho,
            "raw_index_partial_length_spearman": partial_spearman(
                residue_index, raw_prediction, log_length
            ),
            "raw_index_within_16aa_length_bin_knn10_spearman": float(
                within_raw_rho
            ),
            "raw_index_within_16aa_length_bin_n": int(within_bin_usable.sum()),
            "exact_index_at_10_cross_protein": float(
                (residue_index[neighbours] == residue_index[:, None]).mean()
            ),
            "exact_index_at_10_all_neighbours": exact_index_stats[
                "exact_index_purity_at_10"
            ],
            "nearest_neighbour_exact_index_all_neighbours": exact_index_stats[
                "nearest_neighbor_exact_index"
            ],
            "normalized_position_quintile_purity_at_10_cross_protein": float(
                (
                    np.minimum((normalized_position * 5).astype(int), 4)[neighbours]
                    == np.minimum((normalized_position * 5).astype(int), 4)[:, None]
                ).mean()
            ),
            "same_protein_fraction_at_10": same_protein_fraction(
                normalized(latent), protein_index
            ),
        }
        rows.append(row)

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "description": MANIFEST_DESCRIPTION,
        "encoder_source": f"{ENCODER_LABEL} (canonical evaluation encoder)",
        "selection": {
            "proteins": len(protein_ids),
            "aa_ss_valid_residue_pool": selection.n,
            "sampled_residues": len(sampled),
            "seed": SEED,
            "same_physical_residues": True,
            "physical_sample_sha256": physical_sample_sha256,
        },
        "tsne": {
            "metric": "cosine",
            "perplexity": PERPLEXITY,
            "max_iter": MAX_ITER,
            "init": "random",
            "same_settings_but_independent_fit_per_checkpoint": True,
        },
        "position": {
            "raw_index": "one-based local tensor-row ordinal within chain",
            "normalized_position": "one-based i/L",
            "length_control": [
                "partial Spearman after residualizing ranked log(chain length)",
                f"cross-protein kNN restricted to {LENGTH_BIN_WIDTH}-residue length bins",
            ],
        },
        "checkpoints": [
            {
                "id": checkpoint_id,
                "source": str(CHECKPOINTS[checkpoint_id]),
                "staged": str(STAGED[checkpoint_id]),
            }
            for checkpoint_id in checkpoint_ids
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    render_position_grid(
        embeddings, residue_index, normalized_position, lengths, rows
    )
    render_residue_grid(
        embeddings, amino_acids, secondary_structure
    )
    print(f"wrote {RESULT}")
    print(f"wrote {MANIFEST}")
    print(f"wrote {POSITION_FIGURE}")
    print(f"wrote {RESIDUE_FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
