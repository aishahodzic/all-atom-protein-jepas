"""Chain-length-only t-SNE for the stopped pair-only residue-K4 run."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
for path in (REPO, HERE, LINEAR, KNN):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)
from analyze_current_four_health import (  # noqa: E402
    external_neighbours,
    neighbour_spearman,
)
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from run_tsne import load_or_fit  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINT_ID = "canonical-lejepa-residue-k4-pair-only-raw-step1500"
RUN_NAME = (
    "canonical_lejepa_v1_residue_k4_nocrop_"
    "pair_only_position_minvalid4_10k"
)
STEP = 1_500
RUN_ROOT = REPO / "store" / RUN_NAME
SOURCE = RUN_ROOT / "evaluation/snapshots/raw-step00001500-full-state.ckpt"
CONFIG = RUN_ROOT / "checkpoints" / f"exp_config_{RUN_NAME}.json"
STAGED = RUN_ROOT / "evaluation/pair-only-raw-step1500-staged.ckpt"
FIGURE = HERE / (
    "figures/canonical_lejepa/"
    "tsne_canonical_residue_k4_pair_only_raw_1500_chain-length.png"
)
RESULT = HERE / (
    "results/canonical_residue_k4_pair_only_raw_1500_chain_length.json"
)

N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
EXPECTED_PROTEINS = 5_224
EXPECTED_RESIDUE_POOL = 748_402
EXPECTED_PHYSICAL_SAMPLE_SHA256 = (
    "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"
)


def register_raw() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    if not STAGED.is_file():
        STAGED.parent.mkdir(parents=True, exist_ok=True)
        stage_structtokenbench_checkpoint(SOURCE, STAGED, source="raw")

    staged = torch.load(STAGED, map_location="cpu", weights_only=False, mmap=True)
    provenance = staged.get("canonical_lejepa_evaluation", {})
    expected = {
        "encoder_source": "raw",
        "source_global_step": STEP,
        "latent_dim": 8,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid staged raw checkpoint: {mismatches}")

    checkpoints.JEPA_ENCODER_CFG = CONFIG
    entry = (str(STAGED), "jepa")
    checkpoints.CHECKPOINTS[CHECKPOINT_ID] = entry
    checkpoints.ALL_CHECKPOINTS[CHECKPOINT_ID] = entry


def main() -> None:
    register_raw()
    extract(checkpoint_ids=[CHECKPOINT_ID], device="cuda")

    labels = load_labels()
    latent_cache = load_encoder_latents(CHECKPOINT_ID)
    protein_ids = common_protein_ids({CHECKPOINT_ID: latent_cache})
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
            "physical residue sample differs from the frozen reference: "
            f"{physical_sample_sha256}"
        )

    latent = selection.build_X(latent_cache)[sampled].astype(np.float32)
    lengths = np.concatenate(
        [
            np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    protein_index = selection.protein_idx[sampled]
    neighbours = external_neighbours(latent, protein_index)
    length_rho, _ = neighbour_spearman(
        np.log(lengths), np.ones(N_POINTS, dtype=bool), neighbours
    )

    embedding, coordinate_file, input_sha256 = load_or_fit(
        CHECKPOINT_ID,
        latent,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=lengths,
        cmap="magma",
        vmin=float(lengths.min()),
        vmax=float(lengths.max()),
        s=2.8,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Pair-only residue K=4, raw step 1,500\n"
        f"chain length; cross-protein kNN rho={length_rho:.3f}",
        fontsize=11,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.015)
    colorbar.set_label("chain length L")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    result = {
        "description": "Chain-length-only t-SNE of stopped pair-only residue-K4 run",
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_path": str(SOURCE),
        "staged_checkpoint": str(STAGED),
        "encoder_source": "raw",
        "training_step": STEP,
        "proteins": len(protein_ids),
        "residue_pool": selection.n,
        "sampled_residues": len(sampled),
        "sample_seed": SEED,
        "physical_sample_sha256": physical_sample_sha256,
        "latent_dim": int(latent.shape[1]),
        "latent_input_sha256": input_sha256,
        "chain_length_min": float(lengths.min()),
        "chain_length_max": float(lengths.max()),
        "log_chain_length_knn10_spearman": float(length_rho),
        "tsne": {
            "metric": "cosine",
            "perplexity": PERPLEXITY,
            "max_iter": MAX_ITER,
            "init": "random",
            "coordinate_file": str(coordinate_file),
        },
        "figure": str(FIGURE),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
