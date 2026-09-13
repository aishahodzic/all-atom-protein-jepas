"""Matched raw-10k t-SNEs for the protein/residue x rotation crop ablation."""

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
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array  # noqa: E402
import run_tsne as run_tsne_module  # noqa: E402
from run_tsne import coordinate_path, run as run_tsne  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


SPECS = {
    "canonical-lejepa-middle-lr4e4-raw-step10000": {
        "path": REPO / "StructTokenBench/100k-checkpoints/canonical-lejepa/lr-ablation/canonical_lejepa_protein_middle_lr4e4_raw10k_stb.ckpt",
        "source_sha256": "c9a7b99204a2ba6324bd4052a3682082354740b8cbef9872c7d8f2bb18c6541c",
        "label": "Per-protein, no rotation",
    },
    "canonical-lejepa-jarvis-rotation-raw-step10000": {
        "path": REPO / "StructTokenBench/100k-checkpoints/canonical-lejepa/jarvis/rotation/CLJP_step-00010000-raw-stb.ckpt",
        "source_sha256": "06bec76f5bf695ef930f494cdc5449c34375854b91fe53f7876bf74edd897607",
        "label": "Per-protein + rotation",
    },
    "canonical-lejepa-jarvis-per-residue-raw-step10000": {
        "path": REPO / "StructTokenBench/100k-checkpoints/canonical-lejepa/jarvis/per-residue/CLJP_step-00010000-raw-stb.ckpt",
        "source_sha256": "c010ba58ea65016497c79cf17b16341c22a919df36124310e1ae11fef3e94871",
        "label": "Per-residue, no rotation",
    },
    "canonical-lejepa-residue-perview-rot-middle-lr-raw-step10000": {
        "path": REPO / "store/canonical_lejepa_crop_fixed96_residue_perview_sigreg_b96_independent_locals_k4_rotations_lr4e4_cosine_30k/evaluation/canonical-lejepa-residue-perview-rot-middle-lr-step10000-raw-staged.ckpt",
        "source_sha256": "999a3773bb2a6c00be8c16d5daf86dcf30c1577399434a6fe7a1bc70ea35f004",
        "label": "Per-residue + rotation",
    },
}
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
EXPECTED_PROTEINS = 5_224
EXPECTED_RESIDUES = 748_402
EXPECTED_SAMPLE_SHA256 = "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"
OUTPUT_TAG = "canonical-raw10-middle-lr-protein-residue-rotation-2x2"
POSITION_FIGURE = HERE / "figures/canonical_lejepa/tsne_canonical_raw10_middle-lr_protein-residue-rotation-2x2_position.png"
MANIFEST = HERE / "results/tsne_canonical_raw10_middle-lr_protein-residue-rotation-2x2.json"


def register() -> None:
    for checkpoint_id, spec in SPECS.items():
        checkpoint_path = pathlib.Path(spec["path"])
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        provenance = checkpoint.get("canonical_lejepa_evaluation", {})
        expected = {
            "encoder_source": "raw",
            "source_global_step": 10_000,
            "source_checkpoint_sha256": spec["source_sha256"],
            "latent_dim": 8,
        }
        assert all(provenance.get(k) == v for k, v in expected.items()), (
            checkpoint_id,
            provenance,
        )
        state = checkpoint.get("state_dict", {})
        assert sum(k.startswith("online_encoder.") for k in state) == 381
        entry = (str(checkpoint_path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry
        run_tsne_module.ALL_CHECKPOINTS[checkpoint_id] = entry
        run_tsne_module.SHORT_LABEL[checkpoint_id] = str(spec["label"])


def matched_selection(checkpoint_ids: list[str]):
    labels = load_labels()
    caches = {name: load_encoder_latents(name) for name in checkpoint_ids}
    reference = set(caches[checkpoint_ids[0]])
    for name in checkpoint_ids[1:]:
        if set(caches[name]) != reference:
            raise RuntimeError(f"protein cohort differs for {name}")
    protein_ids = common_protein_ids(caches)
    selection = CombinedSelection(protein_ids, labels)
    if len(protein_ids) != EXPECTED_PROTEINS or selection.n != EXPECTED_RESIDUES:
        raise RuntimeError(
            f"frozen cohort changed: proteins={len(protein_ids)}, residues={selection.n}"
        )
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    physical = np.concatenate(
        [
            np.asarray([f"{protein_id}:{int(i)}" for i in indices], dtype=object)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    digest = hashlib.sha256("\n".join(physical.tolist()).encode()).hexdigest()
    if digest != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(f"frozen physical sample changed: {digest}")
    return labels, caches, selection, sampled, digest


def render_position(checkpoint_ids, labels, caches, selection, sampled) -> None:
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate(
        [
            np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    normalized_position = (residue_index + 1.0) / lengths
    colors = (
        (residue_index + 1.0, "turbo", 1.0, 256.0, "absolute residue index i"),
        (normalized_position, "viridis", 0.0, 1.0, "normalized position i/L"),
        (lengths, "magma", 50.0, 256.0, "source chain length L"),
    )
    fig, axes = plt.subplots(len(checkpoint_ids), 3, figsize=(13.2, 3.7 * len(checkpoint_ids)))
    for row, checkpoint_id in enumerate(checkpoint_ids):
        # Build the matched latent matrix as an additional residue-count check.
        latent = selection.build_X(caches[checkpoint_id])[sampled]
        assert latent.shape == (N_POINTS, 8)
        embedding = np.load(
            coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED),
            allow_pickle=False,
        )["embedding"]
        for column, (values, cmap, vmin, vmax, color_label) in enumerate(colors):
            ax = axes[row, column]
            image = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=values, cmap=cmap,
                vmin=vmin, vmax=vmax, s=2.7, alpha=0.72, linewidths=0,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.015).set_label(
                color_label, fontsize=8
            )
        axes[row, 0].set_ylabel(str(SPECS[checkpoint_id]["label"]), fontsize=10)
    fig.suptitle(
        "Raw 10k: matched t-SNE position and chain-length audit\n"
        "Identical 15,000 physical residues; independently fitted cosine t-SNEs",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    POSITION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSITION_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    register()
    checkpoint_ids = list(SPECS)
    extract(checkpoint_ids=checkpoint_ids, device="cuda")
    run_tsne(
        checkpoint_ids=checkpoint_ids,
        n_points=N_POINTS,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
        output_tag=OUTPUT_TAG,
        grid_ncols=3,
        include_purity=False,
        show_suptitle=True,
    )
    labels, caches, selection, sampled, sample_sha = matched_selection(checkpoint_ids)
    render_position(checkpoint_ids, labels, caches, selection, sampled)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "checkpoint_ids": checkpoint_ids,
                "encoder_source": "raw",
                "source_step": 10_000,
                "proteins": EXPECTED_PROTEINS,
                "residue_pool": EXPECTED_RESIDUES,
                "sampled_residues": N_POINTS,
                "physical_sample_sha256": sample_sha,
                "biochemical_output_tag": OUTPUT_TAG,
                "position_figure": str(POSITION_FIGURE),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {POSITION_FIGURE}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
