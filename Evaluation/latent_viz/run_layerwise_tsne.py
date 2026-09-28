"""Render layerwise t-SNE panels from 768D post-transformer-block states.

The extraction deliberately keeps only one fixed, label-valid sample of
residues rather than caching every residue from every block.  This makes the
physical residues identical across blocks and models while keeping the cache
small enough for the project filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from proteinfoundation.utils.dense_padding_data_loader import (
    dense_padded_from_data_list,
)

from Evaluation.latent_viz.run_tsne import (
    input_digest,
    project_tsne,
    scatter_panel,
    view_labels_colors,
)
from Evaluation.latent_viz.viz import CombinedSelection, sample_indices
from Evaluation.linear_probing import checkpoints
from Evaluation.linear_probing.data_pool import (
    build_protein_pool,
    eval_transform_compose,
)


REPO = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO / "Evaluation/linear_probing/latents/labels.pt"
RESULTS_ROOT = REPO / "Evaluation/latent_viz/results/layerwise_tsne"
FIGURE_ROOT = REPO / "iclr2027/figures"
N_POINTS = 15_000
SEED = 42
PERPLEXITY = 30.0
MAX_ITER = 1_000
HIDDEN_WIDTH = 768


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    checkpoint: Path
    checkpoint_sha256: str
    blocks: tuple[int, ...]
    output_name: str
    source: str = "online"


MODEL_SPECS = {
    "uncropped-le-jepa": ModelSpec(
        key="layerwise-tsne-uncropped-le-jepa",
        title="Uncropped LeJEPA",
        checkpoint=REPO
        / "store/canonical_lejepa_v1_100k/evaluation/"
        "canonical-lejepa-v1-step100000-raw-staged.ckpt",
        checkpoint_sha256="28f93a4224323b7366243265abd7721f2dffaaf6f8c9fd4aa4767788fd5d2bac",
        blocks=tuple(range(1, 13)),
        output_name="all-tsne-per-block-uncropped-le-jepa.png",
    ),
    "jepa-rand70": ModelSpec(
        key="layerwise-tsne-jepa-rand70",
        title="JEPA random-mask 70%",
        checkpoint=REPO
        / "StructTokenBench/100k-checkpoints/jepa/random_masking_70pct/100k.ckpt",
        checkpoint_sha256="63183c74a02386fdc9aaca3ff619e9f8e83705a79340610b1f4a064ba547098c",
        blocks=(1, 6, 12),
        output_name="all-tsne-per-block-jepa-rand70.png",
    ),
    "cropped-protein-no-rotation": ModelSpec(
        key="layerwise-tsne-cropped-protein-no-rotation",
        title="Cropped LeJEPA, per-protein, no rotation",
        checkpoint=REPO
        / "StructTokenBench/100k-checkpoints/canonical-lejepa/lr-ablation/"
        "canonical_lejepa_protein_middle_lr4e4_raw10k_stb.ckpt",
        checkpoint_sha256="61efd8bc9e2010792a4c6b76c2576e26d765d97d7c7c2eb0263725c0af77107d",
        blocks=(4, 6, 8, 12),
        output_name="all-tsne-per-block-cropped-per-protein-no-rotation.png",
    ),
    "cropped-residue-no-rotation": ModelSpec(
        key="layerwise-tsne-cropped-residue-no-rotation",
        title="Cropped LeJEPA, per-residue, no rotation",
        checkpoint=REPO
        / "store/canonical_lejepa_crop_fixed96_residue_perview_sigreg_b96_100k_relaunch/"
        "evaluation/canonical-lejepa-residue-perview-b96-relaunch-step10000-raw-staged.ckpt",
        checkpoint_sha256="781107febead25c3abd17fe2cce9f23e51a648ceb740e0487c101a0e6c3a95e3",
        blocks=(4, 6, 8, 12),
        output_name="all-tsne-per-block-cropped-per-residue-no-rotation.png",
    ),
    "cropped-residue-rotation": ModelSpec(
        key="layerwise-tsne-cropped-residue-rotation",
        title="Cropped LeJEPA, per-residue + rotation",
        checkpoint=REPO
        / "store/canonical_lejepa_crop_fixed96_residue_perview_sigreg_b96_"
        "independent_locals_k4_rotations_lr4e4_cosine_30k/evaluation/"
        "canonical-lejepa-residue-perview-rot-middle-lr-step10000-raw-staged.ckpt",
        checkpoint_sha256="a2937cdadc379deaec2be2fafe463440a7cc606cf3e1080580a2cddad5268733",
        blocks=(4, 6, 8, 12),
        output_name="all-tsne-per-block-cropped-per-residue-rotation.png",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_selection() -> tuple[CombinedSelection, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    labels = torch.load(LABELS_PATH, map_location="cpu", weights_only=False)
    selection = CombinedSelection(sorted(labels), labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)

    selected_by_protein: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    offset = 0
    for protein_id, valid_rows in selection.keys:
        stop = offset + len(valid_rows)
        sample_positions = np.flatnonzero((sampled >= offset) & (sampled < stop))
        if sample_positions.size:
            local_rows = valid_rows[sampled[sample_positions] - offset]
            selected_by_protein[protein_id] = (
                local_rows.astype(np.int64, copy=False),
                sample_positions.astype(np.int64, copy=False),
            )
        offset = stop
    if offset != selection.n or sum(len(x[0]) for x in selected_by_protein.values()) != len(sampled):
        raise AssertionError("sample-to-protein mapping is incomplete")
    return selection, sampled, selected_by_protein


def sample_sha256(selection: CombinedSelection, sampled: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(sampled, dtype=np.int64).view(np.uint8))
    digest.update(np.ascontiguousarray(selection.aa[sampled], dtype=np.int64).view(np.uint8))
    digest.update(np.ascontiguousarray(selection.ss[sampled], dtype=np.int64).view(np.uint8))
    for protein_id, valid_rows in selection.keys:
        digest.update(protein_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(valid_rows, dtype=np.int64).view(np.uint8))
    return digest.hexdigest()


class BlockCapture:
    def __init__(self, encoder: torch.nn.Module, blocks: tuple[int, ...]) -> None:
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for block in blocks:
            layer = encoder.transformer_layers[block - 1]
            self.handles.append(layer.register_forward_hook(self._hook(block)))

    def _hook(self, block: int):
        def hook(module, args, output):
            del module, args
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"block {block} produced {type(output)}, expected Tensor")
            self.values[block] = output.detach()

        return hook

    def clear(self) -> None:
        self.values.clear()

    def take_sample(self, blocks: tuple[int, ...], rows: torch.Tensor) -> torch.Tensor:
        if tuple(sorted(self.values)) != blocks:
            raise AssertionError(f"hook mismatch: captured={sorted(self.values)}, expected={blocks}")
        return torch.stack([self.values[block][0, rows] for block in blocks]).float().cpu()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_verified_encoder(spec: ModelSpec, device: str) -> tuple[torch.nn.Module, str]:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)
    checkpoint_hash = sha256_file(spec.checkpoint)
    if checkpoint_hash != spec.checkpoint_sha256:
        raise AssertionError(
            f"checkpoint SHA-256 mismatch for {spec.checkpoint}: {checkpoint_hash}"
        )
    checkpoints.ALL_CHECKPOINTS[spec.key] = (str(spec.checkpoint), "jepa")
    encoder = checkpoints.load_frozen_encoder(
        spec.key, device=device, encoder_source=spec.source
    )
    if len(encoder.transformer_layers) != 12 or encoder.token_dim != HIDDEN_WIDTH:
        raise AssertionError(
            f"unexpected encoder geometry: layers={len(encoder.transformer_layers)}, "
            f"width={encoder.token_dim}"
        )
    return encoder, checkpoint_hash


def extract_sampled_states(
    spec: ModelSpec,
    device: str,
    selected_by_protein: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, torch.Tensor], str, int]:
    encoder, checkpoint_hash = load_verified_encoder(spec, device)
    samples = {
        block: torch.empty((N_POINTS, HIDDEN_WIDTH), dtype=torch.float32)
        for block in spec.blocks
    }
    filled = np.zeros(N_POINTS, dtype=bool)
    pool = build_protein_pool()
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    missing = sorted(set(selected_by_protein) - set(pool_index))
    if missing:
        raise KeyError(f"{len(missing)} selected proteins absent from evaluation pool: {missing[:5]}")

    transform = eval_transform_compose()
    capture = BlockCapture(encoder, spec.blocks)
    selected_proteins = sorted(selected_by_protein)
    try:
        for ordinal, protein_id in enumerate(selected_proteins, start=1):
            local_rows, output_positions = selected_by_protein[protein_id]
            graph = transform(pool.dataset[pool_index[protein_id]])
            batch = dense_padded_from_data_list([graph])
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0].bool()
            batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
            if local_rows.max(initial=-1) >= batch["mask"].shape[1]:
                raise AssertionError(f"row mapping exceeds dense shape for {protein_id}")
            if not bool(batch["mask"][0, torch.from_numpy(local_rows)].all()):
                raise AssertionError(f"sample contains padded rows for {protein_id}")

            capture.clear()
            device_batch = batch.to(device)
            rows_device = torch.as_tensor(local_rows, dtype=torch.long, device=device)
            with torch.inference_mode():
                encoder(device_batch)
            values = capture.take_sample(spec.blocks, rows_device)
            positions = torch.from_numpy(output_positions)
            for block_index, block in enumerate(spec.blocks):
                samples[block][positions] = values[block_index]
            filled[output_positions] = True

            if ordinal == 1 or ordinal % 250 == 0 or ordinal == len(selected_proteins):
                logger.info(
                    f"{spec.title}: {ordinal}/{len(selected_proteins)} proteins; "
                    f"{filled.sum()}/{N_POINTS} sampled residues"
                )
    finally:
        capture.remove()
        del encoder

    if not filled.all():
        raise AssertionError(f"only {filled.sum()}/{N_POINTS} sampled rows were filled")
    for block, values in samples.items():
        if values.shape != (N_POINTS, HIDDEN_WIDTH) or not torch.isfinite(values).all():
            raise AssertionError(f"invalid block {block} sample: {values.shape}")
    return samples, checkpoint_hash, len(selected_proteins)


def load_or_extract_samples(
    spec: ModelSpec,
    device: str,
    selected_by_protein: dict[str, tuple[np.ndarray, np.ndarray]],
    selection_digest: str,
    force_extract: bool,
) -> tuple[dict[int, torch.Tensor], str, int, Path]:
    cache_path = RESULTS_ROOT / f"{spec.key}-sampled-768d.pt"
    if cache_path.exists() and not force_extract:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("selection_sha256") == selection_digest
            and tuple(cached.get("blocks", ())) == spec.blocks
            and cached.get("checkpoint_sha256") == spec.checkpoint_sha256
        ):
            logger.info(f"Reusing sampled block states from {cache_path}")
            return cached["states"], spec.checkpoint_sha256, int(cached["proteins"]), cache_path
        logger.warning(f"Cached sample metadata changed; re-extracting {cache_path}")

    states, checkpoint_hash, proteins = extract_sampled_states(
        spec, device, selected_by_protein
    )
    atomic_torch_save(
        {
            "schema_version": 1,
            "artifact": "matched_sampled_768d_post_block_states",
            "model": spec.title,
            "checkpoint": str(spec.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "encoder_source": spec.source,
            "blocks": spec.blocks,
            "selection_sha256": selection_digest,
            "proteins": proteins,
            "n_points": N_POINTS,
            "states": states,
        },
        cache_path,
    )
    logger.info(f"Saved sampled 768D states to {cache_path}")
    return states, checkpoint_hash, proteins, cache_path


def coordinate_path(spec: ModelSpec, block: int) -> Path:
    return RESULTS_ROOT / f"{spec.key}-block{block}-n{N_POINTS}-cosine-tsne.npz"


def load_or_fit_coordinates(
    spec: ModelSpec, block: int, values: torch.Tensor, force_tsne: bool
) -> tuple[np.ndarray, str, Path]:
    X = values.numpy()
    digest = input_digest(X)
    path = coordinate_path(spec, block)
    if path.exists() and not force_tsne:
        cached = np.load(path, allow_pickle=False)
        if (
            str(cached["input_sha256"].item()) == digest
            and int(cached["max_iter"].item()) == MAX_ITER
            and float(cached["perplexity"].item()) == PERPLEXITY
            and int(cached["seed"].item()) == SEED
        ):
            logger.info(f"Block {block}: reusing {path}")
            return cached["embedding"], digest, path
        logger.warning(f"Block {block}: cached coordinates changed; refitting")

    logger.info(f"Block {block}: fitting cosine t-SNE on {X.shape}")
    embedding = project_tsne(X, perplexity=PERPLEXITY, seed=SEED, max_iter=MAX_ITER)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        embedding=embedding,
        input_sha256=np.array(digest),
        max_iter=np.array(MAX_ITER),
        perplexity=np.array(PERPLEXITY),
        seed=np.array(SEED),
        metric=np.array("cosine"),
    )
    os.replace(temporary, path)
    return embedding, digest, path


def render(spec: ModelSpec, embeddings: dict[int, np.ndarray], aa: np.ndarray) -> Path:
    if len(spec.blocks) == 12:
        ncols, figsize = 4, (13.2, 10.8)
    else:
        ncols, figsize = len(spec.blocks), (4.15 * len(spec.blocks), 4.25)
    nrows = math.ceil(len(spec.blocks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()
    for ax, block in zip(flat_axes, spec.blocks):
        scatter_panel(ax, embeddings[block], aa, view="aa20", annotate_aa=False)
        ax.set_title(f"Block {block}", fontsize=11)
    for ax in flat_axes[len(spec.blocks) :]:
        ax.axis("off")

    labels, colors = view_labels_colors("aa20")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=label)
        for label, color in zip(labels, colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7.5, frameon=False)
    fig.suptitle(
        f"{spec.title}: 768D post-block residue representations",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.057,
        f"Same {N_POINTS:,} residues in every panel · cosine t-SNE · perplexity {PERPLEXITY:g}",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.105, 1, 0.965])
    output_path = FIGURE_ROOT / spec.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}.png")
    fig.savefig(temporary, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    os.replace(temporary, output_path)
    logger.info(f"Wrote {output_path}")
    return output_path


def run(model: str, device: str, force_extract: bool, force_tsne: bool) -> None:
    spec = MODEL_SPECS[model]
    selection, sampled, selected_by_protein = load_selection()
    selection_digest = sample_sha256(selection, sampled)
    aa = selection.aa[sampled].astype(np.int64, copy=False)
    logger.info(
        f"Selection pool={selection.n}; sampled={len(sampled)}; "
        f"proteins={len(selected_by_protein)}; sha256={selection_digest}"
    )
    states, checkpoint_hash, proteins, sample_cache = load_or_extract_samples(
        spec, device, selected_by_protein, selection_digest, force_extract
    )

    embeddings: dict[int, np.ndarray] = {}
    coordinate_records = []
    for block in spec.blocks:
        embedding, digest, path = load_or_fit_coordinates(
            spec, block, states[block], force_tsne
        )
        if embedding.shape != (N_POINTS, 2) or not np.isfinite(embedding).all():
            raise AssertionError(f"invalid t-SNE coordinates for block {block}")
        embeddings[block] = embedding
        coordinate_records.append(
            {"block": block, "path": str(path), "input_sha256": digest}
        )
    output_path = render(spec, embeddings, aa)

    manifest = {
        "schema_version": 1,
        "artifact": "layerwise_768d_residue_tsne",
        "model": spec.title,
        "checkpoint": str(spec.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "encoder_source": spec.source,
        "representation": "post-transformer-block, pre-final-LayerNorm, 768D",
        "context": "full unmasked protein; no evaluation rotation",
        "blocks": list(spec.blocks),
        "selection": {
            "labels_path": str(LABELS_PATH),
            "validity": "AA-valid AND secondary-structure-valid",
            "pool_residues": selection.n,
            "sampled_residues": len(sampled),
            "sampled_proteins": proteins,
            "seed": SEED,
            "sha256": selection_digest,
        },
        "tsne": {
            "metric": "cosine",
            "perplexity": PERPLEXITY,
            "max_iter": MAX_ITER,
            "init": "random",
            "method": "barnes_hut",
            "seed": SEED,
        },
        "sample_cache": str(sample_cache),
        "coordinates": coordinate_records,
        "figure": str(output_path),
    }
    atomic_json_save(manifest, RESULTS_ROOT / f"{spec.key}-manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-tsne", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.device, args.force_extract, args.force_tsne)
