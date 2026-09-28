"""Measure view agreement at the trunk, final LayerNorm, and native latent.

The protocol uses the same held-out proteins and deterministic masks for every
checkpoint.  It reports protein-level agreement across views and exact-residue
agreement/retrieval wherever two views contain the same original residue.
No probe is fitted and no model parameter is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO = pathlib.Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "Evaluation/linear_probing"):
    sys.path.insert(0, str(path))

from proteinfoundation.jepa.canonical_lejepa_model import (  # noqa: E402
    CANONICAL_NUM_VIEWS,
    CanonicalLeJEPAModel,
)
from proteinfoundation.utils.dense_padding_data_loader import (  # noqa: E402
    dense_padded_from_data_list,
)


STAGES = ("block12_768d", "post_layernorm_768d", "final_mean_8d")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if hasattr(value, "to"):
        return value.to(device)
    return value


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear CKA after centering samples, without constructing Gram matrices."""
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = a.T @ b
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.sqrt(np.square(a.T @ a).sum() * np.square(b.T @ b).sum())
    )
    return numerator / max(denominator, 1e-30)


def _normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def summarize_pooled_views(values: np.ndarray) -> dict[str, Any]:
    """Summarize [proteins, five views, dimension] protein representations."""
    if values.ndim != 3 or values.shape[1] != CANONICAL_NUM_VIEWS:
        raise ValueError("pooled values must have shape [proteins, five views, dim]")
    n_proteins = values.shape[0]
    if n_proteins < 2:
        raise ValueError("at least two proteins are required")

    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    standardized = ((values - mean) / np.maximum(std, 1e-8)).astype(np.float64)
    unit = _normalise(values.astype(np.float64))
    unit_standardized = _normalise(standardized)

    raw_paired, raw_shuffled = [], []
    z_paired, z_shuffled = [], []
    ckas, top1s, top5s, reciprocal_ranks = [], [], [], []
    for local in range(1, CANONICAL_NUM_VIEWS):
        shuffled = np.roll(np.arange(n_proteins), 1)
        raw_paired.extend(np.sum(unit[:, 0] * unit[:, local], axis=1))
        raw_shuffled.extend(np.sum(unit[:, 0] * unit[shuffled, local], axis=1))
        z_paired.extend(
            np.sum(unit_standardized[:, 0] * unit_standardized[:, local], axis=1)
        )
        z_shuffled.extend(
            np.sum(
                unit_standardized[:, 0] * unit_standardized[shuffled, local],
                axis=1,
            )
        )
        ckas.append(linear_cka(values[:, 0], values[:, local]))

        similarities = unit_standardized[:, 0] @ unit_standardized[:, local].T
        order = np.argsort(-similarities, axis=1)
        target = np.arange(n_proteins)
        ranks = np.argmax(order == target[:, None], axis=1) + 1
        top1s.append(float(np.mean(ranks == 1)))
        top5s.append(float(np.mean(ranks <= min(5, n_proteins))))
        reciprocal_ranks.extend(1.0 / ranks)

    raw_paired = np.asarray(raw_paired)
    raw_shuffled = np.asarray(raw_shuffled)
    z_paired = np.asarray(z_paired)
    z_shuffled = np.asarray(z_shuffled)
    paired_z_mse = float(
        np.square(
            standardized[:, :1] - standardized[:, 1:]
        ).mean()
    )
    shuffled_z_mse = float(
        np.square(
            standardized[:, :1]
            - np.roll(standardized[:, 1:], 1, axis=0)
        ).mean()
    )
    return {
        "n_proteins": int(n_proteins),
        "raw_paired_cosine": float(raw_paired.mean()),
        "raw_shuffled_cosine": float(raw_shuffled.mean()),
        "raw_cosine_margin": float(raw_paired.mean() - raw_shuffled.mean()),
        "standardized_paired_cosine": float(z_paired.mean()),
        "standardized_shuffled_cosine": float(z_shuffled.mean()),
        "standardized_cosine_margin": float(z_paired.mean() - z_shuffled.mean()),
        "paired_standardized_mse": paired_z_mse,
        "shuffled_standardized_mse": shuffled_z_mse,
        "paired_to_shuffled_mse_ratio": paired_z_mse / max(shuffled_z_mse, 1e-30),
        "linear_cka_mean": float(np.mean(ckas)),
        "cross_view_protein_retrieval_top1": float(np.mean(top1s)),
        "cross_view_protein_retrieval_top5": float(np.mean(top5s)),
        "cross_view_protein_retrieval_mrr": float(np.mean(reciprocal_ranks)),
        "per_local_linear_cka": [float(value) for value in ckas],
        "per_local_retrieval_top1": top1s,
    }


class ResidueAgreement:
    """Weighted exact-residue agreement and within-protein retrieval totals."""

    def __init__(self) -> None:
        self.count = 0
        self.cosine_sum = 0.0
        self.shuffled_cosine_sum = 0.0
        self.top1 = 0
        self.top10 = 0
        self.reciprocal_rank_sum = 0.0
        self.protein_view_pairs = 0

    def add(self, global_values: torch.Tensor, local_values: torch.Tensor) -> None:
        if global_values.ndim != 2 or global_values.shape != local_values.shape:
            raise ValueError("paired residue arrays must have the same [N, D] shape")
        n = int(global_values.shape[0])
        if n == 0:
            return
        a = F.normalize(global_values.float(), dim=-1)
        b = F.normalize(local_values.float(), dim=-1)
        exact = (a * b).sum(dim=-1)
        self.cosine_sum += float(exact.sum())
        self.count += n
        self.protein_view_pairs += 1
        if n > 1:
            self.shuffled_cosine_sum += float((a * b.roll(1, dims=0)).sum())
        else:
            self.shuffled_cosine_sum += float(exact.sum())

        similarities = a @ b.T
        order = torch.argsort(similarities, dim=1, descending=True)
        target = torch.arange(n, device=order.device)[:, None]
        ranks = torch.argmax((order == target).to(torch.int64), dim=1) + 1
        self.top1 += int((ranks == 1).sum())
        self.top10 += int((ranks <= min(10, n)).sum())
        self.reciprocal_rank_sum += float((1.0 / ranks.float()).sum())

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "residue_pairs": 0,
                "protein_view_pairs": 0,
                "paired_cosine": float("nan"),
                "shuffled_cosine": float("nan"),
                "cosine_margin": float("nan"),
                "within_protein_retrieval_top1": float("nan"),
                "within_protein_retrieval_top10": float("nan"),
                "within_protein_retrieval_mrr": float("nan"),
            }
        paired = self.cosine_sum / self.count
        shuffled = self.shuffled_cosine_sum / self.count
        return {
            "residue_pairs": self.count,
            "protein_view_pairs": self.protein_view_pairs,
            "paired_cosine": paired,
            "shuffled_cosine": shuffled,
            "cosine_margin": paired - shuffled,
            "within_protein_retrieval_top1": self.top1 / self.count,
            "within_protein_retrieval_top10": self.top10 / self.count,
            "within_protein_retrieval_mrr": self.reciprocal_rank_sum / self.count,
        }


def _dense_view(
    values: torch.Tensor,
    original_mask: torch.Tensor,
    *,
    packed: bool,
) -> torch.Tensor:
    """Restore one single-protein hook output to original residue rows."""
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(f"expected single-protein [1, rows, dim], got {values.shape}")
    length = int(original_mask.numel())
    if packed:
        count = int(original_mask.sum())
        if values.shape[1] != count:
            raise RuntimeError(
                f"packed output has {values.shape[1]} rows, expected {count}"
            )
        dense = values.new_zeros((length, values.shape[-1]))
        dense[original_mask] = values[0]
        return dense
    if values.shape[1] != length:
        raise RuntimeError(
            f"dense output has {values.shape[1]} rows, expected {length}"
        )
    return values[0]


def _capture_one_protein(
    model: CanonicalLeJEPAModel,
    batch: dict[str, Any],
) -> tuple[
    dict[str, list[torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    captured: dict[str, list[torch.Tensor]] = defaultdict(list)

    def hook(name: str):
        def capture(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{name} hook expected Tensor, got {type(output)}")
            captured[name].append(output.detach().float())

        return capture

    handles = [
        model.encoder.transformer_layers[-1].register_forward_hook(hook(STAGES[0])),
        model.encoder.latent_decoder_mean_n_log_scale[0].register_forward_hook(
            hook(STAGES[1])
        ),
        model.encoder.latent_decoder_mean_n_log_scale[1].register_forward_hook(
            hook("flat_final")
        ),
    ]
    try:
        if model.alignment_level == "protein":
            native_pooled, view_masks = model.encode_views(batch)
            native_residues = None
            query_masks = view_masks
        else:
            native_residues, native_pooled, view_masks, real_masks = (
                model.encode_residue_views(batch)
            )
            query_masks = (
                real_masks[:, None].expand_as(view_masks)
                if real_masks.ndim == 2
                else real_masks
            )
    finally:
        for handle in handles:
            handle.remove()

    for name in (*STAGES[:2], "flat_final"):
        if len(captured[name]) != CANONICAL_NUM_VIEWS:
            raise RuntimeError(
                f"{name} hook fired {len(captured[name])} times, expected five"
            )
    latent_dim = captured["flat_final"][0].shape[-1] // 2
    if latent_dim != 8:
        raise RuntimeError(f"expected native 8D mean, got {latent_dim}D")
    captured[STAGES[2]] = [value[..., :latent_dim] for value in captured["flat_final"]]
    del captured["flat_final"]
    return (
        captured,
        view_masks[0].bool(),
        query_masks[0].bool(),
        native_pooled[0].detach().float(),
        None if native_residues is None else native_residues[0].detach().float(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--proteins", type=int, default=256)
    parser.add_argument("--minimum-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    # data_pool resolves DATA_PATH at import time. Keep this import in main so
    # metric unit tests and --help do not require the dataset environment.
    from data_pool import build_protein_pool, eval_transform_compose

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.proteins < 2:
        raise ValueError("--proteins must be at least two")
    device = torch.device("cuda")
    cfg = OmegaConf.create(json.loads(args.config.read_text()))
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if int(checkpoint["global_step"]) != args.expected_step:
        raise RuntimeError(
            f"expected step {args.expected_step}, got {checkpoint['global_step']}"
        )
    model = CanonicalLeJEPAModel(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    pool = build_protein_pool(seed=42)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool.dataset))
    transform = eval_transform_compose()
    pooled: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGES}
    residue_query = {stage: ResidueAgreement() for stage in STAGES}
    residue_visible = {stage: ResidueAgreement() for stage in STAGES}
    protein_ids: list[str] = []
    lengths: list[int] = []

    for pool_index in order:
        if len(protein_ids) >= args.proteins:
            break
        graph = transform(pool.dataset[int(pool_index)])
        length = int(graph.residue_type.shape[0])
        if length < args.minimum_length:
            continue
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
        batch = to_device(batch, device)

        per_protein_seed = args.seed * 100_000 + int(pool_index)
        torch.manual_seed(per_protein_seed)
        torch.cuda.manual_seed_all(per_protein_seed)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            (
                captured,
                view_masks,
                query_masks,
                native_pooled,
                native_residues,
            ) = _capture_one_protein(model, batch)

        packed = bool(model.packed_views)
        dense: dict[str, list[torch.Tensor]] = {}
        for stage in STAGES:
            dense[stage] = [
                _dense_view(
                    captured[stage][view],
                    query_masks[view],
                    packed=packed,
                ).cpu()
                for view in range(CANONICAL_NUM_VIEWS)
            ]
            protein_views = []
            for view in range(CANONICAL_NUM_VIEWS):
                protein_views.append(
                    dense[stage][view][query_masks[view].cpu()].mean(dim=0)
                )
            stacked_protein_views = torch.stack(protein_views)
            if stage == STAGES[2]:
                if not torch.allclose(
                    stacked_protein_views,
                    native_pooled.cpu(),
                    atol=2e-3,
                    rtol=2e-3,
                ):
                    difference = float(
                        (stacked_protein_views - native_pooled.cpu()).abs().max()
                    )
                    raise RuntimeError(
                        "captured final-mean pooling does not match the model's "
                        f"native output; max difference={difference}"
                    )
                if native_residues is not None:
                    for view in range(CANONICAL_NUM_VIEWS):
                        mask = query_masks[view].cpu()
                        if not torch.allclose(
                            dense[stage][view][mask],
                            native_residues[view].cpu()[mask],
                            atol=2e-3,
                            rtol=2e-3,
                        ):
                            raise RuntimeError(
                                "captured final residue means do not match the "
                                "model's native dense output"
                            )
            pooled[stage].append(stacked_protein_views.numpy())

            for local in range(1, CANONICAL_NUM_VIEWS):
                query_overlap = query_masks[0] & query_masks[local]
                residue_query[stage].add(
                    dense[stage][0][query_overlap.cpu()],
                    dense[stage][local][query_overlap.cpu()],
                )
                visible_overlap = view_masks[0] & view_masks[local]
                residue_visible[stage].add(
                    dense[stage][0][visible_overlap.cpu()],
                    dense[stage][local][visible_overlap.cpu()],
                )

        protein_ids.append(pool.protein_ids[int(pool_index)])
        lengths.append(length)
        if len(protein_ids) % 16 == 0:
            print(f"encoded {len(protein_ids)}/{args.proteins} proteins", flush=True)

    if len(protein_ids) != args.proteins:
        raise RuntimeError(
            f"found only {len(protein_ids)} eligible proteins, expected {args.proteins}"
        )

    report = {
        "description": (
            "Matched paired-view agreement at post-block-12 768D, final "
            "LayerNorm 768D, and native final mean 8D"
        ),
        "model_label": args.model_label,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "global_step": int(checkpoint["global_step"]),
        "alignment_level": model.alignment_level,
        "cropping": bool(model.cropping),
        "packed_views": bool(model.packed_views),
        "protocol": {
            "protein_pool": "frozen held-out validation+test pool",
            "selection_seed": args.seed,
            "mask_seed_is_deterministic_per_protein": True,
            "minimum_length": args.minimum_length,
            "proteins": len(protein_ids),
            "protein_ids": protein_ids,
            "protein_ids_sha256": hashlib.sha256(
                "\n".join(protein_ids).encode()
            ).hexdigest(),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_mean": float(np.mean(lengths)),
            "same_original_residue_correspondence": True,
            "protein_retrieval_uses_standardized_cosine": True,
        },
        "stages": {},
    }
    for stage in STAGES:
        values = np.stack(pooled[stage])
        report["stages"][stage] = {
            "dimension": int(values.shape[-1]),
            "protein_view_agreement": summarize_pooled_views(values),
            "same_residue_query_overlap": residue_query[stage].summary(),
            "same_residue_visible_visible_overlap": residue_visible[stage].summary(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
