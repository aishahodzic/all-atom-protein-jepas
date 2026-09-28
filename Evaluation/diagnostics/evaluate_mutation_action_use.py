"""Test whether a frozen mutation-trained predictor uses the supplied action.

For each protein, the structure, mutation, target representation, and random
50% context/target split are held fixed.  Only the predictor action changes:

* correct replacement identity at the correct residue position
* a shuffled replacement identity at the same position
* no action

The diagnostic evaluates mutations placed in the masked target and visible
context separately.  It is an input intervention on an existing checkpoint,
not a comparison with a separately trained action-free model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from proteinfoundation.jepa.jepa_model import (
    JEPAModel,
    apply_mutation,
    mask_batch_for_encoder,
    uniform_sampling,
)
from proteinfoundation.jepa.masking import apply_mask_to_latents, random_residue_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument(
        "--family",
        choices=("jepa", "jepa_sigreg"),
        required=True,
        help="Training objective used by the checkpoint.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-proteins", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[JEPAModel, Any]:
    cfg = OmegaConf.create(json.loads(args.encoder_config.read_text()))
    cfg.baseline_run_layernorm = False
    cfg.baseline_run_ema = False
    cfg.baseline_run_predictor = False
    cfg.lejepa_relative_positions = False
    cfg.jepa_independent_target_rotation = False
    cfg.jepa_sigreg = False
    cfg.jepa_sigreg_weight = 0.0
    cfg.predictor = {
        "max_len": 256,
        "relative_position_max_distance": 128,
        "use_self_attn": False,
    }
    cfg.opt = {"lr": 1e-4, "max_steps": 100_000}
    cfg.dataset.datamodule.batch_size = args.batch_size
    cfg.dataset.datamodule.num_workers = args.num_workers
    cfg.dataset.datamodule.pin_memory = device.type == "cuda"

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if args.family == "jepa_sigreg":
        model = JEPAModel(
            cfg,
            lamb=float(hparams.get("lamb", 0.02)),
            use_action_conditioning=True,
        )
    else:
        model = JEPAModel(
            cfg,
            ema_decay=float(hparams.get("ema_decay", 0.996)),
            use_action_conditioning=True,
            action_reg_warmup_steps=int(hparams.get("action_reg_warmup_steps", 25_000)),
            action_reg_anneal_steps=int(hparams.get("action_reg_anneal_steps", 50_000)),
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg


def make_loader(cfg: Any):
    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.setup()
    return datamodule.val_dataloader(), len(datamodule.val_ds)


def sample_positions(candidate_mask: torch.Tensor) -> torch.Tensor:
    scores = torch.rand(candidate_mask.shape, device=candidate_mask.device)
    scores.masked_fill_(~candidate_mask, float("-inf"))
    return scores.argmax(dim=-1)


def shuffled_identities(mutated_aa: torch.Tensor) -> torch.Tensor:
    """Roll identities across proteins, repairing accidental identity matches."""
    wrong = mutated_aa.roll(1)
    same = wrong == mutated_aa
    if same.any():
        wrong = wrong.clone()
        wrong[same] = (mutated_aa[same] + 1) % 20
    return wrong


def per_residue_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    family: str,
) -> torch.Tensor:
    if family == "jepa":
        prediction = F.normalize(prediction.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        return F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    return (prediction.float() - target.float()).square().mean(dim=-1)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 500
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarise(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    by_placement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_placement[row["placement"]].append(row)

    output: dict[str, Any] = {}
    for placement, group in by_placement.items():
        placement_summary: dict[str, Any] = {"n_proteins": len(group)}
        for scope in ("all_targets", "mutated_site", "other_targets"):
            valid = [row for row in group if row[f"{scope}_correct"] is not None]
            if not valid:
                continue
            correct = np.asarray([row[f"{scope}_correct"] for row in valid], dtype=float)
            wrong = np.asarray([row[f"{scope}_wrong"] for row in valid], dtype=float)
            absent = np.asarray([row[f"{scope}_absent"] for row in valid], dtype=float)
            delta_wrong = correct - wrong
            delta_absent = correct - absent
            placement_summary[scope] = {
                "n": len(valid),
                "mean_error": {
                    "correct": float(correct.mean()),
                    "wrong": float(wrong.mean()),
                    "absent": float(absent.mean()),
                },
                "correct_minus_wrong": {
                    "mean": float(delta_wrong.mean()),
                    "ci95": bootstrap_ci(delta_wrong, bootstrap_samples, seed + 11),
                    "fraction_correct_lower": float((delta_wrong < 0).mean()),
                },
                "correct_minus_absent": {
                    "mean": float(delta_absent.mean()),
                    "ci95": bootstrap_ci(delta_absent, bootstrap_samples, seed + 23),
                    "fraction_correct_lower": float((delta_absent < 0).mean()),
                },
            }
        for key in (
            "prediction_shift_correct_wrong",
            "prediction_shift_correct_absent",
            "target_shift_mutant_wildtype",
        ):
            values = np.asarray([row[key] for row in group], dtype=float)
            placement_summary[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci95_mean": bootstrap_ci(values, bootstrap_samples, seed + 37),
            }
        output[placement] = placement_summary
    return output


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, cfg = load_model(args, device)
    loader, validation_size = make_loader(cfg)
    rows: list[dict[str, Any]] = []
    proteins_seen = 0

    for batch_index, batch in enumerate(loader):
        if proteins_seen >= args.num_proteins:
            break
        batch = to_device(batch, device)
        batch_size = int(batch["residue_type"].shape[0])
        keep = min(batch_size, args.num_proteins - proteins_seen)
        if keep < batch_size:
            batch = {
                key: (
                    {subkey: value[:keep] for subkey, value in item.items()}
                    if isinstance(item, dict)
                    else item[:keep]
                    if torch.is_tensor(item) and item.shape[:1] == (batch_size,)
                    else item[:keep]
                    if isinstance(item, list) and len(item) == batch_size
                    else item
                )
                for key, item in batch.items()
            }
            batch_size = keep

        torch.manual_seed(args.seed + batch_index)
        valid_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
        batch["mask"] = valid_mask
        context_mask, target_mask = random_residue_mask(batch, masked_pct=0.5)
        masked_batch = mask_batch_for_encoder(batch, context_mask)

        context_mean = model._encode_mean(model.online_encoder, masked_batch)
        packed = apply_mask_to_latents(context_mean, context_mask, target_mask)
        predictor_args = (
            packed["context_latents"],
            packed["context_indices"],
            packed["target_indices"],
        )
        predictor_kwargs = {
            "key_padding_mask": packed["context_padding"],
            "target_padding_mask": packed["target_padding"],
        }
        target_real = ~packed["target_padding"]

        for placement, candidate_mask in (
            ("masked_target", target_mask),
            ("visible_context", context_mask),
        ):
            positions = sample_positions(candidate_mask)
            batch_indices = torch.arange(batch_size, device=device)
            wildtype_aa = batch["residue_type"][batch_indices, positions]
            mutated_aa = uniform_sampling(wildtype_aa)
            wrong_aa = shuffled_identities(mutated_aa)
            mutant_batch = apply_mutation(batch, positions, mutated_aa)

            if args.family == "jepa":
                target_mutant_full = model._encode_mean(model.ema_encoder, mutant_batch)
                target_wildtype_full = model._encode_mean(model.ema_encoder, batch)
            else:
                target_mutant_full = model._encode_mean(model.online_encoder, mutant_batch)
                target_wildtype_full = model._encode_mean(model.online_encoder, batch)
            target_mutant = apply_mask_to_latents(
                target_mutant_full, context_mask, target_mask
            )["target_latents"]
            target_wildtype = apply_mask_to_latents(
                target_wildtype_full, context_mask, target_mask
            )["target_latents"]

            action_position = model.predictor.pos_emb(positions)
            action_correct = (
                model.mutation_embedding(mutated_aa) + action_position
            ).unsqueeze(1)
            action_wrong = (
                model.mutation_embedding(wrong_aa) + action_position
            ).unsqueeze(1)

            prediction_correct = model.predictor(
                *predictor_args,
                **predictor_kwargs,
                action_z=action_correct,
                action_position=positions,
            )
            prediction_wrong = model.predictor(
                *predictor_args,
                **predictor_kwargs,
                action_z=action_wrong,
                action_position=positions,
            )
            prediction_absent = model.predictor(*predictor_args, **predictor_kwargs)

            errors = {
                "correct": per_residue_error(
                    prediction_correct, target_mutant, args.family
                ),
                "wrong": per_residue_error(prediction_wrong, target_mutant, args.family),
                "absent": per_residue_error(
                    prediction_absent, target_mutant, args.family
                ),
            }
            prediction_shift_wrong = (
                prediction_correct.float() - prediction_wrong.float()
            ).norm(dim=-1)
            prediction_shift_absent = (
                prediction_correct.float() - prediction_absent.float()
            ).norm(dim=-1)
            target_shift = (
                target_mutant.float() - target_wildtype.float()
            ).norm(dim=-1)

            for protein_index in range(batch_size):
                real = target_real[protein_index]
                indices = packed["target_indices"][protein_index]
                site = real & (indices == positions[protein_index])
                other = real & ~site
                row: dict[str, Any] = {
                    "protein_index": proteins_seen + protein_index,
                    "placement": placement,
                    "mutation_position": int(positions[protein_index].item()),
                    "wildtype_aa": int(wildtype_aa[protein_index].item()),
                    "mutated_aa": int(mutated_aa[protein_index].item()),
                    "wrong_action_aa": int(wrong_aa[protein_index].item()),
                }
                for scope, mask in (
                    ("all_targets", real),
                    ("mutated_site", site),
                    ("other_targets", other),
                ):
                    for condition in ("correct", "wrong", "absent"):
                        row[f"{scope}_{condition}"] = (
                            float(errors[condition][protein_index][mask].mean().item())
                            if mask.any()
                            else None
                        )
                row["prediction_shift_correct_wrong"] = float(
                    prediction_shift_wrong[protein_index][real].mean().item()
                )
                row["prediction_shift_correct_absent"] = float(
                    prediction_shift_absent[protein_index][real].mean().item()
                )
                row["target_shift_mutant_wildtype"] = float(
                    target_shift[protein_index][real].mean().item()
                )
                rows.append(row)

        proteins_seen += batch_size
        print(
            f"processed {proteins_seen}/{args.num_proteins} proteins",
            flush=True,
        )

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_global_step": int(
            torch.load(args.checkpoint, map_location="cpu", weights_only=False).get(
                "global_step", -1
            )
        ),
        "family": args.family,
        "cohort": (
            "cluster-random representatives from the minimum-length-50 "
            "sequence-clustered validation split"
        ),
        "validation_size": validation_size,
        "num_proteins": proteins_seen,
        "batch_size": args.batch_size,
        "masking": "random 50%",
        "seed": args.seed,
        "comparison": (
            "same structure, mutation, target representation, context/target split, "
            "and action position; only action identity/presence changes"
        ),
        "wrong_action": (
            "replacement identities rolled across the batch, with accidental identity "
            "matches replaced by the next amino-acid index"
        ),
    }
    return rows, metadata


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    rows, metadata = evaluate(args)
    summary = summarise(rows, args.bootstrap_samples, args.seed)
    payload = {"metadata": metadata, "summary": summary}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
