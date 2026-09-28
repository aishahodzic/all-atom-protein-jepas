"""Large matched rotation-invariance audit of uncropped LeJEPA 10k EMA encoders.

The same validation/test proteins and Haar-uniform SO(3) rotations are used for
both frozen models. All views are full-context; the only intervention is a
rigid rotation of both coordinate tensors. Results are saved per protein and
rotation, with protein-level bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Evaluation/linear_probing"))
sys.path.insert(0, str(REPO / "Evaluation/diagnostics"))

from data_pool import build_protein_pool, eval_transform_compose  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list  # noqa: E402
from rotation_latent_gauge import load_encoder  # noqa: E402


CHECKPOINTS = {
    "uncropped_lejepa_ema_10k": REPO / "store/canonical_lejepa_v1_100k/evaluation/canonical-lejepa-v1-step10000-ema-staged.ckpt",
    "uncropped_lejepa_so3_ema_10k": REPO / "store/canonical_lejepa_rotations_100k/evaluation/canonical-lejepa-v1-rotations-step10000-ema-staged.ckpt",
}
RESULT_DIR = REPO / "Evaluation/diagnostics/results"


def haar_rotation(generator: torch.Generator) -> torch.Tensor:
    quaternion = torch.randn(4, generator=generator)
    w, x, y, z = quaternion / quaternion.norm()
    return torch.tensor([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=torch.float32)


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    cosine = F.cosine_similarity(a, b, dim=-1).mean().item()
    rel_l2 = (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()
    return cosine, rel_l2


def mean_and_ci(values: list[float], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {"mean": mean(values), "ci95_low": boot[49], "ci95_high": boot[1949]}


def validate_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["canonical_lejepa_evaluation"]
    expected = {"encoder_source": "ema", "source_global_step": 10000, "latent_dim": 8, "projector": None}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}, found {metadata.get(key)!r}")
    return metadata


def make_batch(pool, indices: list[int], device: torch.device):
    transform = eval_transform_compose()
    graphs = [transform(pool.dataset[i]) for i in indices]
    batch = dense_padded_from_data_list(graphs)
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
    batch["chain_breaks_per_residue"] = torch.stack([
        F.pad(graph.chain_breaks_per_residue, (0, batch["coords"].shape[1] - len(graph.chain_breaks_per_residue)))
        for graph in graphs
    ])
    return batch.to(device)


def run_model(name, checkpoint, pool, selected, rotations, args, device):
    encoder = load_encoder(checkpoint).to(device).eval()
    rows = []
    for start in range(0, len(selected), args.batch_size):
        batch_indices = selected[start:start + args.batch_size]
        batch = make_batch(pool, batch_indices, device)
        original_nm = batch["coords_nm"].clone()
        original_angstrom = batch["coords"].clone()
        with torch.inference_mode():
            original_latents = encoder(batch)["mean"].float().cpu()
            for rotation_index, rotation in enumerate(rotations):
                rotation = rotation.to(device)
                batch["coords_nm"] = original_nm @ rotation
                batch["coords"] = original_angstrom @ rotation
                rotated_latents = encoder(batch)["mean"].float().cpu()
                for i, protein_index in enumerate(batch_indices):
                    mask = batch["mask"][i].cpu()
                    a = original_latents[i][mask]
                    b = rotated_latents[i][mask]
                    if len(a) < 2:
                        raise ValueError(f"Too few residues: {pool.protein_ids[protein_index]}")
                    ac = a - a.mean(0, keepdim=True)
                    bc = b - b.mean(0, keepdim=True)
                    permutation_generator = torch.Generator().manual_seed(args.seed + 1000003 * protein_index + rotation_index)
                    permutation = torch.randperm(len(a), generator=permutation_generator)
                    raw_cosine, raw_rel_l2 = metrics(a, b)
                    centered_cosine, centered_rel_l2 = metrics(ac, bc)
                    shuffle_cosine, shuffle_rel_l2 = metrics(ac, bc[permutation])
                    rows.append({
                        "model": name,
                        "protein_id": pool.protein_ids[protein_index],
                        "origin": pool.origin[pool.protein_ids[protein_index]],
                        "chain_length": len(a),
                        "rotation_index": rotation_index,
                        "rotation_angle_degrees": math.degrees(math.acos(max(-1.0, min(1.0, (rotation.trace().item() - 1) / 2)))),
                        "raw_cosine": raw_cosine,
                        "raw_relative_l2": raw_rel_l2,
                        "centered_cosine": centered_cosine,
                        "centered_relative_l2": centered_rel_l2,
                        "shuffled_centered_cosine": shuffle_cosine,
                        "shuffled_centered_relative_l2": shuffle_rel_l2,
                        "residue_signal_to_total_norm": (torch.linalg.vector_norm(ac) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item(),
                    })
        del batch, original_latents
        if (start // args.batch_size + 1) % 10 == 0 or start + args.batch_size >= len(selected):
            print(f"{name}: {min(start + args.batch_size, len(selected))}/{len(selected)} proteins", flush=True)
    del encoder
    return rows


def summarize(rows, seed):
    per_protein = defaultdict(list)
    for row in rows:
        per_protein[row["protein_id"]].append(row)
    fields = ["raw_cosine", "raw_relative_l2", "centered_cosine", "centered_relative_l2",
              "shuffled_centered_cosine", "shuffled_centered_relative_l2", "residue_signal_to_total_norm"]
    return {
        "n_proteins": len(per_protein),
        "n_residues": sum(group[0]["chain_length"] for group in per_protein.values()),
        "n_protein_rotation_pairs": len(rows),
        "equal_protein_weighted": {field: mean_and_ci([mean(row[field] for row in group) for group in per_protein.values()], seed + i)
                                   for i, field in enumerate(fields)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=int, default=400)
    parser.add_argument("--n-rotations", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("This large diagnostic is intended for a CUDA cluster node")
    device = torch.device("cuda")
    checkpoint_metadata = {name: validate_checkpoint(path) for name, path in CHECKPOINTS.items()}
    pool = build_protein_pool()
    if args.n_proteins > len(pool.dataset):
        raise ValueError("Requested more proteins than available")
    selected = random.Random(args.seed).sample(range(len(pool.dataset)), args.n_proteins)
    rotation_generator = torch.Generator().manual_seed(args.seed)
    rotations = [haar_rotation(rotation_generator) for _ in range(args.n_rotations)]
    for rotation in rotations:
        if not torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5) or not torch.isclose(torch.det(rotation), torch.tensor(1.0), atol=1e-5):
            raise ValueError("Invalid SO(3) rotation")
    stem = f"lejepa_rotation_invariance_ema_10k_{args.n_proteins}p_{args.n_rotations}r_seed{args.seed}"
    csv_path = RESULT_DIR / f"{stem}.csv"
    json_path = RESULT_DIR / f"{stem}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite {stem} results")
    report = {
        "description": "Matched full-context SO(3) rotation-invariance evaluation of uncropped LeJEPA 10k EMA encoders",
        "seed": args.seed,
        "n_proteins_requested": args.n_proteins,
        "n_rotations": args.n_rotations,
        "selected_proteins": [pool.protein_ids[i] for i in selected],
        "origin_counts": {origin: sum(pool.origin[pool.protein_ids[i]] == origin for i in selected) for origin in ("val", "test")},
        "rotation_matrices": [rotation.tolist() for rotation in rotations],
        "checkpoint_provenance": checkpoint_metadata,
        "summary": {},
    }
    all_rows = []
    for name, checkpoint in CHECKPOINTS.items():
        rows = run_model(name, checkpoint, pool, selected, rotations, args, device)
        report["summary"][name] = summarize(rows, args.seed)
        print(json.dumps({name: report["summary"][name]}, indent=2), flush=True)
        all_rows.extend(rows)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
