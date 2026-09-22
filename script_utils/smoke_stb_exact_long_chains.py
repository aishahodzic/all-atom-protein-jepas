#!/usr/bin/env python3
"""Encode real 512- and 600-residue chains with both JEPA encoder branches."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

if os.environ.get("STB_LAPROTEINA_ONLY_IMPORT") == "1":
    vqvae_stub = types.ModuleType("vqvae_model")
    vqvae_stub.VQVAEModel = object
    sys.modules["vqvae_model"] = vqvae_stub

from tokenizer import WrappedLaProteinaTokenizer


CASES = (
    {"pdb_id": "1mmo", "chain_id": "D", "expected_length": 512},
    {"pdb_id": "1lrw", "chain_id": "A", "expected_length": 600},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def encode_branch(args: argparse.Namespace, source: str) -> list[dict]:
    device = torch.device(args.device)
    tokenizer = WrappedLaProteinaTokenizer(
        device=args.device,
        pretrained_ckpt_path=str(args.checkpoint),
        checkpoint_type="jepa",
        encoder_cfg_path=str(args.encoder_config),
        encoder_source=source,
        evaluation_max_length=600,
        evaluation_cache_tag="exact-long-chain-smoke-v1",
    )

    results = []
    for case in CASES:
        pdb_path = args.pdb_root / f"{case['pdb_id']}.cif"
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            latent, residue_index, residue_type = tokenizer.encode_structure(
                str(pdb_path), case["chain_id"]
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        actual_length = int(latent.shape[0])
        if actual_length != case["expected_length"]:
            raise AssertionError(
                f"{case['pdb_id']}_{case['chain_id']}: expected "
                f"{case['expected_length']} residues, encoded {actual_length}"
            )
        if len(residue_index) != actual_length or len(residue_type) != actual_length:
            raise AssertionError("Encoder outputs disagree on residue length")
        if not torch.isfinite(latent).all():
            raise AssertionError("Latent contains NaN or Inf")

        row = {
            **case,
            "encoder_source": source,
            "encoder_state_prefix": tokenizer.encoder_state_prefix,
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype),
            "all_finite": True,
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
                if device.type == "cuda"
                else None
            ),
        }
        print(json.dumps(row), flush=True)
        results.append(row)

    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for source in ("online", "target"):
        results.extend(encode_branch(args, source))

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "encoder_config": str(args.encoder_config.resolve()),
        "pdb_root": str(args.pdb_root.resolve()),
        "execution_device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "results": results,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS: wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
