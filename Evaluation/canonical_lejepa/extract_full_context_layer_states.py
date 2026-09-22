"""Capture canonical LeJEPA full-context residue states at selected depths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from openfold.np import residue_constants
from proteinfoundation.utils.dense_padding_data_loader import (
    dense_padded_from_data_list,
)

from Evaluation.canonical_lejepa.checkpoint import (
    extract_canonical_encoder_state_dict,
    load_canonical_encoder,
)
from Evaluation.linear_probing.data_pool import (
    ProteinPool,
    build_protein_pool,
    eval_transform_compose,
)


REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "store/canonical_lejepa_v1_100k/checkpoints/step-00100000.ckpt"
STAGED = REPO / (
    "store/canonical_lejepa_v1_100k/evaluation/"
    "canonical-lejepa-v1-step100000-ema-staged.ckpt"
)
ENCODER_CONFIG = REPO / "StructTokenBench/100k-checkpoints/shared/exp_config_jepa_nn130m.json"
OUTPUT_ROOT = REPO / (
    "store/canonical_lejepa_v1_100k/evaluation/"
    "full_context_layer_states"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4f09dff610adaa75c8ac51b4cf01acb001c236c72dd91ac31568ffa191484527"
)
EXPECTED_STAGED_SHA256 = (
    "49aaae3dccc3835005bb6470b0f483a57cee4e28bead2408994693974389e273"
)
EXPECTED_PROTEINS = 5_224
EXPECTED_RESIDUES = 748_753
SHARD_SIZE = 100
SELECTION_SEED = 42
VALIDITY_EXPRESSION = 'batch["mask_dict"]["coords"][..., 0, 0].bool()'
CAPTURE_LAYERS = {"block_1": 0, "block_6": 5, "block_12": 11}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_tensor_mapping_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], label: str
) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AssertionError(f"{label} key mismatch: missing={missing[:3]}, extra={extra[:3]}")
    for name in sorted(expected):
        if not torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu()):
            raise AssertionError(f"{label} tensor mismatch: {name}")


def load_verified_encoder(device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f"checkpoint SHA-256 mismatch: {checkpoint_hash}")
    staged_hash = sha256_file(STAGED)
    if staged_hash != EXPECTED_STAGED_SHA256:
        raise AssertionError(f"staged SHA-256 mismatch: {staged_hash}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("global_step") != 100_000:
        raise AssertionError(f"unexpected global_step: {checkpoint.get('global_step')}")
    ema_meta = checkpoint.get("state_dict_ema_meta", {})
    if ema_meta.get("last_ema_step") != 100_000:
        raise AssertionError(f"unexpected EMA metadata: {ema_meta}")
    expected_ema = extract_canonical_encoder_state_dict(checkpoint, source="ema")

    encoder = load_canonical_encoder(
        CHECKPOINT, ENCODER_CONFIG, source="ema", device="cpu"
    )
    _assert_tensor_mapping_equal(encoder.state_dict(), expected_ema, "loaded EMA")

    staged = torch.load(STAGED, map_location="cpu", weights_only=False, mmap=True)
    provenance = staged.get("canonical_lejepa_evaluation", {})
    expected_provenance = {
        "encoder_source": "ema",
        "source_checkpoint_sha256": checkpoint_hash,
        "source_global_step": 100_000,
        "latent_dim": 8,
        "projector": None,
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise AssertionError(f"staged provenance mismatch for {key}: {provenance}")
    staged_state = {
        key.removeprefix("online_encoder."): value
        for key, value in staged.get("state_dict", {}).items()
        if key.startswith("online_encoder.")
    }
    _assert_tensor_mapping_equal(staged_state, expected_ema, "staged EMA")

    if len(encoder.transformer_layers) != 12:
        raise AssertionError("canonical encoder must contain 12 transformer blocks")
    if encoder.token_dim != 768:
        raise AssertionError(f"unexpected hidden width: {encoder.token_dim}")
    head = encoder.latent_decoder_mean_n_log_scale[1]
    if tuple(head.weight.shape) != (16, 768):
        raise AssertionError(f"unexpected latent head shape: {tuple(head.weight.shape)}")

    del checkpoint, staged, expected_ema, staged_state
    encoder = encoder.to(device).eval()
    return encoder, {
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "global_step": 100_000,
        "encoder_source": "ema",
        "ema_metadata": ema_meta,
        "staged_evaluation_path": str(STAGED),
        "staged_evaluation_sha256": staged_hash,
        "staged_evaluation_provenance": provenance,
        "encoder_config_path": str(ENCODER_CONFIG),
        "encoder_config_sha256": sha256_file(ENCODER_CONFIG),
    }


class LayerCapture:
    def __init__(self, encoder: torch.nn.Module) -> None:
        self.values: dict[str, list[torch.Tensor]] = {}
        self.handles = []
        for name, index in CAPTURE_LAYERS.items():
            self.handles.append(
                encoder.transformer_layers[index].register_forward_hook(
                    self._make_hook(name)
                )
            )

    def _make_hook(self, name: str):
        def hook(module, args, output):
            del module, args
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{name} hook expected Tensor, got {type(output)}")
            self.values.setdefault(name, []).append(output.detach().float().cpu())

        return hook

    def clear(self) -> None:
        self.values.clear()

    def take(self) -> dict[str, torch.Tensor]:
        missing = [name for name in CAPTURE_LAYERS if len(self.values.get(name, [])) != 1]
        if missing:
            counts = {name: len(self.values.get(name, [])) for name in CAPTURE_LAYERS}
            raise AssertionError(f"each selected block must fire once: {counts}")
        return {name: self.values[name][0] for name in CAPTURE_LAYERS}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def preprocess_graphs(graphs: list[Any]) -> dict[str, Any]:
    transform = eval_transform_compose()
    transformed = [transform(graph) for graph in graphs]
    batch = dense_padded_from_data_list(transformed)
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0].bool()
    return batch


def _aa_labels(values: torch.Tensor) -> list[str]:
    labels = residue_constants.restypes_with_x
    return [labels[int(value)] if 0 <= int(value) < len(labels) else "X" for value in values]


def _graph_row_count(graph: Any) -> int:
    return int(graph.coords.shape[0])


def _graph_encoder_valid_count(graph: Any) -> int:
    return _graph_row_count(graph)


def metadata_for_row(
    pool: ProteinPool, pool_index: int, batch: dict[str, Any], batch_index: int
) -> dict[str, Any]:
    protein_id = pool.protein_ids[pool_index]
    mask = batch["mask"][batch_index].bool().cpu()
    rows = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.int64)
    aa = batch["residue_type"][batch_index].cpu()[mask].to(torch.int64)
    source_chain = str(pool.dataset.chains[pool_index])
    return {
        "protein_id": protein_id,
        "origin": pool.origin[protein_id],
        "probe_split": pool.probe_split[protein_id],
        "original_dense_row_index": rows,
        "residue_pdb_idx": batch["residue_pdb_idx"][batch_index].cpu()[mask].to(torch.int64),
        "chain_index": batch["chains"][batch_index].cpu()[mask].to(torch.int64),
        "chain_identifier": source_chain,
        "aa_integer": aa,
        "aa_label": _aa_labels(aa),
    }


def capture_batch(
    encoder: torch.nn.Module,
    capture: LayerCapture,
    batch: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    capture.clear()
    device_batch = batch.to(device)
    with torch.inference_mode():
        output = encoder(device_batch)
    states = capture.take()
    mean = output["mean"].detach().float().cpu()
    return states, mean


def record_from_capture(
    metadata: dict[str, Any],
    states: dict[str, torch.Tensor],
    mean: torch.Tensor,
    batch: dict[str, Any],
    batch_index: int,
) -> dict[str, Any]:
    mask = batch["mask"][batch_index].bool().cpu()
    record = dict(metadata)
    for name in CAPTURE_LAYERS:
        record[name] = states[name][batch_index][mask].contiguous()
    record["latent_mean"] = mean[batch_index][mask].contiguous()
    n = int(mask.sum())
    for name in CAPTURE_LAYERS:
        if record[name].shape != (n, 768):
            raise AssertionError(f"{name} shape mismatch: {record[name].shape}")
    if record["latent_mean"].shape != (n, 8):
        raise AssertionError(f"latent_mean shape mismatch: {record['latent_mean'].shape}")
    for name in (*CAPTURE_LAYERS, "latent_mean"):
        if record[name].dtype != torch.float32 or not torch.isfinite(record[name]).all():
            raise AssertionError(f"invalid captured tensor: {name}")
    for key in ("original_dense_row_index", "residue_pdb_idx", "chain_index", "aa_integer"):
        if record[key].shape != (n,):
            raise AssertionError(f"metadata alignment mismatch: {key}")
    if len(record["aa_label"]) != n:
        raise AssertionError("AA label alignment mismatch")
    return record


def select_pool_indices(pool: ProteinPool, count: int | None, seed: int) -> list[int]:
    by_split = {"probe_train": [], "probe_test": []}
    for index, protein_id in enumerate(pool.protein_ids):
        by_split[pool.probe_split[protein_id]].append(index)
    for values in by_split.values():
        values.sort(key=lambda index: pool.protein_ids[index])
    if count is None:
        return sorted(range(len(pool.protein_ids)), key=lambda index: pool.protein_ids[index])
    if count < 1 or count > len(pool.protein_ids):
        raise ValueError(f"invalid selection count: {count}")
    n_test = int(round(count * len(by_split["probe_test"]) / len(pool.protein_ids)))
    n_train = count - n_test
    rng = random.Random(seed)
    chosen = rng.sample(by_split["probe_train"], n_train)
    chosen += rng.sample(by_split["probe_test"], n_test)
    return sorted(chosen, key=lambda index: pool.protein_ids[index])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_saved_output(output_dir: Path, expected_proteins: int, expected_residues: int) -> None:
    proteins = residues = 0
    seen: set[str] = set()
    for path in sorted(output_dir.glob("shard-*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(shard, list) or not 1 <= len(shard) <= SHARD_SIZE:
            raise AssertionError(f"invalid shard record count: {path}")
        for record in shard:
            protein_id = record["protein_id"]
            if protein_id in seen:
                raise AssertionError(f"duplicate protein record: {protein_id}")
            seen.add(protein_id)
            n = int(record["latent_mean"].shape[0])
            if record["latent_mean"].shape != (n, 8):
                raise AssertionError(f"invalid latent shape in {path}")
            for name in CAPTURE_LAYERS:
                if record[name].shape != (n, 768):
                    raise AssertionError(f"invalid {name} shape in {path}")
            proteins += 1
            residues += n
        del shard
    if proteins != expected_proteins or residues != expected_residues:
        raise AssertionError(
            f"saved count mismatch: proteins={proteins}/{expected_proteins}, "
            f"residues={residues}/{expected_residues}"
        )


def run_capture(
    encoder: torch.nn.Module,
    pool: ProteinPool,
    indices: list[int],
    output_dir: Path,
    device: str,
    provenance: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = LayerCapture(encoder)
    shard_records: list[dict[str, Any]] = []
    shard_metadata = []
    total_residues = 0
    split_counts = {"probe_train": 0, "probe_test": 0}
    try:
        for ordinal, pool_index in enumerate(indices):
            graph = pool.dataset[pool_index]
            batch = preprocess_graphs([graph])
            metadata = metadata_for_row(pool, pool_index, batch, 0)
            states, mean = capture_batch(encoder, capture, batch, device)
            record = record_from_capture(metadata, states, mean, batch, 0)
            total_residues += int(record["latent_mean"].shape[0])
            split_counts[record["probe_split"]] += 1
            shard_records.append(record)
            if len(shard_records) == SHARD_SIZE or ordinal + 1 == len(indices):
                shard_index = len(shard_metadata)
                shard_path = output_dir / f"shard-{shard_index:05d}.pt"
                _atomic_torch_save(shard_records, shard_path)
                shard_metadata.append(
                    {
                        "path": shard_path.name,
                        "sha256": sha256_file(shard_path),
                        "size_bytes": shard_path.stat().st_size,
                        "proteins": len(shard_records),
                        "residues": sum(int(r["latent_mean"].shape[0]) for r in shard_records),
                    }
                )
                shard_records.clear()
    finally:
        capture.remove()

    manifest = {
        "schema_version": 1,
        "artifact": "full_context_layer_states",
        "mode": mode,
        "provenance": provenance,
        "capture": {
            "context": "full_unmasked_evaluation",
            "rotation": None,
            "batch_size": 1,
            "dtype": "float32",
            "blocks": {name: index + 1 for name, index in CAPTURE_LAYERS.items()},
            "hidden_width": 768,
            "latent_representation": "mean",
            "latent_width": 8,
            "sampled_z_latent_saved": False,
            "encoder_validity_expression": VALIDITY_EXPRESSION,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "strategy": "all-sorted" if len(indices) == len(pool.protein_ids) else "stratified-existing-probe-split",
            "selected_protein_ids": [pool.protein_ids[index] for index in indices],
            "probe_split_counts": split_counts,
        },
        "counts": {"proteins": len(indices), "encoder_valid_residues": total_residues},
        "shard_size_proteins": SHARD_SIZE,
        "shards": shard_metadata,
    }
    _atomic_json_save(manifest, output_dir / "manifest.json")
    validate_saved_output(output_dir, len(indices), total_residues)
    return manifest


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError("saved tensor round-trip mismatch")
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError("saved mapping keys mismatch")
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError("saved list length mismatch")
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    elif left != right:
        raise AssertionError(f"saved value mismatch: {left!r} != {right!r}")


def run_integration_tests(
    encoder: torch.nn.Module, pool: ProteinPool, device: str
) -> dict[str, str]:
    indices = select_pool_indices(pool, 2, SELECTION_SEED)
    if len({_graph_row_count(pool.dataset[index]) for index in indices}) != 2:
        candidates = select_pool_indices(pool, 20, SELECTION_SEED)
        first = candidates[0]
        first_length = _graph_row_count(pool.dataset[first])
        second = next(
            index
            for index in candidates[1:]
            if _graph_row_count(pool.dataset[index]) != first_length
        )
        indices = [first, second]

    single = preprocess_graphs([pool.dataset[indices[0]]])
    with torch.inference_mode():
        baseline_mean = encoder(single.to(device))["mean"].detach().float().cpu()
    capture = LayerCapture(encoder)
    head_inputs: list[torch.Tensor] = []

    def head_pre_hook(module, args):
        del module
        head_inputs.append(args[0].detach().float().cpu())

    head_handle = encoder.latent_decoder_mean_n_log_scale.register_forward_pre_hook(
        head_pre_hook
    )
    try:
        states, captured_mean = capture_batch(encoder, capture, single, device)
        torch.testing.assert_close(
            captured_mean, baseline_mean, rtol=1e-5, atol=1e-6
        )
        if len(head_inputs) != 1:
            raise AssertionError(f"latent head pre-hook count: {len(head_inputs)}")
        torch.testing.assert_close(states["block_12"], head_inputs[0], rtol=0, atol=0)
        with torch.inference_mode():
            projected = encoder.latent_decoder_mean_n_log_scale(
                head_inputs[0].to(device)
            ).detach().float().cpu()
        valid = single["mask"].cpu()
        torch.testing.assert_close(
            projected[..., :8][valid], captured_mean[valid], rtol=1e-5, atol=1e-6
        )

        padded = preprocess_graphs([pool.dataset[index] for index in indices])
        padded_states, padded_mean = capture_batch(encoder, capture, padded, device)
        records = []
        for batch_index, pool_index in enumerate(indices):
            metadata = metadata_for_row(pool, pool_index, padded, batch_index)
            record = record_from_capture(
                metadata, padded_states, padded_mean, padded, batch_index
            )
            expected_mask = padded["mask_dict"]["coords"][batch_index, ..., 0, 0].bool().cpu()
            if not torch.equal(record["original_dense_row_index"], torch.nonzero(expected_mask).squeeze(-1)):
                raise AssertionError("saved dense rows do not match encoder-valid rows")
            if not torch.equal(
                record["residue_pdb_idx"],
                padded["residue_pdb_idx"][batch_index].cpu()[expected_mask],
            ):
                raise AssertionError("saved PDB indices do not match validity selection")
            if not torch.equal(
                record["aa_integer"], padded["residue_type"][batch_index].cpu()[expected_mask]
            ):
                raise AssertionError("saved AA integers do not match validity selection")
            records.append(record)
        if padded["mask"].all():
            raise AssertionError("two-length padding test did not contain padding")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.pt"
            _atomic_torch_save(records, path)
            reloaded = torch.load(path, map_location="cpu", weights_only=False)
            _assert_nested_equal(records, reloaded)
            if records[0]["protein_id"] == records[1]["protein_id"]:
                raise AssertionError("stale capture isolation test reused a protein")
    finally:
        head_handle.remove()
        capture.remove()

    return {
        "ema_loading": "passed",
        "forward_parity": "passed",
        "capture_correctness": "passed",
        "shapes_and_padding_alignment": "passed",
        "saving_and_isolation": "passed",
    }


def _stage_output_dir(output_root: Path, name: str) -> Path:
    return output_root / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="tests,smoke,pilot,full",
        help="comma-separated subset of tests,smoke,pilot,full",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    invalid = set(stages) - {"tests", "smoke", "pilot", "full"}
    if invalid:
        raise ValueError(f"unknown stages: {sorted(invalid)}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    encoder, provenance = load_verified_encoder(args.device)
    pool = build_protein_pool(seed=SELECTION_SEED)
    if len(pool.protein_ids) != EXPECTED_PROTEINS:
        raise AssertionError(
            f"pool protein count mismatch: {len(pool.protein_ids)} != {EXPECTED_PROTEINS}"
        )

    run_summary: dict[str, Any] = {"provenance": provenance}
    if "tests" in stages:
        run_summary["tests"] = run_integration_tests(encoder, pool, args.device)

    stage_counts = {"smoke": 5, "pilot": 100, "full": None}
    for stage in ("smoke", "pilot", "full"):
        if stage not in stages:
            continue
        indices = select_pool_indices(pool, stage_counts[stage], SELECTION_SEED)
        expected_residues = sum(
            _graph_encoder_valid_count(pool.dataset[index]) for index in indices
        )
        if stage == "full" and expected_residues != EXPECTED_RESIDUES:
            raise AssertionError(
                f"full-pool encoder-valid residue mismatch before extraction: "
                f"{expected_residues} != {EXPECTED_RESIDUES}"
            )
        manifest = run_capture(
            encoder,
            pool,
            indices,
            _stage_output_dir(output_root, stage),
            args.device,
            provenance,
            stage,
        )
        if manifest["counts"]["encoder_valid_residues"] != expected_residues:
            raise AssertionError(
                f"{stage} residue mismatch: {manifest['counts']} expected={expected_residues}"
            )
        run_summary[stage] = {
            "output": str(_stage_output_dir(output_root, stage)),
            "counts": manifest["counts"],
            "probe_split_counts": manifest["selection"]["probe_split_counts"],
            "storage_bytes": sum(item["size_bytes"] for item in manifest["shards"]),
        }

    _atomic_json_save(run_summary, output_root / "run_summary.json")
    print(json.dumps(run_summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
