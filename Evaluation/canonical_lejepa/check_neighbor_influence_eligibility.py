"""Eligibility funnel for the proposed canonical LeJEPA neighbour intervention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from openfold.np import residue_constants

from Evaluation.linear_probing.data_pool import build_protein_pool, eval_transform_compose


REPO = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO / (
    "store/canonical_lejepa_v1_100k/evaluation/"
    "full_context_layer_states/full"
)
OUTPUT_DIR = REPO / "Evaluation/canonical_lejepa/results"
OUTPUT_STEM = "neighbor_influence_eligibility_probe_test100_seed42"
SEED = 42
N_PROTEINS = 100
NEAR_NM = 0.8
FAR_NM = 1.5
MIN_SEQUENCE_SEPARATION = 20
SEQUENCE_TOLERANCE = 5
MAX_MAGNITUDE_MISMATCH = 0.10
MIN_RELATIVE_INTERVENTION = 1e-3
CA_INDEX = 1


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _iter_records(input_dir: Path, manifest: dict[str, Any]):
    for shard in manifest["shards"]:
        records = torch.load(
            input_dir / shard["path"], map_location="cpu", weights_only=False
        )
        for record in records:
            yield record
        del records


def _relative_intervention(source: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    magnitude = float(torch.linalg.vector_norm(source - target).item())
    scale = 0.5 * float(
        (torch.linalg.vector_norm(source) + torch.linalg.vector_norm(target)).item()
    )
    relative = magnitude / max(scale, torch.finfo(source.dtype).eps)
    return magnitude, relative


def _stable_key(protein_id: str, indices: tuple[int, int, int, int]) -> str:
    payload = f"{SEED}|{protein_id}|" + "|".join(str(value) for value in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chain_segments(chain_indices: torch.Tensor, breaks: torch.Tensor) -> torch.Tensor:
    if len(chain_indices) != len(breaks):
        raise AssertionError("chain/break length mismatch")
    segments = torch.zeros_like(chain_indices, dtype=torch.int64)
    segment = 0
    for index in range(1, len(chain_indices)):
        if chain_indices[index] != chain_indices[index - 1] or bool(breaks[index - 1]):
            segment += 1
        segments[index] = segment
    return segments


def evaluate_protein(
    protein_id: str,
    graph: Any,
    record: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, int]]:
    raw_coord_mask = graph.coord_mask.clone().bool()
    graph = eval_transform_compose()(graph)
    ca_observed = raw_coord_mask[:, CA_INDEX]
    ca_coords_nm = graph.coords_nm[:, CA_INDEX].float()
    n = len(ca_observed)
    if record["probe_split"] != "probe_test":
        raise AssertionError(f"selected non-test protein: {protein_id}")
    if record["block_1"].shape != (n, 768):
        raise AssertionError(f"Block-1 alignment failure: {protein_id}")
    rows = record["original_dense_row_index"]
    if not torch.equal(rows, torch.arange(n, dtype=rows.dtype)):
        raise AssertionError(f"dense-row alignment failure: {protein_id}")
    if not torch.equal(record["aa_integer"], graph.residue_type.to(torch.int64)):
        raise AssertionError(f"AA alignment failure: {protein_id}")
    if not torch.equal(record["residue_pdb_idx"], graph.residue_pdb_idx.to(torch.int64)):
        raise AssertionError(f"PDB-index alignment failure: {protein_id}")

    diagnostics = {
        "observed_ca": int(ca_observed.sum().item()),
        "near_pairs": 0,
        "same_aa_far_stage": 0,
        "sequence_matched_stage": 0,
        "nonnegligible_quartets": 0,
        "magnitude_matched_quartets": 0,
    }
    if diagnostics["observed_ca"] < 4:
        return "insufficient_observed_ca", None, diagnostics

    distances = torch.cdist(ca_coords_nm, ca_coords_nm)
    chain_indices = graph.chains.to(torch.int64)
    segments = _chain_segments(chain_indices, graph.chain_breaks_per_residue.bool())
    aa = graph.residue_type.to(torch.int64)
    hidden = record["block_1"]
    candidates: list[dict[str, Any]] = []

    for focal in range(n):
        if not bool(ca_observed[focal]):
            continue
        for near in range(n):
            if focal == near or not bool(ca_observed[near]):
                continue
            near_sep = abs(focal - near)
            if (
                near_sep <= MIN_SEQUENCE_SEPARATION
                or segments[focal] != segments[near]
                or float(distances[focal, near]) > NEAR_NM
            ):
                continue
            diagnostics["near_pairs"] += 1

            same_aa_far = []
            for residue in range(n):
                if residue in (focal, near) or not bool(ca_observed[residue]):
                    continue
                if segments[residue] != segments[focal]:
                    continue
                if aa[residue] != aa[near] or float(distances[focal, residue]) < FAR_NM:
                    continue
                if abs(focal - residue) <= MIN_SEQUENCE_SEPARATION:
                    continue
                same_aa_far.append(residue)
            if len(same_aa_far) < 2:
                continue
            diagnostics["same_aa_far_stage"] += 1
            sequence_matched = [
                residue
                for residue in same_aa_far
                if abs(abs(focal - residue) - near_sep) <= SEQUENCE_TOLERANCE
            ]
            if len(sequence_matched) < 2:
                continue
            diagnostics["sequence_matched_stage"] += 1

            for control in sequence_matched:
                for donor in sequence_matched:
                    if donor == control:
                        continue
                    near_magnitude, near_relative = _relative_intervention(
                        hidden[donor], hidden[near]
                    )
                    far_magnitude, far_relative = _relative_intervention(
                        hidden[donor], hidden[control]
                    )
                    if (
                        near_relative < MIN_RELATIVE_INTERVENTION
                        or far_relative < MIN_RELATIVE_INTERVENTION
                    ):
                        continue
                    diagnostics["nonnegligible_quartets"] += 1
                    mean_magnitude = 0.5 * (near_magnitude + far_magnitude)
                    mismatch = abs(near_magnitude - far_magnitude) / mean_magnitude
                    if mismatch > MAX_MAGNITUDE_MISMATCH:
                        continue
                    diagnostics["magnitude_matched_quartets"] += 1
                    indices = (focal, near, control, donor)
                    candidates.append(
                        {
                            "selection_sha256": _stable_key(protein_id, indices),
                            "protein_id": protein_id,
                            "focal_i": focal,
                            "near_j": near,
                            "far_control_k": control,
                            "shared_donor_d": donor,
                            "aa_integer": int(aa[near]),
                            "aa_label": residue_constants.restypes_with_x[int(aa[near])],
                            "near_distance_nm": float(distances[focal, near]),
                            "far_control_distance_nm": float(distances[focal, control]),
                            "donor_distance_nm": float(distances[focal, donor]),
                            "near_sequence_separation": near_sep,
                            "far_control_sequence_separation": abs(focal - control),
                            "donor_sequence_separation": abs(focal - donor),
                            "near_intervention_l2": near_magnitude,
                            "far_intervention_l2": far_magnitude,
                            "near_intervention_relative": near_relative,
                            "far_intervention_relative": far_relative,
                            "intervention_magnitude_relative_mismatch": mismatch,
                        }
                    )

    if candidates:
        return "eligible", min(candidates, key=lambda item: item["selection_sha256"]), diagnostics
    if diagnostics["near_pairs"] == 0:
        reason = "no_long_range_near_pair"
    elif diagnostics["same_aa_far_stage"] == 0:
        reason = "no_two_same_aa_far_residues"
    elif diagnostics["sequence_matched_stage"] == 0:
        reason = "no_sequence_matched_far_pair"
    elif diagnostics["nonnegligible_quartets"] == 0:
        reason = "negligible_intervention"
    else:
        reason = "intervention_magnitude_mismatch"
    return reason, None, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    with (input_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["counts"]["proteins"] != 5224:
        raise AssertionError("unexpected extraction pool")
    pool = build_protein_pool(seed=SEED)
    test_ids = sorted(
        protein_id
        for protein_id in pool.protein_ids
        if pool.probe_split[protein_id] == "probe_test"
    )
    rng = random.Random(SEED)
    selected_ids = sorted(rng.sample(test_ids, N_PROTEINS))
    selected_set = set(selected_ids)
    records = {
        record["protein_id"]: record
        for record in _iter_records(input_dir, manifest)
        if record["protein_id"] in selected_set
    }
    if set(records) != selected_set:
        raise AssertionError("missing selected extraction records")
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}

    selected_quartets = []
    exclusions = Counter()
    per_protein = []
    for protein_id in selected_ids:
        reason, quartet, diagnostics = evaluate_protein(
            protein_id,
            pool.dataset[pool_index[protein_id]],
            records[protein_id],
        )
        exclusions[reason] += 1
        per_protein.append({"protein_id": protein_id, "status": reason, **diagnostics})
        if quartet is not None:
            selected_quartets.append(quartet)

    config = {
        "seed": SEED,
        "selected_probe_test_proteins": N_PROTEINS,
        "near_distance_nm_max": NEAR_NM,
        "far_distance_nm_min": FAR_NM,
        "minimum_sequence_separation": MIN_SEQUENCE_SEPARATION,
        "sequence_separation_tolerance": SEQUENCE_TOLERANCE,
        "maximum_intervention_magnitude_relative_mismatch": MAX_MAGNITUDE_MISMATCH,
        "minimum_scale_relative_intervention": MIN_RELATIVE_INTERVENTION,
        "same_chain_segment_required": True,
        "observed_ca_required": True,
        "shared_donor": True,
        "encoder_forward_run": False,
    }
    report = {
        "purpose": "eligibility_only_no_interventions",
        "config": config,
        "input_manifest": str(input_dir / "manifest.json"),
        "checkpoint_provenance": manifest["provenance"],
        "selected_protein_ids": selected_ids,
        "counts": {
            "selected": N_PROTEINS,
            "eligible": len(selected_quartets),
            "excluded": N_PROTEINS - len(selected_quartets),
            "status": dict(sorted(exclusions.items())),
        },
        "selected_quartets": selected_quartets,
        "per_protein_funnel": per_protein,
    }
    json_path = output_dir / f"{OUTPUT_STEM}.json"
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    _atomic_write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", json_path)
    if selected_quartets:
        with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected_quartets[0]))
            writer.writeheader()
            writer.writerows(selected_quartets)
            handle.seek(0)
            csv_text = handle.read()
        _atomic_write_text(csv_text, csv_path)
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
