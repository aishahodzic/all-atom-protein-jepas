"""Count strict 3--5-residue matched neighbour-intervention groups."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from Evaluation.canonical_lejepa.check_neighbor_influence_eligibility import (
    CA_INDEX,
    FAR_NM,
    INPUT_DIR,
    MAX_MAGNITUDE_MISMATCH,
    MIN_RELATIVE_INTERVENTION,
    MIN_SEQUENCE_SEPARATION,
    NEAR_NM,
    N_PROTEINS,
    OUTPUT_DIR,
    SEED,
    SEQUENCE_TOLERANCE,
    _chain_segments,
    _iter_records,
    _relative_intervention,
)
from Evaluation.linear_probing.data_pool import build_protein_pool, eval_transform_compose


OUTPUT_NAME = "neighbor_group_eligibility_probe_test100_seed42.json"
GROUP_SIZES = (3, 4, 5)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _options_for_near(
    focal: int,
    near: int,
    observed: torch.Tensor,
    distances: torch.Tensor,
    segments: torch.Tensor,
    aa: torch.Tensor,
    hidden: torch.Tensor,
) -> list[dict[str, Any]]:
    near_sep = abs(focal - near)
    far_candidates = []
    for residue in range(len(observed)):
        if residue in (focal, near) or not bool(observed[residue]):
            continue
        if segments[residue] != segments[focal] or aa[residue] != aa[near]:
            continue
        if float(distances[focal, residue]) < FAR_NM:
            continue
        separation = abs(focal - residue)
        if separation <= MIN_SEQUENCE_SEPARATION or abs(separation - near_sep) > SEQUENCE_TOLERANCE:
            continue
        far_candidates.append(residue)

    options = []
    for control in far_candidates:
        for donor in far_candidates:
            if control == donor:
                continue
            near_l2, near_relative = _relative_intervention(hidden[donor], hidden[near])
            far_l2, far_relative = _relative_intervention(hidden[donor], hidden[control])
            if near_relative < MIN_RELATIVE_INTERVENTION or far_relative < MIN_RELATIVE_INTERVENTION:
                continue
            mismatch = abs(near_l2 - far_l2) / (0.5 * (near_l2 + far_l2))
            if mismatch > MAX_MAGNITUDE_MISMATCH:
                continue
            options.append(
                {
                    "near": near,
                    "control": control,
                    "donor": donor,
                    "near_l2": near_l2,
                    "far_l2": far_l2,
                    "mismatch": mismatch,
                }
            )
    return sorted(options, key=lambda item: (item["control"], item["donor"]))


def _find_group(
    focal: int,
    near_options: list[tuple[int, list[dict[str, Any]]]],
    size: int,
) -> list[dict[str, Any]] | None:
    # Fewest-options-first makes strict disjoint matching fast and deterministic.
    ordered = sorted(near_options, key=lambda item: (len(item[1]), item[0]))

    def search(
        start: int, chosen: list[dict[str, Any]], used: set[int]
    ) -> list[dict[str, Any]] | None:
        if len(chosen) == size:
            near_total = sum(item["near_l2"] ** 2 for item in chosen) ** 0.5
            far_total = sum(item["far_l2"] ** 2 for item in chosen) ** 0.5
            mismatch = abs(near_total - far_total) / (0.5 * (near_total + far_total))
            if mismatch <= MAX_MAGNITUDE_MISMATCH:
                return chosen
            return None
        if len(ordered) - start < size - len(chosen):
            return None
        for position in range(start, len(ordered)):
            near, options = ordered[position]
            if near in used:
                continue
            for option in options:
                control, donor = option["control"], option["donor"]
                if control in used or donor in used:
                    continue
                result = search(
                    position + 1,
                    chosen + [option],
                    used | {near, control, donor},
                )
                if result is not None:
                    return result
        return None

    return search(0, [], {focal})


def evaluate(
    graph: Any, record: dict[str, Any]
) -> tuple[dict[int, dict[str, Any] | None], int]:
    observed = graph.coord_mask[:, CA_INDEX].clone().bool()
    graph = eval_transform_compose()(graph)
    coords = graph.coords_nm[:, CA_INDEX].float()
    distances = torch.cdist(coords, coords)
    segments = _chain_segments(
        graph.chains.to(torch.int64), graph.chain_breaks_per_residue.bool()
    )
    aa = graph.residue_type.to(torch.int64)
    hidden = record["block_1"]
    if hidden.shape[0] != len(observed):
        raise AssertionError("state/graph row mismatch")
    found: dict[int, dict[str, Any] | None] = {size: None for size in GROUP_SIZES}
    maximum_structural_neighbours = 0
    for focal in range(len(observed)):
        if not bool(observed[focal]):
            continue
        near_options = []
        for near in range(len(observed)):
            if near == focal or not bool(observed[near]):
                continue
            if segments[near] != segments[focal]:
                continue
            if abs(focal - near) <= MIN_SEQUENCE_SEPARATION:
                continue
            if float(distances[focal, near]) > NEAR_NM:
                continue
            options = _options_for_near(
                focal, near, observed, distances, segments, aa, hidden
            )
            if options:
                near_options.append((near, options))
        structural_neighbours = sum(
            1
            for near in range(len(observed))
            if near != focal
            and bool(observed[near])
            and segments[near] == segments[focal]
            and abs(focal - near) > MIN_SEQUENCE_SEPARATION
            and float(distances[focal, near]) <= NEAR_NM
        )
        maximum_structural_neighbours = max(
            maximum_structural_neighbours, structural_neighbours
        )
        for size in GROUP_SIZES:
            if found[size] is not None or len(near_options) < size:
                continue
            group = _find_group(focal, near_options, size)
            if group is not None:
                near_total = sum(item["near_l2"] ** 2 for item in group) ** 0.5
                far_total = sum(item["far_l2"] ** 2 for item in group) ** 0.5
                found[size] = {
                    "focal_i": focal,
                    "near_recipients": [item["near"] for item in group],
                    "far_recipients": [item["control"] for item in group],
                    "donors": [item["donor"] for item in group],
                    "near_group_l2": near_total,
                    "far_group_l2": far_total,
                    "group_magnitude_mismatch": abs(near_total - far_total)
                    / (0.5 * (near_total + far_total)),
                }
        if all(found.values()):
            break
    return found, maximum_structural_neighbours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    with (input_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    pool = build_protein_pool(seed=SEED)
    test_ids = sorted(
        protein_id
        for protein_id in pool.protein_ids
        if pool.probe_split[protein_id] == "probe_test"
    )
    selected = sorted(random.Random(SEED).sample(test_ids, N_PROTEINS))
    selected_set = set(selected)
    records = {
        record["protein_id"]: record
        for record in _iter_records(input_dir, manifest)
        if record["protein_id"] in selected_set
    }
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    results = []
    counts = Counter()
    structural_counts = Counter()
    for ordinal, protein_id in enumerate(selected, start=1):
        found, maximum_structural_neighbours = evaluate(
            pool.dataset[pool_index[protein_id]], records[protein_id]
        )
        row = {
            "protein_id": protein_id,
            "maximum_structural_neighbours": maximum_structural_neighbours,
            "eligible": {str(k): v for k, v in found.items()},
        }
        results.append(row)
        for size, group in found.items():
            if group is not None:
                counts[size] += 1
            if maximum_structural_neighbours >= size:
                structural_counts[size] += 1
        print(f"[{ordinal:03d}/{len(selected)}] {protein_id} " + " ".join(
            f"k{size}={'yes' if found[size] else 'no'}" for size in GROUP_SIZES
        ), flush=True)
    report = {
        "purpose": "eligibility_only_no_encoder_forwards_or_interventions",
        "selected_proteins": len(selected),
        "counts": {str(size): counts[size] for size in GROUP_SIZES},
        "structural_ceiling_counts": {
            str(size): structural_counts[size] for size in GROUP_SIZES
        },
        "rules": {
            "near_residues_per_group": list(GROUP_SIZES),
            "near_to_focal_max_nm": NEAR_NM,
            "control_and_donor_to_focal_min_nm": FAR_NM,
            "minimum_sequence_separation": MIN_SEQUENCE_SEPARATION,
            "sequence_tolerance_per_matched_triple": SEQUENCE_TOLERANCE,
            "same_aa_per_near_control_donor_triple": True,
            "unique_recipients_and_donors_within_group": True,
            "per_triple_maximum_magnitude_mismatch": MAX_MAGNITUDE_MISMATCH,
            "group_maximum_magnitude_mismatch": MAX_MAGNITUDE_MISMATCH,
            "same_chain_segment_and_observed_ca": True,
        },
        "results": results,
    }
    output = args.output_dir.resolve() / OUTPUT_NAME
    _atomic_json(report, output)
    print(json.dumps({"counts": report["counts"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
