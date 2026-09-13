"""Eligibility for revised matched three-residue neighbour interventions."""

from __future__ import annotations

import argparse
import hashlib
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


OUTPUT_NAME = "neighbor_group_signed_pair50_eligibility_probe_test100_seed42.json"
GROUP_SIZE = 3
MAX_SINGLE_ENERGY_FRACTION = 0.50
MAX_PAIR_MAGNITUDE_MISMATCH = 0.50
SEQ_SEP_DIM = 127


def _sequence_separation_bin(signed_offset: int) -> int:
    # The 127D feature has half-integer boundaries from -62.5 to 62.5.
    terminal = SEQ_SEP_DIM // 2
    return max(-terminal, min(terminal, signed_offset)) + terminal


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


def _stable_key(protein_id: str, values: tuple[int, ...]) -> str:
    payload = f"{SEED}|{protein_id}|" + "|".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_group(group: list[dict[str, Any]]) -> tuple[bool, dict[str, float]]:
    near_sq = torch.tensor([item["near_l2"] ** 2 for item in group], dtype=torch.float64)
    far_sq = torch.tensor([item["far_l2"] ** 2 for item in group], dtype=torch.float64)
    near_total = float(torch.sqrt(near_sq.sum()).item())
    far_total = float(torch.sqrt(far_sq.sum()).item())
    mismatch = abs(near_total - far_total) / (0.5 * (near_total + far_total))
    near_max_fraction = float((near_sq.max() / near_sq.sum()).item())
    far_max_fraction = float((far_sq.max() / far_sq.sum()).item())
    diagnostics = {
        "near_group_l2": near_total,
        "far_group_l2": far_total,
        "group_magnitude_relative_mismatch": mismatch,
        "near_maximum_single_energy_fraction": near_max_fraction,
        "far_maximum_single_energy_fraction": far_max_fraction,
    }
    valid = (
        mismatch <= MAX_MAGNITUDE_MISMATCH
        and near_max_fraction <= MAX_SINGLE_ENERGY_FRACTION
        and far_max_fraction <= MAX_SINGLE_ENERGY_FRACTION
    )
    return valid, diagnostics


def _find_group(
    protein_id: str,
    focal: int,
    options_by_near: list[tuple[int, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]] | None, dict[str, int]]:
    diagnostics = {"complete_disjoint_groups": 0, "group_magnitude_matches": 0}
    ordered = sorted(options_by_near, key=lambda item: (len(item[1]), item[0]))

    def search(
        start: int, chosen: list[dict[str, Any]], used: set[int]
    ) -> list[dict[str, Any]] | None:
        if len(chosen) == GROUP_SIZE:
            diagnostics["complete_disjoint_groups"] += 1
            valid, group_diagnostics = _valid_group(chosen)
            if valid:
                diagnostics["group_magnitude_matches"] += 1
                selected = [dict(item) for item in chosen]
                selected[0]["group_diagnostics"] = group_diagnostics
                return selected
            return None
        if len(ordered) - start < GROUP_SIZE - len(chosen):
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

    return search(0, [], {focal}), diagnostics


def evaluate_protein(
    protein_id: str, graph: Any, record: dict[str, Any]
) -> tuple[str, dict[str, Any] | None, dict[str, int]]:
    observed = graph.coord_mask[:, CA_INDEX].clone().bool()
    graph = eval_transform_compose()(graph)
    distances = torch.cdist(graph.coords_nm[:, CA_INDEX].float(), graph.coords_nm[:, CA_INDEX].float())
    segments = _chain_segments(
        graph.chains.to(torch.int64), graph.chain_breaks_per_residue.bool()
    )
    aa = graph.residue_type.to(torch.int64)
    hidden = record["block_1"]
    if hidden.shape != (len(observed), 768):
        raise AssertionError(f"state/graph mismatch: {protein_id}")

    diagnostics = {
        "focals_with_three_structural_neighbours": 0,
        "focals_with_three_neighbours_having_controls": 0,
        "focals_with_three_neighbours_having_donors": 0,
        "complete_disjoint_groups": 0,
        "group_magnitude_matches": 0,
    }
    selected_groups: list[dict[str, Any]] = []
    for focal in range(len(observed)):
        if not bool(observed[focal]):
            continue
        near_residues = [
            residue
            for residue in range(len(observed))
            if residue != focal
            and bool(observed[residue])
            and segments[residue] == segments[focal]
            and abs(focal - residue) > MIN_SEQUENCE_SEPARATION
            and float(distances[focal, residue]) <= NEAR_NM
        ]
        if len(near_residues) < GROUP_SIZE:
            continue
        diagnostics["focals_with_three_structural_neighbours"] += 1
        options_by_near = []
        neighbours_with_controls = 0
        neighbours_with_donors = 0
        for near in near_residues:
            near_signed_sep = near - focal
            near_sep = abs(near_signed_sep)
            controls = [
                residue
                for residue in range(len(observed))
                if residue not in (focal, near)
                and bool(observed[residue])
                and segments[residue] == segments[focal]
                and aa[residue] == aa[near]
                and abs(focal - residue) > MIN_SEQUENCE_SEPARATION
                and _sequence_separation_bin(residue - focal)
                == _sequence_separation_bin(near_signed_sep)
                and float(distances[focal, residue]) >= FAR_NM
            ]
            if controls:
                neighbours_with_controls += 1
            donors = [
                residue
                for residue in range(len(observed))
                if residue not in (focal, near)
                and bool(observed[residue])
                and segments[residue] == segments[focal]
                and aa[residue] == aa[near]
                and abs(focal - residue) > MIN_SEQUENCE_SEPARATION
                and float(distances[focal, residue]) >= FAR_NM
            ]
            options = []
            for control in controls:
                for donor in donors:
                    if donor == control:
                        continue
                    near_l2, near_relative = _relative_intervention(
                        hidden[donor], hidden[near]
                    )
                    far_l2, far_relative = _relative_intervention(
                        hidden[donor], hidden[control]
                    )
                    if (
                        near_relative < MIN_RELATIVE_INTERVENTION
                        or far_relative < MIN_RELATIVE_INTERVENTION
                    ):
                        continue
                    pair_mismatch = abs(near_l2 - far_l2) / (
                        0.5 * (near_l2 + far_l2)
                    )
                    if pair_mismatch > MAX_PAIR_MAGNITUDE_MISMATCH:
                        continue
                    options.append(
                        {
                            "near": near,
                            "control": control,
                            "donor": donor,
                            "aa_integer": int(aa[near]),
                            "near_distance_nm": float(distances[focal, near]),
                            "control_distance_nm": float(distances[focal, control]),
                            "donor_distance_nm": float(distances[focal, donor]),
                            "near_sequence_separation": near_sep,
                            "control_sequence_separation": abs(focal - control),
                            "donor_sequence_separation": abs(focal - donor),
                            "near_signed_sequence_separation": near - focal,
                            "control_signed_sequence_separation": control - focal,
                            "donor_signed_sequence_separation": donor - focal,
                            "near_l2": near_l2,
                            "far_l2": far_l2,
                            "near_relative_intervention": near_relative,
                            "far_relative_intervention": far_relative,
                            "pair_magnitude_relative_mismatch": pair_mismatch,
                        }
                    )
            if options:
                neighbours_with_donors += 1
                options.sort(
                    key=lambda item: _stable_key(
                        protein_id,
                        (focal, item["near"], item["control"], item["donor"]),
                    )
                )
                options_by_near.append((near, options))
        if neighbours_with_controls >= GROUP_SIZE:
            diagnostics["focals_with_three_neighbours_having_controls"] += 1
        if neighbours_with_donors >= GROUP_SIZE:
            diagnostics["focals_with_three_neighbours_having_donors"] += 1
        if len(options_by_near) < GROUP_SIZE:
            continue
        group, search_diagnostics = _find_group(
            protein_id, focal, options_by_near
        )
        diagnostics["complete_disjoint_groups"] += search_diagnostics[
            "complete_disjoint_groups"
        ]
        diagnostics["group_magnitude_matches"] += search_diagnostics[
            "group_magnitude_matches"
        ]
        if group is not None:
            group_diagnostics = group[0].pop("group_diagnostics")
            values = tuple(
                value
                for item in group
                for value in (item["near"], item["control"], item["donor"])
            )
            selected_groups.append(
                {
                    "selection_sha256": _stable_key(protein_id, (focal,) + values),
                    "protein_id": protein_id,
                    "focal_i": focal,
                    "triples": group,
                    **group_diagnostics,
                }
            )
    if selected_groups:
        return "eligible", min(
            selected_groups, key=lambda item: item["selection_sha256"]
        ), diagnostics
    if diagnostics["focals_with_three_structural_neighbours"] == 0:
        reason = "no_three_structural_neighbours"
    elif diagnostics["focals_with_three_neighbours_having_controls"] == 0:
        reason = "no_three_same_aa_sequence_matched_controls"
    elif diagnostics["focals_with_three_neighbours_having_donors"] == 0:
        reason = "no_three_donor_options"
    elif diagnostics["complete_disjoint_groups"] == 0:
        reason = "no_disjoint_group"
    else:
        reason = "group_magnitude_or_dominance_mismatch"
    return reason, None, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--all-probe-test", action="store_true")
    parser.add_argument("--output-name", default=OUTPUT_NAME)
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
    selected_ids = (
        test_ids
        if args.all_probe_test
        else sorted(random.Random(SEED).sample(test_ids, N_PROTEINS))
    )
    selected_set = set(selected_ids)
    records = {
        record["protein_id"]: record
        for record in _iter_records(input_dir, manifest)
        if record["protein_id"] in selected_set
    }
    if set(records) != selected_set:
        raise AssertionError("missing selected state records")
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}

    status = Counter()
    direction_balance = Counter()
    selected_groups = []
    per_protein = []
    for ordinal, protein_id in enumerate(selected_ids, start=1):
        reason, group, diagnostics = evaluate_protein(
            protein_id, pool.dataset[pool_index[protein_id]], records[protein_id]
        )
        status[reason] += 1
        per_protein.append({"protein_id": protein_id, "status": reason, **diagnostics})
        if group is not None:
            selected_groups.append(group)
            signed = [item["near_signed_sequence_separation"] for item in group["triples"]]
            if all(value < 0 for value in signed):
                direction_balance["all_before"] += 1
            elif all(value > 0 for value in signed):
                direction_balance["all_after"] += 1
            else:
                direction_balance["mixed"] += 1
        print(f"[{ordinal:03d}/{len(selected_ids)}] {protein_id}: {reason}", flush=True)

    report = {
        "purpose": "revised_group_eligibility_only_no_encoder_forwards_or_interventions",
        "config": {
            "seed": SEED,
            "selected_probe_test_proteins": len(selected_ids),
            "group_size": GROUP_SIZE,
            "near_distance_nm_max": NEAR_NM,
            "control_and_donor_distance_nm_min": FAR_NM,
            "minimum_sequence_separation_for_near_control_and_donor": MIN_SEQUENCE_SEPARATION,
            "near_control_sequence_tolerance": None,
            "near_control_sequence_matching": "exact_encoder_signed_bin",
            "sequence_separation_feature_dimension": SEQ_SEP_DIM,
            "donor_sequence_tolerance": None,
            "same_aa_within_each_near_control_donor_triple": True,
            "unique_near_control_and_donor_rows": True,
            "maximum_group_magnitude_relative_mismatch": MAX_MAGNITUDE_MISMATCH,
            "maximum_pair_magnitude_relative_mismatch": MAX_PAIR_MAGNITUDE_MISMATCH,
            "minimum_relative_intervention_per_row": MIN_RELATIVE_INTERVENTION,
            "maximum_single_squared_magnitude_fraction": MAX_SINGLE_ENERGY_FRACTION,
            "same_chain_segment_and_observed_ca": True,
            "encoder_forward_run": False,
        },
        "input_manifest": str(input_dir / "manifest.json"),
        "checkpoint_provenance": manifest["provenance"],
        "counts": {
            "selected": len(selected_ids),
            "eligible": len(selected_groups),
            "excluded": len(selected_ids) - len(selected_groups),
            "status": dict(sorted(status.items())),
            "eligible_direction_balance": dict(sorted(direction_balance.items())),
        },
        "selected_groups": selected_groups,
        "per_protein_funnel": per_protein,
    }
    output = args.output_dir.resolve() / args.output_name
    _atomic_json(report, output)
    print(json.dumps({"counts": report["counts"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
