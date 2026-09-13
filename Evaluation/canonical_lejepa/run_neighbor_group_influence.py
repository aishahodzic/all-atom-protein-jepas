"""Run the revised matched three-residue neighbour intervention."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from Evaluation.canonical_lejepa.extract_full_context_layer_states import (
    EXPECTED_CHECKPOINT_SHA256,
    load_verified_encoder,
    preprocess_graphs,
    sha256_file,
)
from Evaluation.canonical_lejepa.check_neighbor_group_revised_eligibility import (
    _sequence_separation_bin,
)
from Evaluation.canonical_lejepa.run_neighbor_influence import (
    FORWARD_ATOL,
    FORWARD_RTOL,
    N_BOOTSTRAP,
    OUTPUT_DIR,
    SEED,
    STATE_DIR,
    _assert_clean_cache_parity,
    _assert_snapshot_equal,
    _atomic_torch_save,
    _atomic_write_text,
    _response_metrics,
    _snapshot_tensors,
)
from Evaluation.linear_probing.data_pool import build_protein_pool


REPO = Path(__file__).resolve().parents[2]
ELIGIBILITY = REPO / (
    "Evaluation/canonical_lejepa/results/"
    "neighbor_group_signedbin_pair50_eligibility_probe_test1045_seed42.json"
)
OUTPUT_STEM = "neighbor_group3_signedbin_influence_probe_test564_seed42"
EXPECTED_ELIGIBLE = 564
GROUP_SIZE = 3


def _load_inputs(
    eligibility_path: Path, state_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    with eligibility_path.open(encoding="utf-8") as handle:
        eligibility = json.load(handle)
    groups = eligibility["selected_groups"]
    if eligibility["counts"]["eligible"] != EXPECTED_ELIGIBLE or len(groups) != EXPECTED_ELIGIBLE:
        raise AssertionError("unexpected group eligibility count")
    config = eligibility["config"]
    expected = {
        "group_size": GROUP_SIZE,
        "maximum_pair_magnitude_relative_mismatch": 0.5,
        "maximum_group_magnitude_relative_mismatch": 0.1,
        "maximum_single_squared_magnitude_fraction": 0.5,
        "near_control_sequence_matching": "exact_encoder_signed_bin",
        "encoder_forward_run": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AssertionError(f"eligibility config mismatch for {key}")
    with (state_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["provenance"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("state-cache checkpoint mismatch")
    wanted = {group["protein_id"] for group in groups}
    records: dict[str, dict[str, Any]] = {}
    for shard_meta in manifest["shards"]:
        shard = torch.load(
            state_dir / shard_meta["path"], map_location="cpu", weights_only=False
        )
        for record in shard:
            if record["protein_id"] in wanted:
                records[record["protein_id"]] = record
        del shard
    if set(records) != wanted:
        raise AssertionError("missing group state records")
    return eligibility, records, manifest


class GroupInterventionRunner:
    def __init__(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder
        self.mode = "clean"
        self.recipients: list[int] = []
        self.donor_states: torch.Tensor | None = None
        self.values: dict[str, list[torch.Tensor]] = {}
        self.context_values: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.capture_context = False
        self.handles = [
            encoder.transformer_layers[0].register_forward_hook(self._block1_hook),
            encoder.transformer_layers[11].register_forward_hook(self._block12_hook),
            encoder.transformer_layers[1].register_forward_pre_hook(
                self._block2_pre_hook, with_kwargs=True
            ),
        ]

    def _block1_hook(self, module, args, output):
        del module, args
        self.values.setdefault("block_1_pre", []).append(output.detach().float().cpu())
        post = output
        if self.mode != "clean":
            if len(self.recipients) != GROUP_SIZE or self.donor_states is None:
                raise AssertionError("group intervention is incomplete")
            post = output.clone()
            post[0, self.recipients] = self.donor_states.to(post)
        self.values.setdefault("block_1_post", []).append(post.detach().float().cpu())
        return post

    def _block12_hook(self, module, args, output):
        del module, args
        self.values.setdefault("block_12", []).append(output.detach().float().cpu())

    def _block2_pre_hook(self, module, args, kwargs):
        del module, kwargs
        if self.capture_context:
            self.context_values.append(
                tuple(value.detach().float().cpu() for value in args[1:4])
            )

    def run(
        self,
        batch: Any,
        mode: str,
        recipients: list[int] | None = None,
        donor_states: torch.Tensor | None = None,
        capture_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if mode not in {"clean", "near", "far"}:
            raise ValueError(f"invalid mode: {mode}")
        self.mode = mode
        self.recipients = recipients or []
        self.donor_states = donor_states
        self.capture_context = capture_context
        self.values.clear()
        self.context_values.clear()
        with torch.inference_mode():
            output = self.encoder(batch)
        counts = {
            key: len(self.values.get(key, []))
            for key in ("block_1_pre", "block_1_post", "block_12")
        }
        if any(value != 1 for value in counts.values()):
            raise AssertionError(f"hook counts: {counts}")
        if capture_context and len(self.context_values) != 1:
            raise AssertionError("layer-2 context hook did not fire exactly once")
        return {key: self.values[key][0] for key in counts} | {
            "latent_mean": output["mean"].detach().float().cpu()
        }

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def _indices(group: dict[str, Any]) -> tuple[int, list[int], list[int], list[int]]:
    triples = group["triples"]
    if len(triples) != GROUP_SIZE:
        raise AssertionError("group does not contain three triples")
    focal = int(group["focal_i"])
    near = [int(item["near"]) for item in triples]
    far = [int(item["control"]) for item in triples]
    donors = [int(item["donor"]) for item in triples]
    if len({focal, *near, *far, *donors}) != 1 + 3 * GROUP_SIZE:
        raise AssertionError("group rows are not all distinct")
    for item in triples:
        if _sequence_separation_bin(item["near_signed_sequence_separation"]) != (
            _sequence_separation_bin(item["control_signed_sequence_separation"])
        ):
            raise AssertionError("near/control signed sequence bins differ")
    return focal, near, far, donors


def _direction_group(group: dict[str, Any]) -> str:
    signed = [item["near_signed_sequence_separation"] for item in group["triples"]]
    if all(value < 0 for value in signed):
        return "all_before"
    if all(value > 0 for value in signed):
        return "all_after"
    return "mixed"


def _run_triplet(
    runner: GroupInterventionRunner,
    batch: Any,
    group: dict[str, Any],
    record: dict[str, Any],
    capture_context: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    focal, near, far, donors = _indices(group)
    clean = runner.run(batch, "clean", capture_context=capture_context)
    clean_context = copy.deepcopy(runner.context_values)
    _assert_clean_cache_parity(group["protein_id"], clean, record)
    donor_states = clean["block_1_pre"][0, donors].clone()
    near_run = runner.run(
        batch, "near", near, donor_states, capture_context=capture_context
    )
    near_context = copy.deepcopy(runner.context_values)
    far_run = runner.run(
        batch, "far", far, donor_states, capture_context=capture_context
    )
    far_context = copy.deepcopy(runner.context_values)
    torch.testing.assert_close(near_run["block_1_pre"], clean["block_1_pre"], rtol=0, atol=0)
    torch.testing.assert_close(far_run["block_1_pre"], clean["block_1_pre"], rtol=0, atol=0)
    expected_near = clean["block_1_pre"].clone()
    expected_far = clean["block_1_pre"].clone()
    expected_near[0, near] = donor_states
    expected_far[0, far] = donor_states
    torch.testing.assert_close(near_run["block_1_post"], expected_near, rtol=0, atol=0)
    torch.testing.assert_close(far_run["block_1_post"], expected_far, rtol=0, atol=0)
    torch.testing.assert_close(near_run["block_1_post"][0, focal], clean["block_1_post"][0, focal], rtol=0, atol=0)
    torch.testing.assert_close(far_run["block_1_post"][0, focal], clean["block_1_post"][0, focal], rtol=0, atol=0)
    if capture_context:
        for changed in (near_context, far_context):
            for actual, expected in zip(changed[0], clean_context[0]):
                torch.testing.assert_close(actual, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    return clean, near_run, far_run


def run_tests(
    encoder: torch.nn.Module,
    pool: Any,
    groups: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, str]:
    group = min(groups, key=lambda item: len(records[item["protein_id"]]["block_1"]))
    index = {protein_id: i for i, protein_id in enumerate(pool.protein_ids)}
    batch = preprocess_graphs([pool.dataset[index[group["protein_id"]]]]).to(device)
    snapshot = _snapshot_tensors(dict(batch.items()))
    runner = GroupInterventionRunner(encoder)
    try:
        clean, near_run, _ = _run_triplet(
            runner, batch, group, records[group["protein_id"]], capture_context=True
        )
        _, near, _, donors = _indices(group)
        repeated = runner.run(
            batch, "near", near, clean["block_1_pre"][0, donors]
        )
        for key in ("block_12", "latent_mean"):
            torch.testing.assert_close(repeated[key], near_run[key], rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
        _assert_snapshot_equal(dict(batch.items()), snapshot)
    finally:
        runner.remove()
    return {
        "ema_and_checkpoint_provenance": "passed",
        "clean_forward_cache_parity": "passed",
        "exact_three_row_replacement": "passed",
        "shared_donors_and_all_other_rows_unchanged": "passed",
        "focal_block1_untouched": "passed",
        "input_pair_conditioning_and_masks_unchanged": "passed",
        "deterministic_forward_with_tolerance": "passed",
        "test_protein": group["protein_id"],
    }


def _bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    result = {}
    for space in ("block_12", "latent_mean"):
        selected = [row for row in rows if row["space"] == space]
        near = np.asarray([row["near_transmission_gain"] for row in selected])
        far = np.asarray([row["far_transmission_gain"] for row in selected])
        difference = near - far
        means = difference[
            rng.integers(0, len(difference), size=(N_BOOTSTRAP, len(difference)))
        ].mean(axis=1)
        result[space] = {
            "n_proteins": len(selected),
            "near_gain_mean": float(near.mean()),
            "near_gain_median": float(np.median(near)),
            "far_gain_mean": float(far.mean()),
            "far_gain_median": float(np.median(far)),
            "paired_mean_difference_near_minus_far": float(difference.mean()),
            "paired_mean_difference_bootstrap_95_ci": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
            "fraction_near_gain_greater_than_far": float(np.mean(near > far)),
        }
    return result


def _stratified_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        direction: _bootstrap(
            [row for row in rows if row["direction_group"] == direction]
        )
        for direction in ("all_before", "all_after", "mixed")
    }


def _plot(rows: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    rng = np.random.default_rng(SEED)
    colors = ("#D55E00", "#0072B2")
    for ax, (space, title) in zip(
        axes,
        (("block_12", "Focal Block 12 (768D)"), ("latent_mean", "Focal latent mean (8D)")),
    ):
        selected = [row for row in rows if row["space"] == space]
        near = np.asarray([row["near_transmission_gain"] for row in selected])
        far = np.asarray([row["far_transmission_gain"] for row in selected])
        boxes = ax.boxplot(
            [near, far],
            positions=[0, 1],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
        ax.scatter(
            rng.normal(0, 0.035, len(near)), near, color=colors[0], s=9, alpha=0.22
        )
        ax.scatter(
            rng.normal(1, 0.035, len(far)), far, color=colors[1], s=9, alpha=0.22
        )
        ax.set_xticks([0, 1], ["Near group\n(3 residues)", "Far group\n(3 residues)"])
        ax.set_yscale("log")
        ax.set_ylabel("Transmission gain (log scale)")
        ax.set_title(f"{title} (n={len(selected)})")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Canonical LeJEPA — three-residue neighbour influence, 100k EMA",
        y=0.96,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.80, wspace=0.27)
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def run_experiment(
    encoder: torch.nn.Module,
    pool: Any,
    groups: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {protein_id: i for i, protein_id in enumerate(pool.protein_ids)}
    runner = GroupInterventionRunner(encoder)
    rows = []
    tensors = []
    try:
        for ordinal, group in enumerate(groups, start=1):
            protein_id = group["protein_id"]
            record = records[protein_id]
            if record["probe_split"] != "probe_test":
                raise AssertionError("selected non-test protein")
            batch = preprocess_graphs([pool.dataset[index[protein_id]]]).to(device)
            snapshot = _snapshot_tensors(dict(batch.items()))
            clean, near_run, far_run = _run_triplet(runner, batch, group, record)
            _assert_snapshot_equal(dict(batch.items()), snapshot)
            focal, near, far, donors = _indices(group)
            direction_group = _direction_group(group)
            clean_h1 = clean["block_1_pre"][0]
            near_l2s = torch.linalg.vector_norm(clean_h1[donors] - clean_h1[near], dim=-1)
            far_l2s = torch.linalg.vector_norm(clean_h1[donors] - clean_h1[far], dim=-1)
            near_group_l2 = float(torch.linalg.vector_norm(near_l2s).item())
            far_group_l2 = float(torch.linalg.vector_norm(far_l2s).item())
            np.testing.assert_allclose(near_group_l2, group["near_group_l2"], rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
            np.testing.assert_allclose(far_group_l2, group["far_group_l2"], rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
            tensor_record = {
                "protein_id": protein_id,
                "focal_i": focal,
                "near_recipients": near,
                "far_recipients": far,
                "donors": donors,
                "near_input_delta_l2_per_row": near_l2s.clone(),
                "far_input_delta_l2_per_row": far_l2s.clone(),
            }
            for space in ("block_12", "latent_mean"):
                clean_focal = clean[space][0, focal].clone()
                near_focal = near_run[space][0, focal].clone()
                far_focal = far_run[space][0, focal].clone()
                near_metrics = _response_metrics(clean_focal, near_focal, near_group_l2)
                far_metrics = _response_metrics(clean_focal, far_focal, far_group_l2)
                row = {
                    "protein_id": protein_id,
                    "space": space,
                    "dimension": int(clean_focal.numel()),
                    "focal_i": focal,
                    "direction_group": direction_group,
                    "near_recipients": json.dumps(near),
                    "far_recipients": json.dumps(far),
                    "donors": json.dumps(donors),
                    "near_input_group_l2": near_group_l2,
                    "far_input_group_l2": far_group_l2,
                    "input_group_magnitude_mismatch": group["group_magnitude_relative_mismatch"],
                }
                row.update({f"near_{key}": value for key, value in near_metrics.items()})
                row.update({f"far_{key}": value for key, value in far_metrics.items()})
                if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
                    raise AssertionError(f"non-finite result: {protein_id} {space}")
                rows.append(row)
                tensor_record[space] = {
                    "clean_focal": clean_focal,
                    "near_focal": near_focal,
                    "far_focal": far_focal,
                }
            tensors.append(tensor_record)
            print(f"[{ordinal:02d}/{len(groups)}] {protein_id}", flush=True)
    finally:
        runner.remove()
    return rows, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", default="tests,run")
    parser.add_argument("--eligibility", type=Path, default=ELIGIBILITY)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    args = parser.parse_args()
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    if set(stages) - {"tests", "run"}:
        raise ValueError("stages must be tests and/or run")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    eligibility, records, state_manifest = _load_inputs(
        args.eligibility.resolve(), args.state_dir.resolve()
    )
    groups = eligibility["selected_groups"]
    encoder, provenance = load_verified_encoder(args.device)
    pool = build_protein_pool(seed=SEED)
    tests = run_tests(encoder, pool, groups, records, args.device) if "tests" in stages else {}
    print(json.dumps({"tests": tests}, indent=2), flush=True)
    if "run" not in stages:
        return
    output_dir = args.output_dir.resolve()
    paths = {suffix: output_dir / f"{args.output_stem}.{suffix}" for suffix in ("csv", "json", "pt", "png", "pdf")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite group-intervention output")
    rows, tensors = run_experiment(encoder, pool, groups, records, args.device)
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        _atomic_write_text(handle.read(), paths["csv"])
    _atomic_torch_save(tensors, paths["pt"])
    report = {
        "purpose": "matched_three_residue_near_vs_far_block1_hidden_state_replacement",
        "interpretation_limit": "Hidden-state replacement sensitivity, not mutation or proof of a biological pathway.",
        "provenance": provenance,
        "eligibility_path": str(args.eligibility.resolve()),
        "eligibility_sha256": sha256_file(args.eligibility.resolve()),
        "state_manifest_provenance": state_manifest["provenance"],
        "counts": {"proteins": len(groups), "result_rows": len(rows)},
        "tests": tests,
        "summary": _bootstrap(rows),
        "summary_by_direction": _stratified_bootstrap(rows),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    _atomic_write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", paths["json"])
    _plot(rows, output_dir / args.output_stem)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
