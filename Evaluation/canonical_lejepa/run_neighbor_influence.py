"""Run the matched near-versus-far Block-1 hidden-state intervention."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from Evaluation.canonical_lejepa.extract_full_context_layer_states import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    VALIDITY_EXPRESSION,
    load_verified_encoder,
    preprocess_graphs,
    sha256_file,
)
from Evaluation.linear_probing.data_pool import build_protein_pool


REPO = Path(__file__).resolve().parents[2]
ELIGIBILITY = REPO / (
    "Evaluation/canonical_lejepa/results/"
    "neighbor_influence_eligibility_probe_test100_seed42.json"
)
STATE_DIR = REPO / (
    "store/canonical_lejepa_v1_100k/evaluation/"
    "full_context_layer_states/full"
)
OUTPUT_DIR = REPO / "Evaluation/canonical_lejepa/results"
OUTPUT_STEM = "neighbor_influence_probe_test46_seed42"
SEED = 42
EXPECTED_ELIGIBLE = 46
FORWARD_RTOL = 1e-5
FORWARD_ATOL = 1e-6
N_BOOTSTRAP = 10_000
NORM_EPS = 1e-8


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


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_inputs(
    eligibility_path: Path, state_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    with eligibility_path.open(encoding="utf-8") as handle:
        eligibility = json.load(handle)
    quartets = eligibility["selected_quartets"]
    if len(quartets) != EXPECTED_ELIGIBLE:
        raise AssertionError(f"expected {EXPECTED_ELIGIBLE} quartets, got {len(quartets)}")
    if eligibility["config"]["seed"] != SEED:
        raise AssertionError("unexpected eligibility seed")
    with (state_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["provenance"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("state-cache checkpoint mismatch")
    wanted = {item["protein_id"] for item in quartets}
    records: dict[str, dict[str, Any]] = {}
    for shard_meta in manifest["shards"]:
        shard = torch.load(
            state_dir / shard_meta["path"], map_location="cpu", weights_only=False
        )
        for record in shard:
            if record["protein_id"] in wanted:
                if record["protein_id"] in records:
                    raise AssertionError(f"duplicate cached protein: {record['protein_id']}")
                records[record["protein_id"]] = record
        del shard
    if set(records) != wanted:
        raise AssertionError(f"missing state records: {sorted(wanted - set(records))}")
    return eligibility, records, manifest


def _snapshot_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_tensors(item) for item in value)
    return None


def _assert_snapshot_equal(value: Any, snapshot: Any, path: str = "batch") -> None:
    if isinstance(value, torch.Tensor):
        if not torch.equal(value, snapshot):
            raise AssertionError(f"input tensor was modified: {path}")
    elif isinstance(value, dict):
        if set(value) != set(snapshot):
            raise AssertionError(f"input mapping changed: {path}")
        for key, item in value.items():
            _assert_snapshot_equal(item, snapshot[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        if len(value) != len(snapshot):
            raise AssertionError(f"input sequence changed: {path}")
        for index, item in enumerate(value):
            _assert_snapshot_equal(item, snapshot[index], f"{path}[{index}]")


class InterventionRunner:
    """Capture Block 1/12 and optionally replace one Block-1 residue row."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder
        self.mode = "clean"
        self.recipient: int | None = None
        self.donor_state: torch.Tensor | None = None
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
        if not isinstance(output, torch.Tensor):
            raise TypeError("Block-1 hook expected a tensor")
        self.values.setdefault("block_1_pre", []).append(output.detach().float().cpu())
        if self.mode == "clean":
            post = output
        else:
            if self.recipient is None or self.donor_state is None:
                raise AssertionError("intervention configuration is incomplete")
            post = output.clone()
            post[0, self.recipient] = self.donor_state.to(
                device=post.device, dtype=post.dtype
            )
        self.values.setdefault("block_1_post", []).append(post.detach().float().cpu())
        return post

    def _block12_hook(self, module, args, output):
        del module, args
        self.values.setdefault("block_12", []).append(output.detach().float().cpu())

    def _block2_pre_hook(self, module, args, kwargs):
        del module, kwargs
        if self.capture_context:
            # pair representation, conditioning, and mask entering layer 2
            self.context_values.append(
                tuple(value.detach().float().cpu() for value in args[1:4])
            )

    def run(
        self,
        batch: Any,
        *,
        mode: str,
        recipient: int | None = None,
        donor_state: torch.Tensor | None = None,
        capture_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if mode not in {"clean", "near", "far"}:
            raise ValueError(f"invalid mode: {mode}")
        self.mode = mode
        self.recipient = recipient
        self.donor_state = donor_state
        self.capture_context = capture_context
        self.values.clear()
        self.context_values.clear()
        with torch.inference_mode():
            output = self.encoder(batch)
        counts = {key: len(self.values.get(key, [])) for key in (
            "block_1_pre", "block_1_post", "block_12"
        )}
        if any(count != 1 for count in counts.values()):
            raise AssertionError(f"hooks must each fire once: {counts}")
        if capture_context and len(self.context_values) != 1:
            raise AssertionError(f"layer-2 context hook count: {len(self.context_values)}")
        return {
            key: self.values[key][0] for key in counts
        } | {"latent_mean": output["mean"].detach().float().cpu()}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _response_metrics(
    clean: torch.Tensor, changed: torch.Tensor, input_delta_l2: float
) -> dict[str, float]:
    clean = clean.float()
    changed = changed.float()
    delta = float(torch.linalg.vector_norm(changed - clean).item())
    clean_norm = float(torch.linalg.vector_norm(clean).item())
    changed_norm = float(torch.linalg.vector_norm(changed).item())
    if input_delta_l2 <= NORM_EPS:
        raise ValueError("input intervention norm is too small")
    if clean_norm <= NORM_EPS or changed_norm <= NORM_EPS:
        cosine_change = float("nan")
    else:
        cosine = float(torch.dot(clean, changed).item() / (clean_norm * changed_norm))
        cosine_change = 1.0 - max(-1.0, min(1.0, cosine))
    return {
        "response_l2": delta,
        "transmission_gain": delta / input_delta_l2,
        "relative_response": delta / max(clean_norm, NORM_EPS),
        "cosine_change": cosine_change,
        "clean_norm": clean_norm,
        "changed_norm": changed_norm,
    }


def _validate_quartet(item: dict[str, Any], n: int) -> None:
    indices = [
        int(item[key]) for key in ("focal_i", "near_j", "far_control_k", "shared_donor_d")
    ]
    if len(set(indices)) != 4 or min(indices) < 0 or max(indices) >= n:
        raise AssertionError(f"invalid quartet indices for {item['protein_id']}: {indices}")
    if (
        item["near_distance_nm"] > 0.8
        or item["far_control_distance_nm"] < 1.5
        or item["donor_distance_nm"] < 1.5
    ):
        raise AssertionError("distance eligibility was not preserved")
    if min(
        item["near_sequence_separation"],
        item["far_control_sequence_separation"],
        item["donor_sequence_separation"],
    ) <= 20:
        raise AssertionError("sequence-distance eligibility was not preserved")
    if (
        abs(item["near_sequence_separation"] - item["far_control_sequence_separation"]) > 5
        or abs(item["near_sequence_separation"] - item["donor_sequence_separation"]) > 5
    ):
        raise AssertionError("recipient/donor sequence matching was not preserved")


def _assert_clean_cache_parity(
    protein_id: str, clean: dict[str, torch.Tensor], record: dict[str, Any]
) -> None:
    for capture_key, record_key in (
        ("block_1_pre", "block_1"),
        ("block_12", "block_12"),
        ("latent_mean", "latent_mean"),
    ):
        actual = clean[capture_key][0, : len(record[record_key])]
        torch.testing.assert_close(
            actual,
            record[record_key],
            rtol=FORWARD_RTOL,
            atol=FORWARD_ATOL,
            msg=lambda msg: f"clean/cache parity failed for {protein_id} {record_key}: {msg}",
        )


def _run_triplet(
    runner: InterventionRunner,
    device_batch: Any,
    item: dict[str, Any],
    record: dict[str, Any],
    *,
    capture_context: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    focal = int(item["focal_i"])
    near = int(item["near_j"])
    far = int(item["far_control_k"])
    donor = int(item["shared_donor_d"])
    _validate_quartet(item, len(record["block_1"]))
    aa = record["aa_integer"]
    if not (
        int(aa[near]) == int(aa[far]) == int(aa[donor]) == int(item["aa_integer"])
    ):
        raise AssertionError("shared-donor AA matching was not preserved")
    if item["intervention_magnitude_relative_mismatch"] > 0.10:
        raise AssertionError("intervention-magnitude matching was not preserved")
    clean = runner.run(device_batch, mode="clean", capture_context=capture_context)
    clean_context = copy.deepcopy(runner.context_values)
    _assert_clean_cache_parity(item["protein_id"], clean, record)
    donor_state = clean["block_1_pre"][0, donor].clone()
    near_run = runner.run(
        device_batch,
        mode="near",
        recipient=near,
        donor_state=donor_state,
        capture_context=capture_context,
    )
    near_context = copy.deepcopy(runner.context_values)
    far_run = runner.run(
        device_batch,
        mode="far",
        recipient=far,
        donor_state=donor_state,
        capture_context=capture_context,
    )
    far_context = copy.deepcopy(runner.context_values)

    torch.testing.assert_close(near_run["block_1_pre"], clean["block_1_pre"], rtol=0, atol=0)
    torch.testing.assert_close(far_run["block_1_pre"], clean["block_1_pre"], rtol=0, atol=0)
    expected_near = clean["block_1_pre"].clone()
    expected_near[0, near] = donor_state
    expected_far = clean["block_1_pre"].clone()
    expected_far[0, far] = donor_state
    torch.testing.assert_close(near_run["block_1_post"], expected_near, rtol=0, atol=0)
    torch.testing.assert_close(far_run["block_1_post"], expected_far, rtol=0, atol=0)
    torch.testing.assert_close(
        near_run["block_1_post"][0, focal], clean["block_1_post"][0, focal], rtol=0, atol=0
    )
    torch.testing.assert_close(
        far_run["block_1_post"][0, focal], clean["block_1_post"][0, focal], rtol=0, atol=0
    )
    if capture_context:
        for changed_context in (near_context, far_context):
            for actual, expected in zip(changed_context[0], clean_context[0]):
                torch.testing.assert_close(actual, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    return clean, near_run, far_run


def run_correctness_tests(
    encoder: torch.nn.Module,
    pool: Any,
    quartets: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, str]:
    item = min(quartets, key=lambda value: len(records[value["protein_id"]]["block_1"]))
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    batch = preprocess_graphs([pool.dataset[pool_index[item["protein_id"]]]]).to(device)
    snapshot = _snapshot_tensors(dict(batch.items()))
    runner = InterventionRunner(encoder)
    try:
        clean, near_run, _ = _run_triplet(
            runner, batch, item, records[item["protein_id"]], capture_context=True
        )
        repeated = runner.run(
            batch,
            mode="near",
            recipient=int(item["near_j"]),
            donor_state=clean["block_1_pre"][0, int(item["shared_donor_d"])],
        )
        for key in ("block_12", "latent_mean"):
            torch.testing.assert_close(
                repeated[key], near_run[key], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
            )
        _assert_snapshot_equal(dict(batch.items()), snapshot)
    finally:
        runner.remove()
    return {
        "ema_and_checkpoint_provenance": "passed",
        "clean_forward_cache_parity": "passed",
        "hook_count_and_capture_isolation": "passed",
        "one_row_replacement_and_shared_donor": "passed",
        "focal_block1_untouched": "passed",
        "input_tensors_unchanged_exactly": "passed",
        "pair_conditioning_and_masks_unchanged": "passed",
        "deterministic_forward_with_tolerance": "passed",
        "test_protein": item["protein_id"],
    }


def _bootstrap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    report: dict[str, Any] = {}
    for space in ("block_12", "latent_mean"):
        subset = [row for row in rows if row["space"] == space]
        near = np.asarray([row["near_transmission_gain"] for row in subset])
        far = np.asarray([row["far_transmission_gain"] for row in subset])
        difference = near - far
        sampled = difference[rng.integers(0, len(difference), size=(N_BOOTSTRAP, len(difference)))]
        bootstrap_means = sampled.mean(axis=1)
        report[space] = {
            "n_proteins": len(subset),
            "near_gain_mean": float(near.mean()),
            "near_gain_median": float(np.median(near)),
            "far_gain_mean": float(far.mean()),
            "far_gain_median": float(np.median(far)),
            "paired_mean_difference_near_minus_far": float(difference.mean()),
            "paired_mean_difference_bootstrap_95_ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "fraction_near_gain_greater_than_far": float(np.mean(near > far)),
        }
    return report


def _make_figure(rows: list[dict[str, Any]], path_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = {"near": "#D55E00", "far": "#0072B2"}
    for column, (space, title) in enumerate(
        (("block_12", "Focal Block 12 (768D)"), ("latent_mean", "Focal latent mean (8D)"))
    ):
        subset = [row for row in rows if row["space"] == space]
        near = np.asarray([row["near_transmission_gain"] for row in subset])
        far = np.asarray([row["far_transmission_gain"] for row in subset])
        ax = axes[0, column]
        for near_value, far_value in zip(near, far):
            ax.plot([0, 1], [near_value, far_value], color="0.75", linewidth=0.7, alpha=0.7)
        ax.scatter(np.zeros_like(near), near, color=colors["near"], s=22, label="Near recipient")
        ax.scatter(np.ones_like(far), far, color=colors["far"], s=22, label="Far recipient")
        ax.set_xticks([0, 1], ["Near (≤8 Å)", "Far (≥15 Å)"])
        ax.set_yscale("log")
        ax.set_ylabel("Transmission gain (log scale)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

        ax = axes[1, column]
        near_distance = np.asarray([row["near_distance_nm"] * 10 for row in subset])
        far_distance = np.asarray([row["far_control_distance_nm"] * 10 for row in subset])
        ax.scatter(near_distance, near, color=colors["near"], s=24, alpha=0.8, label="Near")
        ax.scatter(far_distance, far, color=colors["far"], s=24, alpha=0.8, label="Far")
        ax.set_yscale("log")
        ax.set_xlabel("Focal–recipient Cα distance (Å)")
        ax.set_ylabel("Transmission gain (log scale)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Canonical LeJEPA — Block-1 neighbour influence, 100k EMA")
    for suffix in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def run_experiment(
    encoder: torch.nn.Module,
    pool: Any,
    quartets: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    rows: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    runner = InterventionRunner(encoder)
    try:
        for ordinal, item in enumerate(quartets, start=1):
            protein_id = item["protein_id"]
            if protein_id not in pool_index:
                raise AssertionError(f"missing pool protein: {protein_id}")
            record = records[protein_id]
            if record["probe_split"] != "probe_test":
                raise AssertionError(f"non-test protein selected: {protein_id}")
            batch = preprocess_graphs([pool.dataset[pool_index[protein_id]]]).to(device)
            snapshot = _snapshot_tensors(dict(batch.items()))
            clean, near_run, far_run = _run_triplet(runner, batch, item, record)
            _assert_snapshot_equal(dict(batch.items()), snapshot)
            focal = int(item["focal_i"])
            near = int(item["near_j"])
            far = int(item["far_control_k"])
            donor = int(item["shared_donor_d"])
            near_input_l2 = float(torch.linalg.vector_norm(
                clean["block_1_pre"][0, donor] - clean["block_1_pre"][0, near]
            ).item())
            far_input_l2 = float(torch.linalg.vector_norm(
                clean["block_1_pre"][0, donor] - clean["block_1_pre"][0, far]
            ).item())
            np.testing.assert_allclose(
                near_input_l2, item["near_intervention_l2"], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
            )
            np.testing.assert_allclose(
                far_input_l2, item["far_intervention_l2"], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
            )
            tensor_record: dict[str, Any] = {
                "protein_id": protein_id,
                "indices": {"focal_i": focal, "near_j": near, "far_control_k": far, "shared_donor_d": donor},
                "block_1": {
                    "donor_clean": clean["block_1_pre"][0, donor].clone(),
                    "near_recipient_clean": clean["block_1_pre"][0, near].clone(),
                    "far_recipient_clean": clean["block_1_pre"][0, far].clone(),
                },
            }
            for space in ("block_12", "latent_mean"):
                clean_focal = clean[space][0, focal].clone()
                near_focal = near_run[space][0, focal].clone()
                far_focal = far_run[space][0, focal].clone()
                near_metrics = _response_metrics(clean_focal, near_focal, near_input_l2)
                far_metrics = _response_metrics(clean_focal, far_focal, far_input_l2)
                row = {
                    "protein_id": protein_id,
                    "space": space,
                    "dimension": int(clean_focal.numel()),
                    "focal_i": focal,
                    "near_j": near,
                    "far_control_k": far,
                    "shared_donor_d": donor,
                    "aa_integer": item["aa_integer"],
                    "aa_label": item["aa_label"],
                    "near_distance_nm": item["near_distance_nm"],
                    "far_control_distance_nm": item["far_control_distance_nm"],
                    "donor_distance_nm": item["donor_distance_nm"],
                    "near_sequence_separation": item["near_sequence_separation"],
                    "far_control_sequence_separation": item["far_control_sequence_separation"],
                    "donor_sequence_separation": item["donor_sequence_separation"],
                    "near_input_delta_l2": near_input_l2,
                    "far_input_delta_l2": far_input_l2,
                }
                row.update({f"near_{key}": value for key, value in near_metrics.items()})
                row.update({f"far_{key}": value for key, value in far_metrics.items()})
                if not all(np.isfinite(value) for key, value in row.items() if isinstance(value, float)):
                    raise AssertionError(f"non-finite metric for {protein_id} {space}")
                rows.append(row)
                tensor_record[space] = {
                    "clean_focal": clean_focal,
                    "near_focal": near_focal,
                    "far_focal": far_focal,
                }
            tensors.append(tensor_record)
            print(f"[{ordinal:02d}/{len(quartets)}] {protein_id}", flush=True)
    finally:
        runner.remove()
    if len(rows) != 2 * len(quartets):
        raise AssertionError("incomplete result rows")
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
        raise ValueError(f"invalid stages: {stages}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    eligibility_path = args.eligibility.resolve()
    state_dir = args.state_dir.resolve()
    output_dir = args.output_dir.resolve()
    eligibility, records, state_manifest = _load_inputs(eligibility_path, state_dir)
    quartets = eligibility["selected_quartets"]
    encoder, provenance = load_verified_encoder(args.device)
    pool = build_protein_pool(seed=SEED)
    tests: dict[str, str] = {}
    if "tests" in stages:
        tests = run_correctness_tests(encoder, pool, quartets, records, args.device)
        print(json.dumps({"tests": tests}, indent=2), flush=True)
    if "run" not in stages:
        return

    paths = {
        suffix: output_dir / f"{args.output_stem}.{suffix}"
        for suffix in ("csv", "json", "pt", "png", "pdf")
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    rows, tensors = run_experiment(encoder, pool, quartets, records, args.device)
    fieldnames = list(rows[0])
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        _atomic_write_text(handle.read(), paths["csv"])
    _atomic_torch_save(tensors, paths["pt"])
    summary = {
        "purpose": "matched_near_vs_far_block1_hidden_state_replacement_sensitivity",
        "interpretation_limit": (
            "This tests sensitivity to a contextual hidden-state replacement after Block 1; "
            "it is not an amino-acid mutation or evidence of a biological communication pathway."
        ),
        "config": {
            "seed": SEED,
            "n_proteins": len(quartets),
            "context": "full_unmasked_evaluation",
            "encoder_mode": "eval_inference_mode",
            "batch_size": 1,
            "rotation": None,
            "intervention": "clean shared-donor Block-1 768D state copied into recipient row",
            "input_geometry_pair_features_and_masks": "unchanged",
            "encoder_validity_expression": VALIDITY_EXPRESSION,
            "forward_tolerance": {"rtol": FORWARD_RTOL, "atol": FORWARD_ATOL},
            "bootstrap_resamples": N_BOOTSTRAP,
        },
        "provenance": provenance,
        "input_artifacts": {
            "eligibility_path": str(eligibility_path),
            "eligibility_sha256": sha256_file(eligibility_path),
            "state_manifest_path": str(state_dir / "manifest.json"),
            "state_manifest_sha256": sha256_file(state_dir / "manifest.json"),
            "state_checkpoint_provenance": state_manifest["provenance"],
        },
        "tests": tests,
        "summary": _bootstrap_summary(rows),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    if provenance["checkpoint_path"] != str(CHECKPOINT) or sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("checkpoint identity changed")
    _atomic_write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", paths["json"])
    _make_figure(rows, output_dir / args.output_stem)
    if len(torch.load(paths["pt"], map_location="cpu", weights_only=False)) != len(quartets):
        raise AssertionError("tensor artifact round-trip failed")
    with paths["csv"].open(encoding="utf-8") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != 2 * len(quartets):
            raise AssertionError("CSV round-trip failed")
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
