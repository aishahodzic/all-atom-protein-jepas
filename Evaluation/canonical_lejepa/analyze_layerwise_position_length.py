#!/usr/bin/env python3
"""Measure position and chain-length decodability through canonical LeJEPA.

The diagnostic uses the frozen 5,224-protein evaluation cohort and its fixed
protein-level train/test split.  Residue-position probes use the established
15,000-residue physical sample; chain-length probes use one mean embedding per
protein so long chains do not receive disproportionate weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR = ROOT / "Evaluation" / "linear_probing"
KNN = ROOT / "Evaluation" / "knn_purity"
VIZ = ROOT / "Evaluation" / "latent_viz"
RUNTIME = pathlib.Path(
    os.environ.get(
        "LAYERWISE_POSITION_LENGTH_TMP",
        ROOT / ".runtime-tmp/layerwise-position-length",
    )
)
RUNTIME.mkdir(parents=True, exist_ok=True)
(RUNTIME / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(RUNTIME))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME / "matplotlib"))
for path in (ROOT, LINEAR, KNN, VIZ):
    sys.path.insert(0, str(path))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list

from Evaluation.linear_probing.data_pool import (  # noqa: E402
    build_protein_pool,
    eval_transform_compose,
)
from knn import load_labels  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CONFIG = ROOT / "StructTokenBench/100k-checkpoints/shared/exp_config_jepa_nn130m.json"
OUT_DIR = ROOT / "Evaluation/canonical_lejepa/results/layerwise_position_length"
FIGURE_DIR = ROOT / "iclr2027/figures/per-block-results"
EXPECTED_PROTEINS = 5_224
EXPECTED_POOL_RESIDUES = 748_402
EXPECTED_SAMPLE_SHA256 = "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"
N_SAMPLE = 15_000
ALPHAS = np.logspace(-2, 4, 13)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: pathlib.Path
    checkpoint_sha256: str
    source_sha256: str
    step: int


MODELS = {
    "uncropped": ModelSpec(
        label="Uncropped canonical LeJEPA (raw, 100k)",
        checkpoint=ROOT / (
            "store/canonical_lejepa_v1_100k/evaluation/"
            "canonical-lejepa-v1-step100000-raw-staged.ckpt"
        ),
        checkpoint_sha256="28f93a4224323b7366243265abd7721f2dffaaf6f8c9fd4aa4767788fd5d2bac",
        source_sha256="4f09dff610adaa75c8ac51b4cf01acb001c236c72dd91ac31568ffa191484527",
        step=100_000,
    ),
    "cropped_protein": ModelSpec(
        label="Cropped per-protein LeJEPA (raw, 10k)",
        checkpoint=ROOT / (
            "StructTokenBench/100k-checkpoints/canonical-lejepa/lr-ablation/"
            "canonical_lejepa_protein_middle_lr4e4_raw10k_stb.ckpt"
        ),
        checkpoint_sha256="61efd8bc9e2010792a4c6b76c2576e26d765d97d7c7c2eb0263725c0af77107d",
        source_sha256="c9a7b99204a2ba6324bd4052a3682082354740b8cbef9872c7d8f2bb18c6541c",
        step=10_000,
    ),
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder(spec: ModelSpec, device: str) -> EncoderTransformer:
    actual_sha = sha256_file(spec.checkpoint)
    if actual_sha != spec.checkpoint_sha256:
        raise RuntimeError(f"checkpoint SHA mismatch: {actual_sha}")
    saved = torch.load(spec.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    provenance = saved.get("canonical_lejepa_evaluation", {})
    expected = {
        "encoder_source": "raw",
        "source_checkpoint_sha256": spec.source_sha256,
        "source_global_step": spec.step,
        "latent_dim": 8,
        "projector": None,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"bad staged provenance for {key}: {provenance}")
    state = {
        str(key).removeprefix("online_encoder."): value
        for key, value in saved["state_dict"].items()
        if str(key).startswith("online_encoder.")
    }
    if len(state) != 381:
        raise RuntimeError(f"expected 381 encoder tensors, found {len(state)}")
    with CONFIG.open() as handle:
        nn_ae = json.load(handle)["nn_ae"]
    encoder = EncoderTransformer(**nn_ae)
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval()
    del saved, state
    return encoder.to(device)


class AllBlockCapture:
    def __init__(self, encoder: EncoderTransformer):
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [
            layer.register_forward_hook(self._hook(block))
            for block, layer in enumerate(encoder.transformer_layers, start=1)
        ]

    def _hook(self, block: int):
        def hook(module, args, output):
            del module, args
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"block {block} returned {type(output)}")
            if block in self.values:
                raise RuntimeError(f"block {block} fired more than once")
            self.values[block] = output.detach()
        return hook

    def clear(self):
        self.values.clear()

    def take(self) -> dict[int, torch.Tensor]:
        if set(self.values) != set(range(1, 13)):
            raise RuntimeError(f"missing block captures: {sorted(self.values)}")
        return self.values

    def close(self):
        for handle in self.handles:
            handle.remove()


def prepare_batch(graphs):
    transform = eval_transform_compose()
    transformed = [transform(graph) for graph in graphs]
    batch = dense_padded_from_data_list(transformed)
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0].bool()
    return batch


def frozen_sample(labels: dict):
    protein_ids = sorted(labels)
    selection = CombinedSelection(protein_ids, labels)
    if len(protein_ids) != EXPECTED_PROTEINS or selection.n != EXPECTED_POOL_RESIDUES:
        raise RuntimeError(f"cohort changed: proteins={len(protein_ids)}, residues={selection.n}")
    sampled = sample_indices(selection.n, N_SAMPLE, SEED)
    physical = np.concatenate([
        np.asarray([f"{pid}:{int(index)}" for index in indices], dtype=object)
        for pid, indices in selection.keys
    ])[sampled]
    sample_sha = hashlib.sha256("\n".join(physical.tolist()).encode()).hexdigest()
    if sample_sha != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(f"frozen physical sample changed: {sample_sha}")
    selected: dict[str, np.ndarray] = {}
    for item in physical:
        pid, index = str(item).rsplit(":", 1)
        selected.setdefault(pid, []).append(int(index))
    selected = {pid: np.asarray(indices, dtype=np.int64) for pid, indices in selected.items()}
    return selected, sample_sha


def _append(store: dict[str, list[np.ndarray]], key: str, value: torch.Tensor):
    store[key].append(value.detach().float().cpu().numpy())


def extract_features(spec: ModelSpec, device: str, batch_size: int):
    labels = load_labels()
    selected_rows, sample_sha = frozen_sample(labels)
    pool = build_protein_pool(seed=42)
    if len(pool.protein_ids) != EXPECTED_PROTEINS or set(pool.protein_ids) != set(labels):
        raise RuntimeError("protein pool does not match the frozen label cohort")
    ordered_indices = sorted(range(len(pool.protein_ids)), key=lambda i: pool.protein_ids[i])

    encoder = load_encoder(spec, device)
    capture = AllBlockCapture(encoder)
    keys = [f"block_{block}" for block in range(1, 13)] + ["final_8d"]
    protein_features = {key: [] for key in keys}
    residue_features = {key: [] for key in keys}
    protein_ids_out: list[str] = []
    residue_pid: list[str] = []
    residue_index: list[int] = []
    residue_length: list[int] = []

    try:
        for start in range(0, len(ordered_indices), batch_size):
            indices = ordered_indices[start : start + batch_size]
            pids = [pool.protein_ids[index] for index in indices]
            batch = prepare_batch([pool.dataset[index] for index in indices])
            capture.clear()
            with torch.inference_mode():
                output = encoder(batch.to(device))
            blocks = capture.take()
            final = output["mean"].detach()

            for row, pid in enumerate(pids):
                mask = batch["mask"][row].to(device)
                length = int(mask.sum())
                if length != len(labels[pid]["aa"]):
                    raise RuntimeError(f"length mismatch for {pid}: {length}")
                protein_ids_out.append(pid)
                for block in range(1, 13):
                    _append(protein_features, f"block_{block}", blocks[block][row][mask].mean(0))
                _append(protein_features, "final_8d", final[row][mask].mean(0))

                local = selected_rows.get(pid)
                if local is None:
                    continue
                local_tensor = torch.as_tensor(local, device=device)
                for block in range(1, 13):
                    _append(residue_features, f"block_{block}", blocks[block][row, local_tensor])
                _append(residue_features, "final_8d", final[row, local_tensor])
                residue_pid.extend([pid] * len(local))
                residue_index.extend(local.tolist())
                residue_length.extend([length] * len(local))

            if start % (batch_size * 50) == 0:
                print(f"{spec.label}: {start + len(indices)}/{len(ordered_indices)} proteins", flush=True)
            del batch, blocks, final, output
    finally:
        capture.close()

    if len(protein_ids_out) != EXPECTED_PROTEINS or len(residue_index) != N_SAMPLE:
        raise RuntimeError(
            f"extraction count mismatch: proteins={len(protein_ids_out)}, residues={len(residue_index)}"
        )
    return {
        "sample_sha256": sample_sha,
        "pool": pool,
        "protein_ids": np.asarray(protein_ids_out),
        "residue_pid": np.asarray(residue_pid),
        "residue_index": np.asarray(residue_index, dtype=np.float64),
        "residue_length": np.asarray(residue_length, dtype=np.float64),
        "protein_features": {key: np.stack(values) for key, values in protein_features.items()},
        "residue_features": {key: np.concatenate(values, axis=0) for key, values in residue_features.items()},
    }


def fit_ridge(x_train, y_train, x_test):
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, gcv_mode="svd"),
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), float(model[-1].alpha_)


def metric_row(spec, representation, width, target, truth, prediction, alpha):
    rho = float(spearmanr(truth, prediction).statistic)
    return {
        "model": spec.label,
        "step": spec.step,
        "representation": representation,
        "width": width,
        "target": target,
        "spearman": rho,
        "r2": float(r2_score(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "ridge_alpha": alpha,
        "n_test": int(len(truth)),
    }


def analyze(spec: ModelSpec, extracted: dict):
    pool = extracted["pool"]
    labels = load_labels()
    protein_ids = extracted["protein_ids"]
    residue_pid = extracted["residue_pid"]
    protein_test = np.asarray([pool.probe_split[pid] == "probe_test" for pid in protein_ids])
    residue_test = np.asarray([pool.probe_split[pid] == "probe_test" for pid in residue_pid])
    protein_lengths = np.asarray([len(labels[pid]["aa"]) for pid in protein_ids], dtype=np.float64)
    residue_index = extracted["residue_index"]
    residue_length = extracted["residue_length"]
    position_targets = np.column_stack((
        residue_index,
        (residue_index + 1.0) / residue_length,
    ))
    rows = []
    for key in [f"block_{block}" for block in range(1, 13)] + ["final_8d"]:
        px = extracted["protein_features"][key]
        rx = extracted["residue_features"][key]
        length_prediction, length_alpha = fit_ridge(
            px[~protein_test], np.log(protein_lengths[~protein_test]), px[protein_test]
        )
        rows.append(metric_row(
            spec, key, px.shape[1], "log_chain_length",
            np.log(protein_lengths[protein_test]), length_prediction, length_alpha,
        ))
        position_prediction, position_alpha = fit_ridge(
            rx[~residue_test], position_targets[~residue_test], rx[residue_test]
        )
        for column, target in enumerate(("raw_residue_index", "normalized_position")):
            rows.append(metric_row(
                spec, key, rx.shape[1], target,
                position_targets[residue_test, column], position_prediction[:, column], position_alpha,
            ))
        truth_quintile = np.minimum((position_targets[residue_test, 1] * 5).astype(int), 4)
        pred_quintile = np.minimum(np.maximum((position_prediction[:, 1] * 5).astype(int), 0), 4)
        rows[-1]["quintile_accuracy"] = float(np.mean(truth_quintile == pred_quintile))
    return rows


def write_results(model_key: str, spec: ModelSpec, rows: list[dict], sample_sha: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{model_key}_layerwise_position_length.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT_DIR / f"{model_key}_layerwise_position_length.json"
    json_path.write_text(json.dumps({
        "model_key": model_key,
        "model": spec.label,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_sha256": spec.checkpoint_sha256,
        "source_checkpoint_sha256": spec.source_sha256,
        "physical_sample_sha256": sample_sha,
        "protein_split": "fixed 80/20 split by protein, seed 42",
        "position_sample_residues": N_SAMPLE,
        "chain_length_unit": "one mean embedding per protein",
        "probe": "StandardScaler + RidgeCV",
        "ridge_alphas": ALPHAS.tolist(),
        "rows": rows,
    }, indent=2) + "\n")
    return csv_path, json_path


def plot_if_complete():
    paths = {key: OUT_DIR / f"{key}_layerwise_position_length.csv" for key in MODELS}
    if not all(path.exists() for path in paths.values()):
        return None
    runtime = pathlib.Path.home() / "tmp/layerwise-position-length/matplotlib"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(runtime))
    import matplotlib.pyplot as plt

    colors = {"uncropped": "#8F5DA2", "cropped_protein": "#0072B2"}
    targets = (
        ("log_chain_length", "Log chain length"),
        ("raw_residue_index", "Raw residue index"),
        ("normalized_position", "Normalized residue position"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    for model_key, path in paths.items():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for ax, (target, title) in zip(axes, targets):
            selected = [row for row in rows if row["target"] == target]
            x = list(range(1, 13)) + [13.7]
            y = [float(row["spearman"]) for row in selected]
            ax.plot(x, y, marker="o", linewidth=2, markersize=4.5,
                    color=colors[model_key], label=MODELS[model_key].label)
            ax.set_title(title, fontsize=12, fontweight="demibold")
            ax.set_xticks((1, 3, 5, 7, 9, 11, 13.7), ("1", "3", "5", "7", "9", "11", "Final\n8D"))
            ax.set_xlabel("Encoder block")
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out Spearman correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Layerwise sequence-position and chain-length decodability", y=1.10,
                 fontsize=14, fontweight="demibold")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "canonical_lejepa_layerwise_position_length_decodability.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        print(plot_if_complete())
        return
    if args.model is None:
        parser.error("--model is required unless --plot-only is used")
    spec = MODELS[args.model]
    extracted = extract_features(spec, args.device, args.batch_size)
    rows = analyze(spec, extracted)
    print(write_results(args.model, spec, rows, extracted["sample_sha256"]))
    print(plot_if_complete())


if __name__ == "__main__":
    main()
