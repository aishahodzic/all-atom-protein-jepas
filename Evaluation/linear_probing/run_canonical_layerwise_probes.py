"""Layerwise linear probes for canonical LeJEPA full-context residue states."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from openfold.np import residue_constants
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Evaluation.linear_probing.data_pool import build_protein_pool
from Evaluation.linear_probing.probe import PROPERTIES, SEED, _gather


REPO = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO / (
    "store/canonical_lejepa_v1_100k/evaluation/"
    "full_context_layer_states/full"
)
LABELS_PATH = REPO / "Evaluation/linear_probing/latents/labels.pt"
RESULTS_DIR = REPO / "Evaluation/linear_probing/results"
OUTPUT_STEM = "canonical_lejepa_full_context_layerwise_probes"
EXPECTED_CHECKPOINT_SHA256 = (
    "4f09dff610adaa75c8ac51b4cf01acb001c236c72dd91ac31568ffa191484527"
)

DEPTHS = {
    "block_1": {"label": "Block 1 (768D)", "dimension": 768},
    "block_6": {"label": "Block 6 (768D)", "dimension": 768},
    "block_12": {"label": "Block 12 (768D)", "dimension": 768},
    "latent_mean": {"label": "Latent mean (8D)", "dimension": 8},
}
PROPERTY_LABELS = {
    "aa": "AA accuracy (%)",
    "ss": "Secondary-structure accuracy (%)",
    "rsa": "RSA R²",
    "bfactor": "B-factor Spearman ρ",
}
METRIC_NAMES = {
    "aa": "accuracy",
    "ss": "accuracy",
    "rsa": "r2",
    "bfactor": "spearman_rho",
}
FIT_ORDER = [
    ("latent_mean", "aa"),
    *(("block_1", prop) for prop in PROPERTIES),
    *(("block_6", prop) for prop in PROPERTIES),
    *(("block_12", prop) for prop in PROPERTIES),
    *(("latent_mean", prop) for prop in PROPERTIES if prop != "aa"),
]

CSV_FIELDS = [
    "depth",
    "dimension",
    "property",
    "metric",
    "test_score",
    "valid_train_residues",
    "valid_test_residues",
    "seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "global_step",
    "encoder_source",
    "evaluation_context",
    "convergence_warning",
    "fit_warnings",
]


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


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    with (input_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["counts"] != {"proteins": 5224, "encoder_valid_residues": 748753}:
        raise AssertionError(f"unexpected extraction counts: {manifest['counts']}")
    if manifest["selection"]["probe_split_counts"] != {
        "probe_train": 4179,
        "probe_test": 1045,
    }:
        raise AssertionError("extraction probe split counts changed")
    provenance = manifest["provenance"]
    if (
        provenance["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256
        or provenance["global_step"] != 100_000
        or provenance["encoder_source"] != "ema"
        or manifest["capture"]["context"] != "full_unmasked_evaluation"
    ):
        raise AssertionError("unexpected checkpoint or capture provenance")
    if manifest["capture"]["blocks"] != {
        "block_1": 1,
        "block_6": 6,
        "block_12": 12,
    }:
        raise AssertionError("unexpected captured blocks")
    return manifest


def _iter_records(input_dir: Path, manifest: dict[str, Any]):
    for shard_info in manifest["shards"]:
        shard = torch.load(
            input_dir / shard_info["path"], map_location="cpu", weights_only=False
        )
        if len(shard) != shard_info["proteins"]:
            raise AssertionError(f"shard record-count mismatch: {shard_info['path']}")
        for record in shard:
            yield record
        del shard


def _selection_digest(
    protein_ids: list[str],
    probe_split: dict[str, str],
    labels: dict[str, dict[str, np.ndarray]],
    property_name: str,
    split_name: str,
) -> tuple[str, int]:
    valid_key = PROPERTIES[property_name]["valid_key"]
    digest = hashlib.sha256()
    count = 0
    for protein_id in protein_ids:
        if probe_split[protein_id] != split_name:
            continue
        rows = np.flatnonzero(labels[protein_id][valid_key]).astype("<i8", copy=False)
        encoded = protein_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(rows).to_bytes(8, "little"))
        digest.update(rows.tobytes())
        count += len(rows)
    return digest.hexdigest(), count


def validate_alignment(
    input_dir: Path,
    manifest: dict[str, Any],
    labels: dict[str, dict[str, np.ndarray]],
) -> tuple[list[str], dict[str, str], dict[str, dict[str, dict[str, Any]]]]:
    protein_ids = manifest["selection"]["selected_protein_ids"]
    if len(protein_ids) != len(set(protein_ids)):
        raise AssertionError("duplicate protein IDs in extraction manifest")
    if set(labels) != set(protein_ids):
        missing = sorted(set(protein_ids) - set(labels))
        extra = sorted(set(labels) - set(protein_ids))
        raise AssertionError(f"label protein mismatch: missing={missing[:3]} extra={extra[:3]}")

    pool = build_protein_pool(seed=SEED)
    if set(pool.protein_ids) != set(protein_ids):
        raise AssertionError("current evaluation pool does not match extraction manifest")
    pool_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    seen: list[str] = []
    probe_split: dict[str, str] = {}

    for record in _iter_records(input_dir, manifest):
        protein_id = record["protein_id"]
        if protein_id not in pool_index:
            raise AssertionError(f"unknown saved protein: {protein_id}")
        graph = pool.dataset[pool_index[protein_id]]
        lab = labels[protein_id]
        rows = record["original_dense_row_index"].numpy().astype(np.int64)
        if not np.array_equal(rows, np.arange(graph.coords.shape[0])):
            raise AssertionError(f"ambiguous/non-complete dense row mapping: {protein_id}")
        for key in (
            "aa",
            "aa_valid",
            "ss",
            "ss_valid",
            "rsa",
            "rsa_valid",
            "bfactor_z",
            "bfactor_valid",
        ):
            if len(lab[key]) != graph.coords.shape[0]:
                raise AssertionError(f"label length mismatch for {protein_id}/{key}")
        saved_aa = record["aa_integer"].numpy()
        if not np.array_equal(saved_aa, lab["aa"][rows]):
            raise AssertionError(f"cached AA alignment mismatch: {protein_id}")
        if not np.array_equal(saved_aa, graph.residue_type.numpy()[rows]):
            raise AssertionError(f"source AA alignment mismatch: {protein_id}")
        if not np.array_equal(
            record["residue_pdb_idx"].numpy(), graph.residue_pdb_idx.numpy()[rows]
        ):
            raise AssertionError(f"PDB residue-index alignment mismatch: {protein_id}")
        if not np.array_equal(record["chain_index"].numpy(), graph.chains.numpy()[rows]):
            raise AssertionError(f"chain-index alignment mismatch: {protein_id}")
        if record["chain_identifier"] != str(pool.dataset.chains[pool_index[protein_id]]):
            raise AssertionError(f"chain-identifier alignment mismatch: {protein_id}")
        expected_aa_labels = [residue_constants.restypes_with_x[int(v)] for v in saved_aa]
        if record["aa_label"] != expected_aa_labels:
            raise AssertionError(f"AA-label mapping mismatch: {protein_id}")
        if record["probe_split"] != pool.probe_split[protein_id]:
            raise AssertionError(f"probe split mismatch: {protein_id}")
        for depth, config in DEPTHS.items():
            if record[depth].shape != (len(rows), config["dimension"]):
                raise AssertionError(f"state shape mismatch: {protein_id}/{depth}")
        probe_split[protein_id] = record["probe_split"]
        seen.append(protein_id)

    if seen != protein_ids:
        raise AssertionError("shard protein order/content differs from manifest")
    train_ids = {pid for pid in protein_ids if probe_split[pid] == "probe_train"}
    test_ids = {pid for pid in protein_ids if probe_split[pid] == "probe_test"}
    if train_ids & test_ids or train_ids | test_ids != set(protein_ids):
        raise AssertionError("probe train/test IDs are not a disjoint partition")
    if (len(train_ids), len(test_ids)) != (4179, 1045):
        raise AssertionError("probe train/test protein counts changed")

    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for property_name in PROPERTIES:
        selections[property_name] = {}
        for split_name in ("probe_train", "probe_test"):
            digest, count = _selection_digest(
                protein_ids, probe_split, labels, property_name, split_name
            )
            selections[property_name][split_name] = {
                "sha256": digest,
                "residues": count,
            }
    return protein_ids, probe_split, selections


def load_depth(
    input_dir: Path,
    manifest: dict[str, Any],
    depth: str,
    protein_ids: list[str],
) -> dict[str, torch.Tensor]:
    dimension = DEPTHS[depth]["dimension"]
    values: dict[str, torch.Tensor] = {}
    observed_ids = []
    for record in _iter_records(input_dir, manifest):
        protein_id = record["protein_id"]
        tensor = record[depth]
        if tensor.dtype != torch.float32 or tensor.ndim != 2 or tensor.shape[1] != dimension:
            raise AssertionError(f"invalid state tensor: {protein_id}/{depth}")
        if tensor.shape[0] != len(record["original_dense_row_index"]):
            raise AssertionError(f"state/metadata row mismatch: {protein_id}/{depth}")
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"non-finite state: {protein_id}/{depth}")
        values[protein_id] = tensor
        observed_ids.append(protein_id)
    if observed_ids != protein_ids:
        raise AssertionError(f"protein selection changed while loading {depth}")
    return values


def fit_probe(
    property_name: str,
    protein_ids: list[str],
    probe_split: dict[str, str],
    labels: dict[str, dict[str, np.ndarray]],
    latents: dict[str, torch.Tensor],
    selections: dict[str, dict[str, dict[str, Any]]],
    seed: int,
) -> tuple[float, int, int, list[str], dict[str, Any]]:
    spec = PROPERTIES[property_name]
    X_train, y_train = _gather(
        protein_ids,
        probe_split,
        "probe_train",
        labels,
        spec["label_key"],
        spec["valid_key"],
        latents_by_protein=latents,
    )
    X_test, y_test = _gather(
        protein_ids,
        probe_split,
        "probe_test",
        labels,
        spec["label_key"],
        spec["valid_key"],
        latents_by_protein=latents,
    )
    if len(y_train) != selections[property_name]["probe_train"]["residues"]:
        raise AssertionError(f"train row selection changed for {property_name}")
    if len(y_test) != selections[property_name]["probe_test"]["residues"]:
        raise AssertionError(f"test row selection changed for {property_name}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if spec["kind"] == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            )
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    warning_messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    if not np.isfinite(predictions).all():
        raise AssertionError(f"non-finite predictions for {property_name}")
    if spec["kind"] == "classification":
        score = float(accuracy_score(y_test, predictions))
        recomputed = float(np.equal(predictions, y_test).sum() / len(y_test))
        if score != recomputed:
            raise AssertionError(f"accuracy recomputation mismatch for {property_name}")
    elif spec["kind"] == "regression_r2":
        score = float(r2_score(y_test, predictions))
        recomputed = score
    else:
        score = float(spearmanr(y_test, predictions).statistic)
        recomputed = score
    if not np.isfinite(score):
        raise AssertionError(f"non-finite score for {property_name}")
    diagnostics = {
        "prediction_count": int(len(predictions)),
        "prediction_finite": True,
        "recomputed_score": recomputed,
        "classes": (
            sorted(np.unique(predictions).astype(int).tolist())
            if spec["kind"] == "classification"
            else None
        ),
    }
    del X_train, X_test, y_train, y_test, predictions, model
    gc.collect()
    return score, selections[property_name]["probe_train"]["residues"], selections[property_name]["probe_test"]["residues"], warning_messages, diagnostics


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    buffer = []
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        buffer.append(handle.read())
    _atomic_write_text("".join(buffer), path)


def make_plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
    row_map = {(row["depth"], row["property"]): row for row in rows}
    depth_order = list(DEPTHS)
    x = np.arange(len(depth_order))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for axis, property_name in zip(axes.flat, PROPERTIES):
        values = np.array(
            [float(row_map[(depth, property_name)]["test_score"]) for depth in depth_order]
        )
        plotted = values * 100.0 if property_name in ("aa", "ss") else values
        line = axis.plot(x, plotted, marker="o", linewidth=2.0, markersize=6.5)[0]
        if not np.array_equal(line.get_ydata(), plotted):
            raise AssertionError(f"plotted values differ from CSV rows: {property_name}")
        axis.set_title(PROPERTY_LABELS[property_name])
        axis.set_xticks(x, [DEPTHS[depth]["label"] for depth in depth_order], rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.15)
        for x_value, y_value in zip(x, plotted):
            axis.annotate(
                f"{y_value:.2f}" if property_name in ("aa", "ss") else f"{y_value:.3f}",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    fig.suptitle("Canonical LeJEPA — 100k EMA, unmasked context", fontsize=15)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("this analysis requires the established seed 42")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = _load_manifest(input_dir)
    labels = torch.load(args.labels, map_location="cpu", weights_only=False)
    protein_ids, probe_split, selections = validate_alignment(input_dir, manifest, labels)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "alignment": {
            "passed": True,
            "protein_counts": manifest["selection"]["probe_split_counts"],
            "property_row_selections": selections,
        },
        "fits": {},
    }
    loaded_depth = None
    latents = None
    for depth, property_name in FIT_ORDER:
        if depth != loaded_depth:
            del latents
            gc.collect()
            latents = load_depth(input_dir, manifest, depth, protein_ids)
            loaded_depth = depth
        score, n_train, n_test, fit_warnings, fit_diagnostics = fit_probe(
            property_name,
            protein_ids,
            probe_split,
            labels,
            latents,
            selections,
            args.seed,
        )
        convergence_warning = any(
            message.startswith("ConvergenceWarning:") for message in fit_warnings
        )
        key = f"{depth}/{property_name}"
        diagnostics["fits"][key] = {
            **fit_diagnostics,
            "warnings": fit_warnings,
            "convergence_warning": convergence_warning,
        }
        provenance = manifest["provenance"]
        rows.append(
            {
                "depth": depth,
                "dimension": DEPTHS[depth]["dimension"],
                "property": property_name,
                "metric": METRIC_NAMES[property_name],
                "test_score": score,
                "valid_train_residues": n_train,
                "valid_test_residues": n_test,
                "seed": args.seed,
                "checkpoint_path": provenance["checkpoint_path"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "global_step": provenance["global_step"],
                "encoder_source": provenance["encoder_source"],
                "evaluation_context": manifest["capture"]["context"],
                "convergence_warning": convergence_warning,
                "fit_warnings": " | ".join(fit_warnings),
            }
        )
        print(
            f"FIT {len(rows):02d}/16 {depth:11s} {property_name:7s} "
            f"score={score:.8f} n_train={n_train} n_test={n_test} "
            f"warnings={len(fit_warnings)}",
            flush=True,
        )

    if len(rows) != 16 or len({(row["depth"], row["property"]) for row in rows}) != 16:
        raise AssertionError("expected exactly 16 unique probe results")
    if not all(np.isfinite(float(row["test_score"])) for row in rows):
        raise AssertionError("CSV would contain a non-finite score")

    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    report_path = output_dir / f"{OUTPUT_STEM}.json"
    _write_csv(rows, csv_path)
    make_plot(rows, png_path, pdf_path)
    diagnostics["outputs"] = {
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    diagnostics["provenance"] = manifest["provenance"]
    diagnostics["fit_order"] = [f"{depth}/{prop}" for depth, prop in FIT_ORDER]
    _atomic_write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", report_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    if len(saved_rows) != 16:
        raise AssertionError("saved CSV does not contain 16 rows")
    if any(not np.isfinite(float(row["test_score"])) for row in saved_rows):
        raise AssertionError("saved CSV contains non-finite scores")
    print(json.dumps(diagnostics["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
