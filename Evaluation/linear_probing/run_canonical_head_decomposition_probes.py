"""Probe the canonical LeJEPA 768D-to-16D-to-8D head decomposition."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from Evaluation.canonical_lejepa.checkpoint import extract_canonical_encoder_state_dict
from Evaluation.linear_probing.probe import PROPERTIES, SEED
from Evaluation.linear_probing.run_canonical_layerwise_probes import (
    EXPECTED_CHECKPOINT_SHA256,
    INPUT_DIR,
    LABELS_PATH,
    RESULTS_DIR,
    _iter_records,
    _load_manifest,
    fit_probe,
    validate_alignment,
)


OUTPUT_STEM = "canonical_lejepa_head_decomposition_probes"
REPRESENTATIONS = {
    "block_12_layernorm": {
        "label": "Block 12 + LN\n(768D)",
        "dimension": 768,
    },
    "full_head": {"label": "Full head\n(16D)", "dimension": 16},
    "latent_mean": {"label": "Mean\n(8D)", "dimension": 8},
    "log_scale": {"label": "Log-scale\n(8D)", "dimension": 8},
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
    *(("full_head", prop) for prop in PROPERTIES),
    *(("latent_mean", prop) for prop in PROPERTIES),
    *(("log_scale", prop) for prop in PROPERTIES),
    *(("block_12_layernorm", prop) for prop in PROPERTIES),
]
CSV_FIELDS = [
    "representation",
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
    "canonical_objective_uses_representation",
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


def load_ema_head(manifest: dict[str, Any]) -> torch.nn.Sequential:
    provenance = manifest["provenance"]
    if provenance["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("unexpected source checkpoint")
    checkpoint = torch.load(
        provenance["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    encoder_state = extract_canonical_encoder_state_dict(checkpoint, source="ema")
    prefix = "latent_decoder_mean_n_log_scale."
    head_state = {
        key.removeprefix(prefix): value
        for key, value in encoder_state.items()
        if key.startswith(prefix)
    }
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(768),
        torch.nn.Linear(768, 16, bias=False),
    )
    head.load_state_dict(head_state, strict=True)
    head.requires_grad_(False).eval()
    del checkpoint, encoder_state, head_state
    return head


def load_representation(
    input_dir: Path,
    manifest: dict[str, Any],
    representation: str,
    protein_ids: list[str],
    head: torch.nn.Sequential,
) -> tuple[dict[str, torch.Tensor], float]:
    values: dict[str, torch.Tensor] = {}
    observed_ids = []
    max_mean_difference = 0.0
    dimension = REPRESENTATIONS[representation]["dimension"]
    with torch.inference_mode():
        for record in _iter_records(input_dir, manifest):
            protein_id = record["protein_id"]
            block_12 = record["block_12"]
            normalized = head[0](block_12)
            full_head = None
            if representation == "block_12_layernorm":
                tensor = normalized
            elif representation == "latent_mean":
                tensor = record["latent_mean"]
            else:
                full_head = head[1](normalized)
                difference = float(
                    (full_head[:, :8] - record["latent_mean"]).abs().max().item()
                )
                max_mean_difference = max(max_mean_difference, difference)
                torch.testing.assert_close(
                    full_head[:, :8],
                    record["latent_mean"],
                    rtol=1e-5,
                    atol=1e-6,
                )
                tensor = full_head if representation == "full_head" else full_head[:, 8:]
            tensor = tensor.detach().float().contiguous()
            if tensor.shape != (len(record["original_dense_row_index"]), dimension):
                raise AssertionError(f"invalid reconstructed shape: {protein_id}/{representation}")
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"non-finite reconstruction: {protein_id}/{representation}")
            values[protein_id] = tensor
            observed_ids.append(protein_id)
            del block_12, normalized, full_head
    if observed_ids != protein_ids:
        raise AssertionError(f"protein selection changed for {representation}")
    return values, max_mean_difference


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        text = handle.read()
    _atomic_write_text(text, path)


def make_plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
    row_map = {(row["representation"], row["property"]): row for row in rows}
    representation_order = list(REPRESENTATIONS)
    x = np.arange(len(representation_order))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    for axis, property_name in zip(axes.flat, PROPERTIES):
        values = np.array(
            [
                float(row_map[(representation, property_name)]["test_score"])
                for representation in representation_order
            ]
        )
        plotted = values * 100.0 if property_name in ("aa", "ss") else values
        line = axis.plot(x, plotted, marker="o", linewidth=2.0, markersize=6.5)[0]
        if not np.array_equal(line.get_ydata(), plotted):
            raise AssertionError(f"plotted values differ from CSV: {property_name}")
        axis.set_title(PROPERTY_LABELS[property_name])
        axis.set_xticks(
            x,
            [REPRESENTATIONS[name]["label"] for name in representation_order],
        )
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
    fig.suptitle(
        "Canonical LeJEPA head decomposition — 100k EMA, unmasked context",
        fontsize=15,
    )
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
    head = load_ema_head(manifest)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "alignment": {
            "passed": True,
            "protein_counts": manifest["selection"]["probe_split_counts"],
            "property_row_selections": selections,
        },
        "head": {
            "mapping": "LayerNorm(768) -> Linear(768,16,bias=False) -> mean[:8], log_scale[8:]",
            "weights_source": "top-level state_dict_ema",
            "canonical_objective_consumes": "mean only",
            "encoder_rerun": False,
        },
        "fits": {},
    }
    loaded_representation = None
    latents = None
    for representation, property_name in FIT_ORDER:
        if representation != loaded_representation:
            del latents
            gc.collect()
            latents, max_mean_difference = load_representation(
                input_dir,
                manifest,
                representation,
                protein_ids,
                head,
            )
            diagnostics["head"].setdefault("mean_reconstruction_max_abs_difference", {})[
                representation
            ] = max_mean_difference
            loaded_representation = representation
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
        key = f"{representation}/{property_name}"
        diagnostics["fits"][key] = {
            **fit_diagnostics,
            "warnings": fit_warnings,
            "convergence_warning": convergence_warning,
        }
        provenance = manifest["provenance"]
        rows.append(
            {
                "representation": representation,
                "dimension": REPRESENTATIONS[representation]["dimension"],
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
                "canonical_objective_uses_representation": representation == "latent_mean",
                "convergence_warning": convergence_warning,
                "fit_warnings": " | ".join(fit_warnings),
            }
        )
        print(
            f"FIT {len(rows):02d}/16 {representation:22s} {property_name:7s} "
            f"score={score:.8f} warnings={len(fit_warnings)}",
            flush=True,
        )

    if len(rows) != 16 or len({(r["representation"], r["property"]) for r in rows}) != 16:
        raise AssertionError("expected exactly 16 unique results")
    if not all(np.isfinite(float(row["test_score"])) for row in rows):
        raise AssertionError("non-finite probe score")

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
    diagnostics["fit_order"] = [f"{rep}/{prop}" for rep, prop in FIT_ORDER]
    _atomic_write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", report_path)
    print(json.dumps(diagnostics["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
