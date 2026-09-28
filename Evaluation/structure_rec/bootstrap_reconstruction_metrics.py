"""Protein-level bootstrap intervals for reconstruction metrics.

The input is produced by ``compare_ca_cond_jepa_reconstruction.py``.  Whole
proteins, rather than atoms, are resampled so that long chains do not dominate
the uncertainty estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def interval(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--jepa-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_921)
    args = parser.parse_args()

    report = json.loads(args.metrics.read_text())
    records = report["proteins"]
    protein_ids = [record["protein_id"] for record in records]
    if len(protein_ids) != 100 or len(set(protein_ids)) != 100:
        raise ValueError("expected exactly 100 unique proteins")

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(records), size=(args.replicates, len(records)))
    intervals: dict[str, object] = {}
    comparisons = records[0]["comparisons"]
    for comparison, model_values in comparisons.items():
        intervals[comparison] = {}
        for model, values in model_values.items():
            intervals[comparison][model] = {}
            for key in ("rmsd_angstrom", "lddt", "tm_score_ca"):
                metric_values = [
                    record["comparisons"][comparison][model][key]
                    for record in records
                ]
                if any(value is None for value in metric_values):
                    continue
                intervals[comparison][model][key] = interval(
                    np.asarray(metric_values, dtype=np.float64), indices
                )

    summary = json.loads(args.jepa_summary.read_text())
    rates_by_id = {
        item["protein_id"]: float(item["sequence_recovery"])
        for item in summary["proteins"]
    }
    if not set(protein_ids).issubset(rates_by_id):
        missing = sorted(set(protein_ids) - set(rates_by_id))
        raise ValueError(f"JEPA summary is missing cohort proteins: {missing}")
    rates = np.asarray([rates_by_id[protein_id] for protein_id in protein_ids])
    intervals["sequence_recovery"] = {
        "ca_conditioned_baseline": {
            "mean": 1.0,
            "ci95_lower": 1.0,
            "ci95_upper": 1.0,
        },
        "jepa": interval(rates, indices),
    }

    result = {
        "protocol": {
            "sampling_unit": "protein",
            "protein_count": len(protein_ids),
            "replicates": args.replicates,
            "seed": args.seed,
            "interval": "percentile 95%",
        },
        "source_metrics": str(args.metrics.resolve()),
        "source_jepa_summary": str(args.jepa_summary.resolve()),
        "protein_ids": protein_ids,
        "intervals": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
