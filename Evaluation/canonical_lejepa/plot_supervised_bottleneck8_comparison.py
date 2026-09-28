#!/usr/bin/env python3
"""Compare 768D block readouts with task-supervised and native final 8D readouts."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ROOT
    / "StructTokenBench"
    / "struct_token_bench_logs"
    / "upstream-l600-v1-full"
    / "runs"
)
LAYERWISE_CSV = (
    Path(__file__).resolve().parent
    / "results"
    / "layerwise_stb_all_blocks"
    / "canonical_lejepa_raw100k_layerwise_stb.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "results" / "supervised_bottleneck8"
FIGURE_DIR = ROOT / "thesis-latex" / "figures" / "results" / "block-investigation"

BOTTLENECK_RUNS = {
    block: (
        RUNS
        / f"slurm-261428_{block}-canonical-lejepa-v1-raw-step100k-"
        f"block{block}-supervised-bottleneck8-all"
    )
    for block in (4, 8, 12)
}
NATIVE_FINAL_RUN = RUNS / "slurm-258768_0-canonical-lejepa-v1-raw-step100k-all"

RESULT_LINE = re.compile(
    r"│\s*(?P<metric>test_(?:family|fold|superfamily)_holdout_f1_score|test_auroc)"
    r"\s*│\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*│"
)
FAMILY_FOR_METRIC = {
    "AUROC": "Functional sites",
    "Spearman": "Flexibility",
    "Macro-F1": "Remote homology",
}


def read_run(run_dir: Path) -> dict[tuple[str, str, str], float]:
    scores = {}
    with (run_dir / "summary_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            metric = {"auroc": "AUROC", "spearmanr": "Spearman"}[row["metric"]]
            scores[(metric, row["task"], "Fold")] = float(row["fold_test"])
            scores[(metric, row["task"], "SF")] = float(row["superfamily_test"])

    for path in (run_dir / "results").glob("*.txt"):
        values = {
            match.group("metric"): float(match.group("value"))
            for match in map(RESULT_LINE.search, path.read_text().splitlines())
            if match
        }
        if "_BindShake_" in path.name and "test_auroc" in values:
            scores[("AUROC", "BindShake", "Test")] = values["test_auroc"]
        if "_Homo_" in path.name:
            for key, split in (
                ("test_fold_holdout_f1_score", "Fold"),
                ("test_family_holdout_f1_score", "Family"),
                ("test_superfamily_holdout_f1_score", "SF"),
            ):
                if key in values:
                    scores[("Macro-F1", "Homo", split)] = values[key]
    return scores


def read_768d_blocks() -> dict[int, dict[tuple[str, str, str], float]]:
    result = defaultdict(dict)
    with LAYERWISE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            block = int(row["block"])
            if block in BOTTLENECK_RUNS:
                key = (row["metric"], row["task"], row["split"])
                result[block][key] = float(row["score"])
    return dict(result)


def aggregate(scores):
    grouped = defaultdict(list)
    for (metric, _task, _split), value in scores.items():
        grouped[FAMILY_FOR_METRIC[metric]].append(value * 100)
    expected = {"Functional sites": 15, "Flexibility": 6, "Remote homology": 3}
    for family, count in expected.items():
        if len(grouped[family]) != count:
            raise ValueError(
                f"Expected {count} {family} scores, found {len(grouped[family])}"
            )
    return {family: sum(values) / len(values) for family, values in grouped.items()}


def write_summary(rows):
    path = OUT_DIR / "canonical_lejepa_raw100k_supervised_bottleneck8_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("representation", "block", "task_family", "score_x100"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figure(rows):
    lookup = {
        (row["representation"], row["block"], row["task_family"]): row["score_x100"]
        for row in rows
    }
    families = ("Functional sites", "Flexibility", "Remote homology")
    ylabel = {
        "Functional sites": "Mean AUROC",
        "Flexibility": r"Mean Spearman's $\rho$",
        "Remote homology": "Mean macro-F1",
    }
    ylim = {
        "Functional sites": (0, 88),
        "Flexibility": (0, 62),
        "Remote homology": (0, 34),
    }
    x = np.arange(4)
    width = 0.31
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3))
    for ax, family in zip(axes, families):
        values_768 = [lookup[("768D block", str(block), family)] for block in (4, 8, 12)]
        values_task8 = [
            lookup[("Task-supervised 8D", str(block), family)] for block in (4, 8, 12)
        ]
        bars_768 = ax.bar(
            x[:3] - width / 2,
            values_768,
            width,
            color="#8F5DA2",
            label="768D block",
        )
        bars_task8 = ax.bar(
            x[:3] + width / 2,
            values_task8,
            width,
            color="#C7B3DD",
            label="Task-supervised 8D",
        )
        native_final8 = lookup[("Native final 8D", "final", family)]
        bars_final8 = ax.bar(
            x[3],
            native_final8,
            width * 1.35,
            color="#0072B2",
            label="Final 8D",
        )
        for bars in (bars_768, bars_task8, bars_final8):
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(family, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel[family], fontsize=11)
        ax.set_ylim(*ylim[family])
        ax.set_xticks(x, ("Block 4", "Block 8", "Block 12", "Final 8D"))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.13, wspace=0.28)
    path = FIGURE_DIR / "canonical_lejepa_raw100k_supervised_bottleneck8_comparison.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
        }
    )

    rows = []
    full_width = read_768d_blocks()
    for block in (4, 8, 12):
        for representation, scores in (
            ("768D block", full_width[block]),
            ("Task-supervised 8D", read_run(BOTTLENECK_RUNS[block])),
        ):
            for family, value in aggregate(scores).items():
                rows.append(
                    {
                        "representation": representation,
                        "block": str(block),
                        "task_family": family,
                        "score_x100": value,
                    }
                )
    for family, value in aggregate(read_run(NATIVE_FINAL_RUN)).items():
        rows.append(
            {
                "representation": "Native final 8D",
                "block": "final",
                "task_family": family,
                "score_x100": value,
            }
        )

    csv_path = write_summary(rows)
    figure_path = save_figure(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
