"""Generate result-summary figure previews requested in sec5-results.tex.

The plots use only archived, provenance-checked outputs:

* ``thesis-latex/handoff/experiments/evaluation_metadata.json`` for L600 STB;
* the original 10k chemical-mask W&B export under
  ``thesis-latex/handoff/wandb/protein_jepa/``.

No model inference or probe fitting is performed here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "thesis-latex/handoff/experiments/evaluation_metadata.json"
RAW_CANONICAL_RUNS = {
    "canonical-lejepa-v1-raw-step100k-upstream-l600-v1": (
        ROOT
        / "StructTokenBench/struct_token_bench_logs/upstream-l600-v1-full/runs"
        / "slurm-258768_0-canonical-lejepa-v1-raw-step100k-all/manifests"
    ),
    "canonical-lejepa-v1-rotations-raw-step100k-upstream-l600-v1": (
        ROOT
        / "StructTokenBench/struct_token_bench_logs/upstream-l600-v1-full/runs"
        / "slurm-258768_1-canonical-lejepa-v1-rotations-raw-step100k-all/manifests"
    ),
}
CHEM_DIR = (
    ROOT
    / "thesis-latex/handoff/wandb/protein_jepa"
    / "le_jepa_chemmask_aa_partial50_old_10k"
)
OUTPUT_DIR = ROOT / "Evaluation/diagnostics/figures/results_placeholders"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GREY = "#6B7280"


TASK_METRICS = [
    ("BindInt · Fold", "BindInt", "fold_test_auroc"),
    ("BindInt · SF", "BindInt", "superfamily_test_auroc"),
    ("BindBio · Fold", "BindBio", "fold_test_auroc"),
    ("BindBio · SF", "BindBio", "superfamily_test_auroc"),
    ("BindShake · Test", "BindShake", "test_auroc"),
    ("CatInt · Fold", "CatInt", "fold_test_auroc"),
    ("CatInt · SF", "CatInt", "superfamily_test_auroc"),
    ("CatBio · Fold", "CatBio", "fold_test_auroc"),
    ("CatBio · SF", "CatBio", "superfamily_test_auroc"),
    ("Con · Fold", "Con", "fold_test_auroc"),
    ("Con · SF", "Con", "superfamily_test_auroc"),
    ("Rep · Fold", "Rep", "fold_test_auroc"),
    ("Rep · SF", "Rep", "superfamily_test_auroc"),
    ("Ept · Fold", "Ept", "fold_test_auroc"),
    ("Ept · SF", "Ept", "superfamily_test_auroc"),
    ("Homo · Fold", "Homo", "test_fold_holdout_f1_score"),
    ("Homo · Family", "Homo", "test_family_holdout_f1_score"),
    ("Homo · SF", "Homo", "test_superfamily_holdout_f1_score"),
    ("FlexRMSF · Fold", "FlexRMSF", "fold_test_spearmanr"),
    ("FlexRMSF · SF", "FlexRMSF", "superfamily_test_spearmanr"),
    ("FlexBFactor · Fold", "FlexBFactor", "fold_test_spearmanr"),
    ("FlexBFactor · SF", "FlexBFactor", "superfamily_test_spearmanr"),
    ("FlexNEQ · Fold", "FlexNEQ", "fold_test_spearmanr"),
    ("FlexNEQ · SF", "FlexNEQ", "superfamily_test_spearmanr"),
]


def _metadata_records() -> list[dict]:
    with METADATA.open() as handle:
        records = json.load(handle)["records"]

    # These matched raw-weight runs finished after the compact metadata export.
    # Read their immutable task manifests directly and apply the same compact
    # record shape used above. Strict assertions prevent an EMA or mismatched
    # protocol result from entering the rotation comparison.
    for experiment, manifest_root in RAW_CANONICAL_RUNS.items():
        manifests = sorted(manifest_root.rglob("*.json"))
        if len(manifests) != 12:
            raise ValueError(f"Expected 12 raw canonical manifests in {manifest_root}, found {len(manifests)}")
        for manifest in manifests:
            with manifest.open() as handle:
                payload = json.load(handle)
            if payload["run"]["experiment"] != experiment:
                raise ValueError(f"Unexpected experiment in {manifest}")
            if payload["protocol"]["name"] != "upstream-l600-v1":
                raise ValueError(f"Unexpected STB protocol in {manifest}")
            if payload["protocol"]["encoder_source"] != "online":
                raise ValueError(f"Raw comparison did not use online weights in {manifest}")
            if payload["encoder"]["state_prefix"] != "online_encoder":
                raise ValueError(f"Raw comparison loaded an unexpected state prefix in {manifest}")
            records.append(
                {
                    "experiment": experiment,
                    "task": payload["run"]["task"],
                    "metrics": payload["metrics"],
                }
            )
    return records


def _record(records: list[dict], experiment: str, task: str) -> dict:
    matches = [
        record
        for record in records
        if record.get("experiment") == experiment and record.get("task") == task
    ]
    if not matches:
        raise KeyError(f"Missing archived result for {experiment=} {task=}")

    # A few tasks were repeated after interruption. Require repeated records to
    # agree rather than silently choosing whichever happens to appear last.
    reference = matches[0]["metrics"]
    for candidate in matches[1:]:
        common = reference.keys() & candidate["metrics"].keys()
        for metric in common:
            left, right = reference[metric], candidate["metrics"][metric]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if not np.isclose(left, right, rtol=0.0, atol=1e-10):
                    raise ValueError(
                        f"Conflicting repeats for {experiment=} {task=} {metric=}: "
                        f"{left} versus {right}"
                    )
    return matches[-1]


def _scores(records: list[dict], experiment: str) -> np.ndarray:
    return np.asarray(
        [
            float(_record(records, experiment, task)["metrics"][metric])
            for _, task, metric in TASK_METRICS
        ]
    )


def plot_masking_effects(records: list[dict]) -> Path:
    comparisons = [
        (
            "JEPA",
            "jepa-rand-mask-50-upstream-l600-v1",
            "jepa-rand-mask-70-upstream-l600-v1",
            BLUE,
        ),
        (
            "JEPA+SIGReg · absolute",
            "le-jepa-rand-mask-50-upstream-l600-v1",
            "le-jepa-rand-mask-70-upstream-l600-v1",
            ORANGE,
        ),
        (
            "JEPA+SIGReg · relative",
            "le-jepa-rand50-relative-pos-100k-upstream-l600-v1",
            "le-jepa-rand70-relative-pos-100k-upstream-l600-v1",
            GREEN,
        ),
    ]

    deltas = [100.0 * (_scores(records, high) - _scores(records, low)) for _, low, high, _ in comparisons]
    extent = max(5.0, float(np.ceil(max(np.abs(delta).max() for delta in deltas))))
    y = np.arange(len(TASK_METRICS))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 8.8), sharey=True)
    for ax, (title, _, _, color), delta in zip(axes, comparisons, deltas):
        ax.axvspan(-extent, 0, color="#FDECEC", alpha=0.55, zorder=0)
        ax.axvspan(0, extent, color="#EAF7EF", alpha=0.55, zorder=0)
        ax.axvline(0, color="#222222", linewidth=0.9, zorder=1)
        ax.scatter(delta, y, s=34, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlim(-extent, extent)
        ax.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("70% − 50% masking\n(score points)")
        ax.tick_params(axis="y", length=0)
        for boundary in (14.5, 17.5):
            ax.axhline(boundary, color="#9CA3AF", linewidth=0.8, linestyle="--")

    axes[0].set_yticks(y, [label for label, _, _ in TASK_METRICS], fontsize=8.7)
    axes[0].invert_yaxis()
    fig.suptitle("Task-wise effect of increasing target masking at 100k", fontsize=13, weight="bold")
    fig.text(0.012, 0.375, "Remote\nhomology", fontsize=8, color=GREY, va="center")
    fig.text(0.012, 0.165, "Flexibility", fontsize=8, color=GREY, va="center")
    fig.tight_layout(rect=(0.06, 0.02, 1, 0.965))
    return _save(fig, "masking_70_minus_50_taskwise_100k")


def _chemical_history() -> tuple[np.ndarray, np.ndarray, float, int]:
    history_path = CHEM_DIR / "history_sampled.csv"
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    paired = [
        row
        for row in rows
        if row.get("val_jepa_loss") and row.get("val_partial_context/loss_context_copy")
    ]
    predictor = np.asarray([float(row["val_jepa_loss"]) for row in paired])
    copy = np.asarray(
        [float(row["val_partial_context/loss_context_copy"]) for row in paired]
    )
    with (CHEM_DIR / "run.json").open() as handle:
        summary = json.load(handle)["summary"]
    rank = float(summary["val_effective_rank/target"])
    step = int(summary["trainer/global_step"])
    if len(paired) != 10 or int(np.sum(copy < predictor)) != 8:
        raise ValueError("Archived chemical-mask trajectory no longer matches its audit")
    return predictor, copy, rank, step


def plot_chemical_masking() -> Path:
    predictor, copy, target_rank, final_step = _chemical_history()
    records = np.arange(1, len(predictor) + 1)

    fig, (loss_ax, rank_ax) = plt.subplots(
        1, 2, figsize=(9.5, 4.0), gridspec_kw={"width_ratios": [3.2, 1]}
    )
    loss_ax.plot(records, predictor, "o-", color=ORANGE, label="Predictor", linewidth=2)
    loss_ax.plot(records, copy, "o-", color=BLUE, label="Context copy", linewidth=2)
    loss_ax.set_yscale("log")
    loss_ax.set_xticks(records)
    loss_ax.set_xlabel("Recorded validation order")
    loss_ax.set_ylabel("Validation MSE (log scale)")
    loss_ax.grid(color="#D1D5DB", linewidth=0.6, alpha=0.8)
    loss_ax.legend(frameon=False, ncol=2, loc="upper right")
    loss_ax.text(
        0.03,
        0.06,
        "Context copy has lower MSE at 8/10 records",
        transform=loss_ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "pad": 4},
    )

    rank_ax.bar([0], [target_rank], width=0.58, color=PURPLE)
    rank_ax.axhline(8, color="#222222", linestyle="--", linewidth=0.9)
    rank_ax.set_ylim(0, 8.35)
    rank_ax.set_xlim(-0.7, 0.7)
    rank_ax.set_xticks([0], ["Target"])
    rank_ax.set_ylabel("Effective rank (maximum 8)")
    rank_ax.text(0, target_rank - 0.32, f"{target_rank:.3f}", color="white", ha="center", va="top", weight="bold")
    rank_ax.text(0.5, -0.23, f"endpoint: trainer step {final_step:,}", transform=rank_ax.transAxes, ha="center", fontsize=8, color=GREY)
    rank_ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)

    fig.suptitle("Chemical masking: prediction versus copying despite full target rank", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, "chemical_mask_predictor_vs_copy_and_rank_10k")


def plot_rotation_effects(records: list[dict]) -> Path:
    families = [
        (
            "Canonical LeJEPA · raw weights",
            "canonical-lejepa-v1-raw-step100k-upstream-l600-v1",
            "canonical-lejepa-v1-rotations-raw-step100k-upstream-l600-v1",
            BLUE,
        ),
        (
            "Relative-position JEPA+SIGReg",
            "le-jepa-rand70-relative-pos-100k-upstream-l600-v1",
            "le-jepa-rand70-relative-pos-rotation-100k-upstream-l600-v1",
            ORANGE,
        ),
    ]
    metric_panels = [
        ("Fold", "fold_test_spearmanr"),
        ("Superfamily", "superfamily_test_spearmanr"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=True)
    for ax, (split_name, metric) in zip(axes, metric_panels):
        for label, no_rotation, rotation, color in families:
            before = 100.0 * float(_record(records, no_rotation, "FlexRMSF")["metrics"][metric])
            after = 100.0 * float(_record(records, rotation, "FlexRMSF")["metrics"][metric])
            ax.plot([0, 1], [before, after], "o-", color=color, linewidth=2.3, markersize=7, label=label)
            ax.text(-0.04, before, f"{before:.1f}", color=color, ha="right", va="center", fontsize=8.5)
            ax.text(1.04, after, f"{after:.1f}", color=color, ha="left", va="center", fontsize=8.5)
        ax.set_xlim(-0.28, 1.28)
        ax.set_xticks([0, 1], ["No rotations", "Independent\nrotations"])
        ax.set_title(split_name, weight="bold", fontsize=11)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("FlexRMSF Spearman’s ρ ×100")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Paired effect of independent rotations at 100k", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return _save(fig, "flexrmsf_rotation_effects_100k")


def plot_rotation_effects_all_tasks(records: list[dict]) -> Path:
    families = [
        (
            "Canonical LeJEPA · raw weights",
            "canonical-lejepa-v1-raw-step100k-upstream-l600-v1",
            "canonical-lejepa-v1-rotations-raw-step100k-upstream-l600-v1",
        ),
        (
            "JEPA+SIGReg · relative positions",
            "le-jepa-rand70-relative-pos-100k-upstream-l600-v1",
            "le-jepa-rand70-relative-pos-rotation-100k-upstream-l600-v1",
        ),
    ]
    groups = [
        ("Functional sites · AUROC ×100", TASK_METRICS[:15]),
        ("Remote homology · macro-F1 ×100", TASK_METRICS[15:18]),
        ("Flexibility · Spearman’s ρ ×100", TASK_METRICS[18:]),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.2, 11.2),
        gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]},
    )
    for row, (family, no_rotation, rotation) in enumerate(families):
        for col, (group_title, metrics) in enumerate(groups):
            ax = axes[row, col]
            y = np.arange(len(metrics))
            before = np.asarray(
                [
                    100.0 * float(_record(records, no_rotation, task)["metrics"][metric])
                    for _, task, metric in metrics
                ]
            )
            after = np.asarray(
                [
                    100.0 * float(_record(records, rotation, task)["metrics"][metric])
                    for _, task, metric in metrics
                ]
            )
            for index, (left, right) in enumerate(zip(before, after)):
                change_color = GREEN if right >= left else ORANGE
                ax.plot([left, right], [index, index], color=change_color, alpha=0.68, linewidth=2.1, zorder=1)
            ax.scatter(before, y, s=42, color=BLUE, edgecolor="white", linewidth=0.55, zorder=2, label="No rotations")
            ax.scatter(after, y, s=42, color=ORANGE, edgecolor="white", linewidth=0.55, zorder=3, label="Independent rotations")
            ax.set_yticks(y, [label for label, _, _ in metrics], fontsize=8.2)
            ax.invert_yaxis()
            ax.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.8)
            if row == 0:
                ax.set_title(group_title, fontsize=10.5, weight="bold")
            if col == 0:
                ax.set_ylabel(family, fontsize=10.5, weight="bold", labelpad=13)
            ax.set_xlabel("Downstream score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Paired task-wise effect of independent rotations at 100k", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.965), h_pad=2.3, w_pad=2.0)
    return _save(fig, "rotation_effects_all_tasks_100k")


def _save(fig: plt.Figure, stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 9,
            "figure.dpi": 120,
        }
    )
    records = _metadata_records()
    outputs = [
        plot_masking_effects(records),
        plot_chemical_masking(),
        plot_rotation_effects(records),
        plot_rotation_effects_all_tasks(records),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
