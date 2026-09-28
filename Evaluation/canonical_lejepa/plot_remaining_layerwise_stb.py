#!/usr/bin/env python3
"""Plot the remaining JEPA/cropped layerwise STB sweeps.

Produces one 4x3 per-task figure per model.  Missing task results remain gaps
and are labelled as pending, so rerunning this script safely refreshes the
cropped per-residue figure while its STB array is still completing.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

RUNTIME = Path(os.environ.get("TMPDIR", Path.home() / "tmp")) / "plot-remaining-layerwise-stb"
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(RUNTIME))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_layerwise_stb_all_blocks as common


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "StructTokenBench" / "struct_token_bench_logs" / "upstream-l600-v1-full" / "runs"
OUT_ROOT = Path(__file__).resolve().parent / "results" / "remaining_layerwise_stb"
FIGURE_DIR = ROOT / "iclr2027" / "figures" / "per-block-results"

TASKS = (
    ("AUROC", "BindInt"),
    ("AUROC", "BindBio"),
    ("AUROC", "BindShake"),
    ("AUROC", "CatInt"),
    ("AUROC", "CatBio"),
    ("AUROC", "Con"),
    ("AUROC", "Rep"),
    ("AUROC", "Ept"),
    ("Macro-F1", "Homo"),
    ("Spearman", "FlexRMSF"),
    ("Spearman", "FlexBFactor"),
    ("Spearman", "FlexNEQ"),
)
EXPECTED_SPLITS = {
    ("AUROC", "BindShake"): ("Test",),
    ("Macro-F1", "Homo"): ("Fold", "SF", "Family"),
}
for metric, task in TASKS:
    EXPECTED_SPLITS.setdefault((metric, task), ("Fold", "SF"))


@dataclass(frozen=True)
class Variant:
    slug: str
    title: str
    blocks: tuple[int, ...]
    summary_runs: dict[int, tuple[str, ...]]
    bindshake_runs: dict[int, tuple[str, ...]]
    homo_runs: dict[int, tuple[str, ...]]
    final_runs: tuple[str, ...]


def runs_by_block(prefix: str, indices: tuple[int, ...], blocks: tuple[int, ...]):
    return {
        block: (f"{prefix}{index}-crop-{('protein' if index < 8 else 'residue')}-norot"
                f"{'-relaunch' if index >= 8 else ''}-raw-step10k-block{block}-nonhomo",)
        for index, block in zip(indices, blocks)
    }


BLOCKS_CROP = (4, 6, 8, 12)

VARIANTS = (
    Variant(
        slug="jepa_rand70_raw100k",
        title="JEPA-rand70 (raw, 100k)",
        blocks=(1, 6, 12),
        summary_runs={
            1: (
                "slurm-258919-jepa-rand-mask-70-block1-panel",
                "slurm-259251-jepa-rand-mask-70-block1-panel",
                "slurm-259393-jepa-rand-mask-70-block1-panel",
            ),
            6: (
                "slurm-258920-jepa-rand-mask-70-block6-panel",
                "slurm-259252-jepa-rand-mask-70-block6-panel",
                "slurm-259394-jepa-rand-mask-70-block6-panel",
            ),
            12: (
                "slurm-258921-jepa-rand-mask-70-block12-panel",
                "slurm-259253-jepa-rand-mask-70-block12-panel",
                "slurm-259395-jepa-rand-mask-70-block12-panel",
            ),
        },
        bindshake_runs={
            1: ("slurm-258919-jepa-rand-mask-70-block1-panel",),
            6: ("slurm-258920-jepa-rand-mask-70-block6-panel",),
            12: ("slurm-258921-jepa-rand-mask-70-block12-panel",),
        },
        homo_runs={
            1: ("slurm-259393-jepa-rand-mask-70-block1-panel",),
            6: ("slurm-259394-jepa-rand-mask-70-block6-panel",),
            12: ("slurm-259395-jepa-rand-mask-70-block12-panel",),
        },
        final_runs=("slurm-256820-jepa-rand-mask-70-100k",),
    ),
    Variant(
        slug="cropped_protein_norotation_raw10k",
        title="Cropped LeJEPA, per-protein, no rotations (raw, 10k)",
        blocks=BLOCKS_CROP,
        summary_runs={
            block: (f"slurm-261515_{i}-crop-protein-norot-raw-step10k-block{block}-nonhomo",)
            for i, block in enumerate(BLOCKS_CROP)
        },
        bindshake_runs={
            block: (f"slurm-261515_{i}-crop-protein-norot-raw-step10k-block{block}-nonhomo",)
            for i, block in enumerate(BLOCKS_CROP)
        },
        homo_runs={
            block: (f"slurm-261515_{i + 4}-crop-protein-norot-raw-step10k-block{block}-homo",)
            for i, block in enumerate(BLOCKS_CROP)
        },
        final_runs=("slurm-258285-canonical-lejepa-middle-lr4e4-raw-step10k-all",),
    ),
    Variant(
        slug="cropped_residue_norotation_raw10k",
        title="Cropped LeJEPA, per-residue, no rotations (raw, 10k; evaluation in progress)",
        blocks=BLOCKS_CROP,
        summary_runs={
            # Use only the active clean restart.  The aborted 261515 attempt
            # reached BindInt, but mixing repeated downstream fits introduces
            # small seed-level differences (for example 0.4829 vs 0.4828).
            block: (f"slurm-261547_{i + 8}-crop-residue-norot-relaunch-raw-step10k-block{block}-nonhomo",)
            for i, block in enumerate(BLOCKS_CROP)
        },
        bindshake_runs={
            block: (f"slurm-261547_{i + 8}-crop-residue-norot-relaunch-raw-step10k-block{block}-nonhomo",)
            for i, block in enumerate(BLOCKS_CROP)
        },
        homo_runs={
            block: (f"slurm-261547_{i + 12}-crop-residue-norot-relaunch-raw-step10k-block{block}-homo",)
            for i, block in enumerate(BLOCKS_CROP)
        },
        final_runs=("slurm-258319_1-canonical-lejepa-jarvis-per-residue-raw-step10k-full-stb",),
    ),
)


def read_runs(scores, block, run_names, reader):
    for run_name in run_names:
        if reader is common.read_summary:
            reader(scores, block, RUNS / run_name / "summary_results.csv")
        else:
            reader(scores, block, run_name)


def load_scores(variant: Variant):
    scores = {}
    for block in variant.blocks:
        read_runs(scores, block, variant.summary_runs[block], common.read_summary)
        read_runs(scores, block, variant.bindshake_runs[block], common.read_single_test_auroc)
        read_runs(scores, block, variant.homo_runs[block], common.read_homology)
    return scores


def task_result(run_names, task):
    matches = []
    for run_name in run_names:
        path = common.find_task_result(run_name, task)
        if path is not None:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one final {task} result, found {matches}")
    return matches[0]


def load_final_reference(variant: Variant):
    reference = {}
    for run_name in variant.final_runs:
        path = RUNS / run_name / "summary_results.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                metric = {"auroc": "AUROC", "spearmanr": "Spearman"}[row["metric"]]
                reference[(metric, row["task"], "Fold")] = float(row["fold_test"]) * 100
                reference[(metric, row["task"], "SF")] = float(row["superfamily_test"]) * 100

    bindshake = task_result(variant.final_runs, "BindShake")
    values = {
        match.group("metric"): float(match.group("value"))
        for match in map(common.RESULT_LINE.search, bindshake.read_text().splitlines())
        if match
    }
    reference[("AUROC", "BindShake", "Test")] = values["test_auroc"] * 100

    homo = task_result(variant.final_runs, "Homo")
    split_for_metric = {
        "test_fold_holdout_f1_score": "Fold",
        "test_superfamily_holdout_f1_score": "SF",
        "test_family_holdout_f1_score": "Family",
    }
    for line in homo.read_text().splitlines():
        match = common.RESULT_LINE.search(line)
        if match and match.group("metric") in split_for_metric:
            reference[("Macro-F1", "Homo", split_for_metric[match.group("metric")])] = (
                float(match.group("value")) * 100
            )
    return reference


def write_csv(variant, scores):
    out_dir = OUT_ROOT / variant.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant.slug}_layerwise_stb.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "task", "split", "block", "score", "score_x100", "source"))
        for key in sorted(scores, key=lambda k: (k[0], common.TASK_ORDER.index(k[1]), k[2], k[3])):
            metric, task, split, block = key
            score, source = scores[key]
            writer.writerow((metric, task, split, block, score, score * 100, source))
    return path


def draw_panel(ax, variant, scores, references, metric, task, show_ylabel):
    block_x = {block: i for i, block in enumerate(variant.blocks)}
    split_order = ("Fold", "SF", "Family", "Test")
    layer_values = []
    complete = True
    for split in EXPECTED_SPLITS[(metric, task)]:
        values = [
            scores.get((metric, task, split, block), (math.nan, None))[0] * 100
            for block in variant.blocks
        ]
        complete &= all(math.isfinite(value) for value in values)
        if any(math.isfinite(value) for value in values):
            linestyle, marker = common.SPLIT_STYLE[split]
            ax.plot(
                tuple(block_x.values()), values, color=common.TASK_COLORS[task],
                linestyle=linestyle, marker=marker, linewidth=1.9, markersize=4.5,
                markeredgewidth=0.5, markeredgecolor="white",
            )
            layer_values.extend(value for value in values if math.isfinite(value))

    ref_names = ("Final 8D",) if metric == "Macro-F1" else ("Final 8D", "SLAE-Hidden", "ESM3")
    ref_x = {name: len(variant.blocks) + 0.35 + 1.7 * i for i, name in enumerate(ref_names)}
    ref_values = []
    for name in ref_names:
        for split in EXPECTED_SPLITS[(metric, task)]:
            key = (metric, task, split)
            if key not in references[name]:
                continue
            value = references[name][key]
            _, marker = common.SPLIT_STYLE[split]
            ax.scatter(ref_x[name], value, color=common.REFERENCE_STYLES[name]["color"],
                       marker=marker, s=46, edgecolor="black", linewidth=0.65, zorder=6)
            ref_values.append(value)

    region_start = len(variant.blocks) - 0.32
    x_max = max(ref_x.values()) + 0.55
    ax.axvspan(region_start, x_max, color="#F2F2F2", zorder=0)
    ax.axvline(region_start, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(-0.40, x_max)
    ax.set_xticks((*block_x.values(), *ref_x.values()))
    labels = [str(block) for block in variant.blocks] + [
        {"Final 8D": "Final\n8D", "SLAE-Hidden": "SLAE", "ESM3": "ESM3"}[name]
        for name in ref_names
    ]
    ax.set_xticklabels(labels)

    finite = layer_values + ref_values
    if finite:
        pad = max(1.5, (max(finite) - min(finite)) * 0.10)
        ax.set_ylim(min(finite) - pad, max(finite) + pad)
    if not complete:
        ax.text(0.03, 0.04, "Layerwise evaluation in progress", transform=ax.transAxes,
                fontsize=8, color="#666666", ha="left", va="bottom")
    ax.set_title("Homology" if task == "Homo" else task, fontsize=12.5,
                 fontweight="demibold", color="black", pad=5)
    ylabel = {"AUROC": r"AUROC ($\times 100$)", "Spearman": r"Spearman's $\rho$ ($\times 100$)",
               "Macro-F1": r"Macro-F1 ($\times 100$)"}[metric]
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9.5)
    ax.set_xlabel("Encoder block", fontsize=8.8)
    midpoint = sum(block_x.values()) / len(block_x)
    ax.xaxis.set_label_coords((midpoint + 0.4) / (x_max + 0.4), -0.15)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=8.5)
    return complete


def save_figure(variant, scores, references):
    fig = plt.figure(figsize=(15.5, 17.5))
    grid = fig.add_gridspec(4, 3, left=0.065, right=0.985, top=0.905, bottom=0.075,
                           hspace=0.50, wspace=0.27)
    complete = True
    for index, (metric, task) in enumerate(TASKS):
        row, col = divmod(index, 3)
        complete &= draw_panel(fig.add_subplot(grid[row, col]), variant, scores, references,
                               metric, task, show_ylabel=col == 0 or metric == "Macro-F1")

    handles = [
        Line2D([0], [0], color="black", linestyle=common.SPLIT_STYLE[split][0],
               marker=common.SPLIT_STYLE[split][1], linewidth=1.7, markersize=4.5, label=split)
        for split in ("Fold", "SF", "Family", "Test")
    ]
    for name, style in common.REFERENCE_STYLES.items():
        handles.append(Line2D([0], [0], color=style["color"], linestyle="None", marker="o",
                              markeredgecolor="black", markeredgewidth=0.6, markersize=5.5, label=name))
    fig.suptitle(variant.title, y=0.985, fontsize=15, fontweight="demibold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.525, 0.958), ncol=7,
               frameon=False, fontsize=9.5, handlelength=3)
    fig.text(0.525, 0.018,
             "Final 8D is this model's native final representation; SLAE-Hidden and ESM3 are shown for comparison.",
             ha="center", va="bottom", fontsize=9.2, color="#4F493F")
    path = FIGURE_DIR / f"{variant.slug}_layerwise_stb_all_metrics.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path, complete


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })
    common.verify_published_references()
    for variant in VARIANTS:
        scores = load_scores(variant)
        references = {"Final 8D": load_final_reference(variant), **common.PUBLISHED_REFERENCES}
        csv_path = write_csv(variant, scores)
        figure_path, complete = save_figure(variant, scores, references)
        counts = {block: sum(key[3] == block for key in scores) for block in variant.blocks}
        print(f"{variant.slug}: scores={len(scores)} counts={counts} complete={complete}")
        print(f"  {csv_path}")
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
