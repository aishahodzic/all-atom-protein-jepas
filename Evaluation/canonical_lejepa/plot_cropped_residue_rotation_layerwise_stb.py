#!/usr/bin/env python3
"""Plot STB scores for cropped per-residue + rotation LeJEPA blocks.

The layout intentionally matches ``plot_layerwise_stb_all_blocks.py``.  The
four encoder readouts are the raw/online 768D states after blocks 4, 6, 8, and
12 of the step-10k model.  Its native final 8D representation and published
SLAE-Hidden/ESM3 scores are shown as reference columns.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

RUNTIME_DIR = Path.home() / "tmp" / "plot-cropped-residue-rotation-stb"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(RUNTIME_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_layerwise_stb_all_blocks as common


ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ROOT
    / "StructTokenBench"
    / "struct_token_bench_logs"
    / "upstream-l600-v1-full"
    / "runs"
)
OUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "cropped_residue_rotation_layerwise_stb"
)
FIGURE_DIR = ROOT / "iclr2027" / "figures"

BLOCKS = (4, 6, 8, 12)
BLOCK_X = dict(zip(BLOCKS, range(len(BLOCKS))))
REFERENCE_X = {"Final 8D": 4.35, "SLAE-Hidden": 6.05, "ESM3": 7.75}
REFERENCE_REGION_START = 3.68
REFERENCE_X_MAX = 8.28

NONHOMO_RUNS = {
    4: "slurm-261455_0-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block4-nonhomo",
    6: "slurm-261455_1-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block6-nonhomo",
    8: "slurm-261455_2-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block8-nonhomo",
    12: "slurm-261455_3-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block12-nonhomo",
}
HOMO_RUNS = {
    4: "slurm-261455_4-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block4-homo",
    6: "slurm-261455_5-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block6-homo",
    8: "slurm-261455_6-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block8-homo",
    12: "slurm-261455_7-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-block12-homo",
}
FINAL_8D_RUNS = (
    "slurm-258633-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-shortest-four",
    "slurm-258635-canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k-remaining-eight",
)


def load_scores():
    scores = {}
    for block in BLOCKS:
        run_name = NONHOMO_RUNS[block]
        common.read_summary(scores, block, RUNS / run_name / "summary_results.csv")
        common.read_single_test_auroc(scores, block, run_name)
        common.read_homology(scores, block, HOMO_RUNS[block])
    return scores


def find_final_result(task):
    matches = []
    for run_name in FINAL_8D_RUNS:
        result_dir = RUNS / run_name / "results"
        matches.extend(sorted(result_dir.glob(f"*_{task}_*.txt")))
    if len(matches) != 1:
        raise ValueError(f"Expected one native final-8D {task} result, found {matches}")
    return matches[0]


def load_final_8d_reference():
    reference = {}
    for run_name in FINAL_8D_RUNS:
        summary = RUNS / run_name / "summary_results.csv"
        with summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                metric = {"auroc": "AUROC", "spearmanr": "Spearman"}[row["metric"]]
                reference[(metric, row["task"], "Fold")] = float(row["fold_test"]) * 100
                reference[(metric, row["task"], "SF")] = (
                    float(row["superfamily_test"]) * 100
                )

    bindshake_values = {
        match.group("metric"): float(match.group("value"))
        for match in map(
            common.RESULT_LINE.search,
            find_final_result("BindShake").read_text().splitlines(),
        )
        if match
    }
    reference[("AUROC", "BindShake", "Test")] = (
        bindshake_values["test_auroc"] * 100
    )

    split_for_metric = {
        "test_fold_holdout_f1_score": "Fold",
        "test_family_holdout_f1_score": "Family",
        "test_superfamily_holdout_f1_score": "SF",
    }
    for line in find_final_result("Homo").read_text().splitlines():
        match = common.RESULT_LINE.search(line)
        if match and match.group("metric") in split_for_metric:
            split = split_for_metric[match.group("metric")]
            reference[("Macro-F1", "Homo", split)] = (
                float(match.group("value")) * 100
            )

    expected = {"AUROC": 15, "Spearman": 6, "Macro-F1": 3}
    for metric, count in expected.items():
        actual = sum(key[0] == metric for key in reference)
        if actual != count:
            raise ValueError(
                f"Expected {count} native final-8D {metric} scores, found {actual}"
            )
    return reference


def write_long_csv(scores):
    path = OUT_DIR / "cropped_residue_rotation_raw10k_layerwise_stb.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("metric", "task", "split", "block", "score", "score_x100", "source")
        )
        for key in sorted(
            scores,
            key=lambda item: (
                item[0],
                common.TASK_ORDER.index(item[1]),
                item[2],
                item[3],
            ),
        ):
            metric, task, split, block = key
            score, source = scores[key]
            writer.writerow(
                (metric, task, split, block, f"{score:.4f}", f"{score * 100:.2f}", source)
            )
    return path


def series_for_task(scores, metric, task):
    split_order = ("Fold", "SF", "Family", "Test")
    splits = sorted(
        {split for score_metric, score_task, split, _ in scores if score_metric == metric and score_task == task},
        key=split_order.index,
    )
    return [
        (
            split,
            [scores[(metric, task, split, block)][0] * 100 for block in BLOCKS],
        )
        for split in splits
    ]


def reference_names(metric):
    if metric == "Macro-F1":
        return ("Final 8D",)
    return ("Final 8D", "SLAE-Hidden", "ESM3")


def draw_references(ax, references, metric, task):
    values = []
    for model_name in reference_names(metric):
        color = common.REFERENCE_STYLES[model_name]["color"]
        for (ref_metric, ref_task, split), value in references[model_name].items():
            if ref_metric != metric or ref_task != task:
                continue
            _, marker = common.SPLIT_STYLE[split]
            ax.scatter(
                REFERENCE_X[model_name],
                value,
                color=color,
                marker=marker,
                s=46,
                edgecolor="black",
                linewidth=0.65,
                zorder=6,
            )
            values.append(value)
    return values


def draw_task(ax, scores, references, metric, task, show_ylabel):
    finite = []
    for split, values in series_for_task(scores, metric, task):
        linestyle, marker = common.SPLIT_STYLE[split]
        ax.plot(
            tuple(BLOCK_X.values()),
            values,
            color=common.TASK_COLORS[task],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.9,
            markersize=4.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        finite.extend(values)
    finite.extend(draw_references(ax, references, metric, task))

    names = reference_names(metric)
    x_max = (
        REFERENCE_X_MAX
        if len(names) > 1
        else REFERENCE_X["Final 8D"] + 0.55
    )
    ax.axvspan(REFERENCE_REGION_START, x_max, color="#F2F2F2", zorder=0)
    ax.axvline(
        REFERENCE_REGION_START,
        color="#999999",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )
    ax.set_xlim(-0.40, x_max)
    ticks = list(BLOCK_X.values()) + [REFERENCE_X[name] for name in names]
    labels = [str(block) for block in BLOCKS] + [
        {"Final 8D": "Final\n8D", "SLAE-Hidden": "SLAE", "ESM3": "ESM3"}[name]
        for name in names
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    pad = max(1.5, (max(finite) - min(finite)) * 0.10)
    ax.set_ylim(min(finite) - pad, max(finite) + pad)
    ax.set_title(
        "Homology" if task == "Homo" else task,
        fontsize=12.5,
        fontweight="demibold",
        color="black",
        pad=5,
    )
    ylabel = {
        "AUROC": r"AUROC ($\times 100$)",
        "Spearman": r"Spearman's $\rho$ ($\times 100$)",
        "Macro-F1": r"Macro-F1 ($\times 100$)",
    }[metric]
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9.5)
    ax.set_xlabel("Encoder block", fontsize=8.8)
    encoder_midpoint = sum(BLOCK_X.values()) / len(BLOCK_X)
    ax.xaxis.set_label_coords(
        (encoder_midpoint + 0.40) / (x_max + 0.40),
        -0.15,
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=8.5)


def save_figure(scores, references):
    fig = plt.figure(figsize=(15.5, 17.5))
    grid = fig.add_gridspec(
        4,
        3,
        left=0.065,
        right=0.985,
        top=0.925,
        bottom=0.075,
        hspace=0.50,
        wspace=0.27,
    )

    auroc_tasks = (
        "BindInt",
        "BindBio",
        "BindShake",
        "CatInt",
        "CatBio",
        "Con",
        "Rep",
        "Ept",
    )
    for index, task in enumerate(auroc_tasks):
        row, col = divmod(index, 3)
        draw_task(
            fig.add_subplot(grid[row, col]),
            scores,
            references,
            "AUROC",
            task,
            show_ylabel=col == 0,
        )

    draw_task(
        fig.add_subplot(grid[2, 2]),
        scores,
        references,
        "Macro-F1",
        "Homo",
        show_ylabel=True,
    )

    for index, task in enumerate(("FlexRMSF", "FlexBFactor", "FlexNEQ")):
        draw_task(
            fig.add_subplot(grid[3, index]),
            scores,
            references,
            "Spearman",
            task,
            show_ylabel=index == 0,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=common.SPLIT_STYLE[split][0],
            marker=common.SPLIT_STYLE[split][1],
            linewidth=1.7,
            markersize=4.5,
            label=split,
        )
        for split in ("Fold", "SF", "Family", "Test")
    ]
    for model_name, style in common.REFERENCE_STYLES.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="None",
                marker="o",
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=5.5,
                label=model_name,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.975),
        ncol=7,
        frameon=False,
        fontsize=9.5,
        handlelength=3,
    )
    fig.text(
        0.525,
        0.018,
        "Final 8D is this model's native shared encoder mean; SLAE-Hidden and "
        "ESM3 are shown for comparison.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#4F493F",
    )

    path = (
        FIGURE_DIR
        / "canonical_lejepa_cropped_residue_rotation_raw10k_layerwise_stb_all_metrics.png"
    )
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Computer Modern Roman",
                "DejaVu Serif",
            ],
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    common.verify_published_references()
    scores = load_scores()
    counts = defaultdict(int)
    for _, _, _, block in scores:
        counts[block] += 1
    if any(counts[block] != 24 for block in BLOCKS):
        raise ValueError(f"Incomplete layerwise results: {dict(counts)}")

    references = {
        "Final 8D": load_final_8d_reference(),
        **common.PUBLISHED_REFERENCES,
    }
    csv_path = write_long_csv(scores)
    figure_path = save_figure(scores, references)
    print(f"Wrote {len(scores)} scores to {csv_path}")
    print("Per-block score counts:", ", ".join(f"{b}={counts[b]}" for b in BLOCKS))
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
