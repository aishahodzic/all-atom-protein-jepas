#!/usr/bin/env python3
"""Plot full-L600 STB performance across all canonical LeJEPA encoder blocks.

The figure uses raw/online weights from the 100k uncropped canonical LeJEPA
checkpoint and the full-context 768D hidden state after each transformer block.
Results are read directly from the frozen STB run outputs.  Missing results
remain NaN and are not joined across gaps. Published SLAE-Hidden and ESM3
results are shown as separate point columns after block 12; they are not
additional LeJEPA blocks.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ROOT
    / "StructTokenBench"
    / "struct_token_bench_logs"
    / "upstream-l600-v1-full"
    / "runs"
)
OUT_DIR = Path(__file__).resolve().parent / "results" / "layerwise_stb_all_blocks"
FIGURE_DIR = ROOT / "thesis-latex" / "figures" / "results" / "block-investigation"
APPENDIX_FIGURE_DIR = ROOT / "iclr2027" / "figures" / "appendix"

BLOCKS = tuple(range(1, 13))
# Place our native final 8D representation before the published references.
REFERENCE_X = {"Final 8D": 13.5, "SLAE-Hidden": 15.5, "ESM3": 17.75}
REFERENCE_REGION_START = 12.72
REFERENCE_X_MAX = 18.30
REFERENCE_STYLES = {
    "Final 8D": {"color": "#D55E00"},
    "SLAE-Hidden": {"color": "#777777"},
    "ESM3": {"color": "#0072B2"},
}

FINAL_8D_RUN = RUNS / "slurm-258768_0-canonical-lejepa-v1-raw-step100k-all"

# Published StructTokenBench results for SLAE's 1024D final decoder-layer
# representation (Table 8 of https://openreview.net/pdf?id=EgPqKZHfVY).
# Values are already in percent units. SLAE did not report STB Homo results;
# its separate SCOPe fold-classification experiment uses a different protocol
# and accuracy rather than STB's macro-F1, so it is intentionally excluded.
SLAE_HIDDEN = {
    ("AUROC", "BindInt", "Fold"): 51.13,
    ("AUROC", "BindInt", "SF"): 95.78,
    ("AUROC", "BindBio", "Fold"): 77.98,
    ("AUROC", "BindBio", "SF"): 79.33,
    ("AUROC", "BindShake", "Test"): 81.83,
    ("AUROC", "CatInt", "Fold"): 75.50,
    ("AUROC", "CatInt", "SF"): 96.68,
    ("AUROC", "CatBio", "Fold"): 81.52,
    ("AUROC", "CatBio", "SF"): 93.76,
    ("AUROC", "Con", "Fold"): 67.21,
    ("AUROC", "Con", "SF"): 93.73,
    ("AUROC", "Rep", "Fold"): 72.89,
    ("AUROC", "Rep", "SF"): 85.29,
    ("AUROC", "Ept", "Fold"): 63.59,
    ("AUROC", "Ept", "SF"): 76.92,
    ("Spearman", "FlexRMSF", "Fold"): 68.24,
    ("Spearman", "FlexRMSF", "SF"): 69.81,
    ("Spearman", "FlexBFactor", "Fold"): 38.60,
    ("Spearman", "FlexBFactor", "SF"): 35.56,
    ("Spearman", "FlexNEQ", "Fold"): 65.56,
    ("Spearman", "FlexNEQ", "SF"): 64.51,
}

# ESM3 structure-tokenizer results from the same published Table 8.
ESM3 = {
    ("AUROC", "BindInt", "Fold"): 44.30,
    ("AUROC", "BindInt", "SF"): 90.77,
    ("AUROC", "BindBio", "Fold"): 62.84,
    ("AUROC", "BindBio", "SF"): 65.22,
    ("AUROC", "BindShake", "Test"): 66.10,
    ("AUROC", "CatInt", "Fold"): 61.09,
    ("AUROC", "CatInt", "SF"): 89.82,
    ("AUROC", "CatBio", "Fold"): 65.33,
    ("AUROC", "CatBio", "SF"): 74.65,
    ("AUROC", "Con", "Fold"): 55.22,
    ("AUROC", "Con", "SF"): 80.53,
    ("AUROC", "Rep", "Fold"): 74.70,
    ("AUROC", "Rep", "SF"): 82.36,
    ("AUROC", "Ept", "Fold"): 63.69,
    ("AUROC", "Ept", "SF"): 61.97,
    ("Spearman", "FlexRMSF", "Fold"): 44.53,
    ("Spearman", "FlexRMSF", "SF"): 39.68,
    ("Spearman", "FlexBFactor", "Fold"): 23.60,
    ("Spearman", "FlexBFactor", "SF"): 25.80,
    ("Spearman", "FlexNEQ", "Fold"): 45.08,
    ("Spearman", "FlexNEQ", "SF"): 45.43,
}

PUBLISHED_REFERENCES = {
    "SLAE-Hidden": SLAE_HIDDEN,
    "ESM3": ESM3,
}

# Blocks 1, 6, and 12 were evaluated in the earlier publication runs.  The
# remaining blocks use the matched September full-L600 runs.
SUMMARY_RUNS = {
    1: (
        "slurm-258922_1-canonical-lejepa-v1-raw-step100k-block1-all",
        "slurm-259033_1-canonical-lejepa-v1-raw-step100k-block1-gapfill",
    ),
    2: ("slurm-261382_2-canonical-lejepa-v1-raw-step100k-block2-nonhomo",),
    3: ("slurm-261383_3-canonical-lejepa-v1-raw-step100k-block3-nonhomo",),
    4: ("slurm-261383_4-canonical-lejepa-v1-raw-step100k-block4-nonhomo",),
    5: ("slurm-261383_5-canonical-lejepa-v1-raw-step100k-block5-nonhomo",),
    6: (
        "slurm-258922_6-canonical-lejepa-v1-raw-step100k-block6-all",
        "slurm-258990-canonical-lejepa-v1-raw-step100k-block6-fast",
        "slurm-258991-canonical-lejepa-v1-raw-step100k-block6-slow",
        "slurm-259033_6-canonical-lejepa-v1-raw-step100k-block6-gapfill",
    ),
    7: ("slurm-261383_7-canonical-lejepa-v1-raw-step100k-block7-nonhomo",),
    8: ("slurm-261383_8-canonical-lejepa-v1-raw-step100k-block8-nonhomo",),
    9: ("slurm-261383_9-canonical-lejepa-v1-raw-step100k-block9-nonhomo",),
    10: ("slurm-261383_10-canonical-lejepa-v1-raw-step100k-block10-nonhomo",),
    11: ("slurm-261383_11-canonical-lejepa-v1-raw-step100k-block11-nonhomo",),
    12: (
        "slurm-258992-canonical-lejepa-v1-raw-step100k-block12-fast",
        "slurm-258993-canonical-lejepa-v1-raw-step100k-block12-slow",
        "slurm-259033_12-canonical-lejepa-v1-raw-step100k-block12-gapfill",
    ),
}

BINDSHAKE_RUN = {
    1: "slurm-258922_1-canonical-lejepa-v1-raw-step100k-block1-all",
    2: "slurm-261382_2-canonical-lejepa-v1-raw-step100k-block2-nonhomo",
    3: "slurm-261383_3-canonical-lejepa-v1-raw-step100k-block3-nonhomo",
    4: "slurm-261383_4-canonical-lejepa-v1-raw-step100k-block4-nonhomo",
    5: "slurm-261383_5-canonical-lejepa-v1-raw-step100k-block5-nonhomo",
    6: "slurm-258990-canonical-lejepa-v1-raw-step100k-block6-fast",
    7: "slurm-261383_7-canonical-lejepa-v1-raw-step100k-block7-nonhomo",
    8: "slurm-261383_8-canonical-lejepa-v1-raw-step100k-block8-nonhomo",
    9: "slurm-261383_9-canonical-lejepa-v1-raw-step100k-block9-nonhomo",
    10: "slurm-261383_10-canonical-lejepa-v1-raw-step100k-block10-nonhomo",
    11: "slurm-261383_11-canonical-lejepa-v1-raw-step100k-block11-nonhomo",
    12: "slurm-258992-canonical-lejepa-v1-raw-step100k-block12-fast",
}

HOMO_RUN = {
    1: "slurm-259039_1-canonical-lejepa-v1-raw-step100k-block1-homo",
    2: "slurm-261416_2-canonical-lejepa-v1-raw-step100k-block2-homo32",
    3: "slurm-261416_3-canonical-lejepa-v1-raw-step100k-block3-homo32",
    4: "slurm-261416_4-canonical-lejepa-v1-raw-step100k-block4-homo32",
    5: "slurm-261416_5-canonical-lejepa-v1-raw-step100k-block5-homo32",
    6: "slurm-259039_6-canonical-lejepa-v1-raw-step100k-block6-homo",
    7: "slurm-261416_7-canonical-lejepa-v1-raw-step100k-block7-homo32",
    8: "slurm-261416_8-canonical-lejepa-v1-raw-step100k-block8-homo32",
    9: "slurm-261416_9-canonical-lejepa-v1-raw-step100k-block9-homo32",
    10: "slurm-261416_10-canonical-lejepa-v1-raw-step100k-block10-homo32",
    11: "slurm-261416_11-canonical-lejepa-v1-raw-step100k-block11-homo32",
    12: "slurm-259039_12-canonical-lejepa-v1-raw-step100k-block12-homo",
}

TASK_ORDER = (
    "BindInt",
    "BindBio",
    "BindShake",
    "CatInt",
    "CatBio",
    "Con",
    "Rep",
    "Ept",
    "FlexRMSF",
    "FlexBFactor",
    "FlexNEQ",
    "Homo",
)
TASK_COLORS = {
    "BindInt": "#8F5DA2",
    "BindBio": "#8F5DA2",
    "BindShake": "#8F5DA2",
    "CatInt": "#8F5DA2",
    "CatBio": "#8F5DA2",
    "Con": "#8F5DA2",
    "Rep": "#8F5DA2",
    "Ept": "#8F5DA2",
    "FlexRMSF": "#009E73",
    "FlexBFactor": "#009E73",
    "FlexNEQ": "#009E73",
    "Homo": "#6F4C9B",
}
SPLIT_STYLE = {
    "Fold": ("-", "o"),
    "SF": ("--", "s"),
    "Family": (":", "^"),
    "Test": ("-.", "D"),
}

RESULT_LINE = re.compile(
    r"│\s*(?P<metric>test_(?:family|fold|superfamily)_holdout_f1_score|test_auroc)"
    r"\s*│\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*│"
)


def add_score(scores, key, value, source):
    """Add one score and reject conflicting duplicate source records."""
    if key in scores and not math.isclose(scores[key][0], value, abs_tol=1e-8):
        old_value, old_source = scores[key]
        raise ValueError(
            f"Conflicting result for {key}: {old_value} ({old_source}) versus "
            f"{value} ({source})"
        )
    scores[key] = (value, source)


def read_summary(scores, block, path):
    if not path.exists():
        return
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            metric = {"auroc": "AUROC", "spearmanr": "Spearman"}[row["metric"]]
            add_score(
                scores,
                (metric, row["task"], "Fold", block),
                float(row["fold_test"]),
                path,
            )
            add_score(
                scores,
                (metric, row["task"], "SF", block),
                float(row["superfamily_test"]),
                path,
            )


def find_task_result(run_name, task):
    result_dir = RUNS / run_name / "results"
    matches = sorted(result_dir.glob(f"*_{task}_*.txt")) if result_dir.exists() else []
    if len(matches) > 1:
        raise ValueError(f"Multiple {task} result files in {result_dir}: {matches}")
    return matches[0] if matches else None


def read_single_test_auroc(scores, block, run_name):
    path = find_task_result(run_name, "BindShake")
    if path is None:
        return
    matches = [m for m in map(RESULT_LINE.search, path.read_text().splitlines()) if m]
    values = {m.group("metric"): float(m.group("value")) for m in matches}
    if "test_auroc" in values:
        add_score(scores, ("AUROC", "BindShake", "Test", block), values["test_auroc"], path)


def read_homology(scores, block, run_name):
    path = find_task_result(run_name, "Homo")
    if path is None:
        return
    split_for_metric = {
        "test_fold_holdout_f1_score": "Fold",
        "test_family_holdout_f1_score": "Family",
        "test_superfamily_holdout_f1_score": "SF",
    }
    for line in path.read_text().splitlines():
        match = RESULT_LINE.search(line)
        if match and match.group("metric") in split_for_metric:
            split = split_for_metric[match.group("metric")]
            add_score(scores, ("Macro-F1", "Homo", split, block), float(match.group("value")), path)


def load_scores():
    scores = {}
    for block in BLOCKS:
        for run_name in SUMMARY_RUNS[block]:
            read_summary(scores, block, RUNS / run_name / "summary_results.csv")
        read_single_test_auroc(scores, block, BINDSHAKE_RUN[block])
        read_homology(scores, block, HOMO_RUN[block])
    return scores


def load_final_8d_reference():
    """Read the native shared final-mean 8D representation from its STB run."""
    reference = {}
    summary = FINAL_8D_RUN / "summary_results.csv"
    with summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            metric = {"auroc": "AUROC", "spearmanr": "Spearman"}[row["metric"]]
            reference[(metric, row["task"], "Fold")] = float(row["fold_test"]) * 100
            reference[(metric, row["task"], "SF")] = float(row["superfamily_test"]) * 100

    bindshake = find_task_result(FINAL_8D_RUN.name, "BindShake")
    homo = find_task_result(FINAL_8D_RUN.name, "Homo")
    if bindshake is None or homo is None:
        raise FileNotFoundError("Native final 8D BindShake or Homo result is missing")

    bindshake_values = {
        match.group("metric"): float(match.group("value"))
        for match in map(RESULT_LINE.search, bindshake.read_text().splitlines())
        if match
    }
    reference[("AUROC", "BindShake", "Test")] = bindshake_values["test_auroc"] * 100

    split_for_metric = {
        "test_fold_holdout_f1_score": "Fold",
        "test_family_holdout_f1_score": "Family",
        "test_superfamily_holdout_f1_score": "SF",
    }
    for line in homo.read_text().splitlines():
        match = RESULT_LINE.search(line)
        if match and match.group("metric") in split_for_metric:
            split = split_for_metric[match.group("metric")]
            reference[("Macro-F1", "Homo", split)] = float(match.group("value")) * 100

    expected = {"AUROC": 15, "Spearman": 6, "Macro-F1": 3}
    for metric, count in expected.items():
        actual = sum(key[0] == metric for key in reference)
        if actual != count:
            raise ValueError(f"Expected {count} native final 8D {metric} scores, found {actual}")
    return reference


def load_references():
    return {"Final 8D": load_final_8d_reference(), **PUBLISHED_REFERENCES}


def write_long_csv(scores):
    path = OUT_DIR / "canonical_lejepa_raw100k_layerwise_stb.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "task", "split", "block", "score", "score_x100", "source"))
        for key in sorted(scores, key=lambda k: (k[0], TASK_ORDER.index(k[1]), k[2], k[3])):
            metric, task, split, block = key
            score, source = scores[key]
            writer.writerow((metric, task, split, block, f"{score:.4f}", f"{score * 100:.2f}", source))
    return path


def series_for_metric(scores, metric):
    keys = sorted(
        {(task, split) for m, task, split, _ in scores if m == metric},
        key=lambda item: (TASK_ORDER.index(item[0]), ("Fold", "Family", "SF", "Test").index(item[1])),
    )
    return [
        (
            task,
            split,
            [scores.get((metric, task, split, block), (math.nan, None))[0] * 100 for block in BLOCKS],
        )
        for task, split in keys
    ]


def block_task_complete(scores, metric, task, block):
    """Return whether every expected split for one task is present at a block."""
    expected_splits = {
        split
        for score_metric, score_task, split, _ in scores
        if score_metric == metric and score_task == task
    }
    return bool(expected_splits) and all(
        (metric, task, split, block) in scores for split in expected_splits
    )


def mark_incomplete_block(ax, block11_complete, label=True):
    if block11_complete:
        return
    ax.axvspan(10.72, 11.28, color="#EFEFEF", zorder=0)
    if label:
        ax.text(
            11,
            0.985,
            "in progress",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=90,
            fontsize=8,
            color="#666666",
        )


def draw_reference_points(ax, references, metric, task=None, model_names=None):
    """Draw matching native-final and published scores as point columns."""
    values = []
    if model_names is None:
        model_names = tuple(references)
    for model_name in model_names:
        model_values = references[model_name]
        style = REFERENCE_STYLES[model_name]
        for (ref_metric, ref_task, split), value in model_values.items():
            if ref_metric != metric or (task is not None and ref_task != task):
                continue
            _, marker = SPLIT_STYLE[split]
            ax.scatter(
                REFERENCE_X[model_name],
                value,
                color=style["color"],
                marker=marker,
                s=46,
                edgecolor="black",
                linewidth=0.65,
                zorder=6,
            )
            values.append(value)
    return values


def mark_reference_region(ax, x_max=REFERENCE_X_MAX):
    """Separate native-final and published point columns from encoder blocks."""
    ax.axvspan(REFERENCE_REGION_START, x_max, color="#F2F2F2", zorder=0)
    ax.axvline(
        REFERENCE_REGION_START,
        color="#999999",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )


def verify_published_references():
    """Catch transcription errors against the averages printed in Table 8."""
    published_averages = {
        "SLAE-Hidden": {"AUROC": 79.54, "Spearman": 57.05},
        "ESM3": {"AUROC": 69.24, "Spearman": 37.35},
    }
    for model_name, model_values in PUBLISHED_REFERENCES.items():
        for metric, published_average in published_averages[model_name].items():
            values = [
                value
                for (ref_metric, _, _), value in model_values.items()
                if ref_metric == metric
            ]
            calculated_average = sum(values) / len(values)
            if not math.isclose(calculated_average, published_average, abs_tol=0.005):
                raise ValueError(
                    f"{model_name} values do not reproduce the published average: "
                    f"{calculated_average:.4f} versus {published_average:.2f}"
                )


def style_axes(ax, metric, block11_complete, reference_names=()):
    ylabels = {
        "AUROC": r"AUROC ($\times 100$)",
        "Spearman": r"Spearman's $\rho$ ($\times 100$)",
        "Macro-F1": r"Macro-F1 ($\times 100$)",
    }
    titles = {
        "AUROC": "Functional-site prediction",
        "Spearman": "Structural flexibility",
        "Macro-F1": "Remote homology",
    }
    ax.set_title(titles[metric], fontsize=15, fontweight="demibold", color="black", pad=8)
    ax.set_ylabel(ylabels[metric], fontsize=12, color="black")
    if reference_names:
        x_max = max(REFERENCE_X[name] for name in reference_names) + 0.55
        labels = {
            "Final 8D": "Final\n8D",
            "SLAE-Hidden": "SLAE",
            "ESM3": "ESM3",
        }
        ax.set_xlim(0.7, x_max)
        ax.set_xticks(
            (2, 4, 6, 8, 10, 12, *(REFERENCE_X[name] for name in reference_names))
        )
        ax.set_xticklabels(
            ("2", "4", "6", "8", "10", "12", *(labels[name] for name in reference_names))
        )
        mark_reference_region(ax, x_max)
    else:
        ax.set_xlim(0.7, 12.3)
        ax.set_xticks((2, 4, 6, 8, 10, 12))
        ax.set_xticklabels(("2", "4", "6", "8", "10", "12"))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=10)
    mark_incomplete_block(ax, block11_complete)


def draw_metric(ax, scores, references, metric, block11_complete):
    series = series_for_metric(scores, metric)
    for task, split, values in series:
        linestyle, marker = SPLIT_STYLE[split]
        ax.plot(
            BLOCKS,
            values,
            label=f"{task} · {split}",
            color=TASK_COLORS[task],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.65,
            markersize=4.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
    reference_names = (
        ("Final 8D",)
        if metric == "Macro-F1"
        else ("Final 8D", "SLAE-Hidden", "ESM3")
    )
    style_axes(ax, metric, block11_complete, reference_names)
    reference_values = draw_reference_points(
        ax,
        references,
        metric,
        model_names=reference_names,
    )
    finite = [value for _, _, values in series for value in values if math.isfinite(value)]
    finite.extend(reference_values)
    pad = max(2.5, (max(finite) - min(finite)) * 0.08)
    ax.set_ylim(min(finite) - pad, max(finite) + pad)
    legend_columns = 3 if metric == "AUROC" else 4 if metric == "Spearman" else 1
    handles, labels = ax.get_legend_handles_labels()
    if reference_names:
        for model_name in reference_names:
            style = REFERENCE_STYLES[model_name]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle="None",
                    marker="o",
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    markersize=5.5,
                )
            )
            labels.append(model_name)
    legend_position = (
        {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)}
        if metric == "Macro-F1"
        else {"loc": "upper center", "bbox_to_anchor": (0.5, -0.17)}
    )
    ax.legend(
        handles,
        labels,
        **legend_position,
        ncol=legend_columns,
        frameon=False,
        fontsize=8.2,
        handlelength=2.8,
        columnspacing=1.4,
    )


def draw_auroc_task(
    ax, scores, references, task, block11_complete, show_ylabel, show_xlabel
):
    series = [item for item in series_for_metric(scores, "AUROC") if item[0] == task]
    for _, split, values in series:
        linestyle, marker = SPLIT_STYLE[split]
        ax.plot(
            BLOCKS,
            values,
            color=TASK_COLORS[task],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.9,
            markersize=4.2,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
    reference_values = draw_reference_points(ax, references, "AUROC", task)
    finite = [value for _, _, values in series for value in values if math.isfinite(value)]
    finite.extend(reference_values)
    pad = max(1.3, (max(finite) - min(finite)) * 0.12)
    ax.set_ylim(min(finite) - pad, max(finite) + pad)
    ax.set_xlim(0.7, REFERENCE_X_MAX)
    ax.set_xticks((2, 4, 6, 8, 10, 12, *REFERENCE_X.values()))
    ax.set_xticklabels(
        ("2", "4", "6", "8", "10", "12", "Final\n8D", "SLAE", "ESM3")
    )
    ax.set_title(task, fontsize=12.5, fontweight="demibold", color="black", pad=5)
    ax.set_ylabel(r"AUROC ($\times 100$)" if show_ylabel else "", fontsize=9.5)
    ax.set_xlabel(
        "Encoder block" if show_xlabel else "",
        fontsize=8.8,
    )
    ax.xaxis.set_label_coords((6.5 - 0.7) / (REFERENCE_X_MAX - 0.7), -0.15)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=8.5)
    mark_reference_region(ax)
    mark_incomplete_block(ax, block11_complete, label=False)


def draw_flexibility_task(ax, scores, references, task, block11_complete, show_ylabel):
    """Draw one flexibility benchmark so all three tasks can share one color."""
    series = [item for item in series_for_metric(scores, "Spearman") if item[0] == task]
    for _, split, values in series:
        linestyle, marker = SPLIT_STYLE[split]
        ax.plot(
            BLOCKS,
            values,
            color=TASK_COLORS[task],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.9,
            markersize=4.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
    reference_values = draw_reference_points(ax, references, "Spearman", task)
    finite = [value for _, _, values in series for value in values if math.isfinite(value)]
    finite.extend(reference_values)
    pad = max(1.8, (max(finite) - min(finite)) * 0.10)
    ax.set_ylim(min(finite) - pad, max(finite) + pad)
    ax.set_xlim(0.7, REFERENCE_X_MAX)
    ax.set_xticks((2, 4, 6, 8, 10, 12, *REFERENCE_X.values()))
    ax.set_xticklabels(
        ("2", "4", "6", "8", "10", "12", "Final\n8D", "SLAE", "ESM3")
    )
    ax.set_title(task, fontsize=13, fontweight="demibold", color="black", pad=5)
    ax.set_ylabel(
        r"Spearman's $\rho$ ($\times 100$)" if show_ylabel else "",
        fontsize=10.5,
    )
    ax.set_xlabel("Encoder block", fontsize=9.5)
    ax.xaxis.set_label_coords((6.5 - 0.7) / (REFERENCE_X_MAX - 0.7), -0.15)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=9)
    mark_reference_region(ax)
    mark_incomplete_block(ax, block11_complete, label=False)


def draw_homology_task(ax, scores, references, block11_complete):
    """Draw remote homology as one task panel in the common 4-by-3 grid."""
    series = [item for item in series_for_metric(scores, "Macro-F1") if item[0] == "Homo"]
    for _, split, values in series:
        linestyle, marker = SPLIT_STYLE[split]
        ax.plot(
            BLOCKS,
            values,
            color=TASK_COLORS["Homo"],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.9,
            markersize=4.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
            label=split,
        )

    reference_values = draw_reference_points(
        ax, references, "Macro-F1", "Homo", model_names=("Final 8D",)
    )
    finite = [value for _, _, values in series for value in values if math.isfinite(value)]
    finite.extend(reference_values)
    pad = max(1.8, (max(finite) - min(finite)) * 0.10)
    ax.set_ylim(min(finite) - pad, max(finite) + pad)
    x_max = REFERENCE_X["Final 8D"] + 0.55
    ax.set_xlim(0.7, x_max)
    ax.set_xticks((2, 4, 6, 8, 10, 12, REFERENCE_X["Final 8D"]))
    ax.set_xticklabels(("2", "4", "6", "8", "10", "12", "Final\n8D"))
    ax.set_title("Homology", fontsize=13, fontweight="demibold", color="black", pad=5)
    ax.set_ylabel(r"Macro-F1 ($\times 100$)", fontsize=10.5)
    ax.set_xlabel("Encoder block", fontsize=9.5)
    ax.xaxis.set_label_coords((6.5 - 0.7) / (x_max - 0.7), -0.15)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=9)
    mark_reference_region(ax, x_max)
    mark_incomplete_block(ax, block11_complete, label=False)

def save_combined(scores, references, block11_complete):
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

    auroc_tasks = ("BindInt", "BindBio", "BindShake", "CatInt", "CatBio", "Con", "Rep", "Ept")
    for index, task in enumerate(auroc_tasks):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(grid[row, col])
        draw_auroc_task(
            ax,
            scores,
            references,
            task,
            block_task_complete(scores, "AUROC", task, 11),
            show_ylabel=col == 0,
            show_xlabel=True,
        )

    split_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=SPLIT_STYLE[split][0],
            marker=SPLIT_STYLE[split][1],
            linewidth=1.7,
            markersize=4.5,
            label=split,
        )
        for split in ("Fold", "SF", "Family", "Test")
    ]
    for model_name, style in REFERENCE_STYLES.items():
        split_handles.append(
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
        handles=split_handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.975),
        ncol=7,
        frameon=False,
        fontsize=9.5,
        handlelength=3,
    )
    if not block11_complete:
        fig.text(
            0.982,
            0.958,
            "Gray block-11 bands mark pending task results",
            ha="right",
            fontsize=8.5,
            color="#666666",
        )

    homology_ax = fig.add_subplot(grid[2, 2])
    draw_homology_task(
        homology_ax,
        scores,
        references,
        block_task_complete(scores, "Macro-F1", "Homo", 11),
    )

    for index, task in enumerate(("FlexRMSF", "FlexBFactor", "FlexNEQ")):
        grid_index = 9 + index
        row, col = divmod(grid_index, 3)
        ax = fig.add_subplot(grid[row, col])
        draw_flexibility_task(
            ax,
            scores,
            references,
            task,
            block_task_complete(scores, "Spearman", task, 11),
            show_ylabel=index == 0 or col == 0,
        )

    fig.text(
        0.525,
        0.018,
        "Final 8D is the native shared encoder mean; SLAE-Hidden and ESM3 are shown "
        "for comparison.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#4F493F",
    )
    path = FIGURE_DIR / "canonical_lejepa_raw100k_layerwise_stb_all_metrics.png"
    fig.savefig(path, dpi=300, facecolor="white")
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    appendix_path = (
        APPENDIX_FIGURE_DIR
        / "canonical_lejepa_raw100k_layerwise_stb_all_metrics.png"
    )
    fig.savefig(appendix_path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    verify_published_references()
    scores = load_scores()
    references = load_references()
    expected_per_block = 15 + 6 + 3
    counts = defaultdict(int)
    for _, _, _, block in scores:
        counts[block] += 1
    block11_complete = counts[11] == expected_per_block

    csv_path = write_long_csv(scores)
    figures = [save_combined(scores, references, block11_complete)]

    print(f"Wrote {len(scores)} scores to {csv_path}")
    print("Per-block score counts:", ", ".join(f"{b}={counts[b]}" for b in BLOCKS))
    for path in figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
