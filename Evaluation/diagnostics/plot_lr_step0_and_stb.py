"""Plot the matched seed-42 random-init baseline against LR trajectories.

Intrinsic values use the fixed 5,224-protein, 15,000-residue cohort with
physical sample SHA256 6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8.
The random-init encoder was verified tensor-for-tensor against the canonical
Middle-LR seed-42 initialization. STB values come from locked upstream-L600
runs 257166 (random init), 258280 (High/Low 10k), and 258285 (Middle 10k).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Evaluation/diagnostics/figures/lr_step0_trajectory_and_stb.png"

STEPS = np.array([10_000, 16_667, 20_000, 30_000])
COLORS = {
    "High LR (1e-3)": "#D55E00",
    "Middle LR (4e-4)": "#0072B2",
    "Low LR (1e-4)": "#009E73",
}

INTRINSIC = {
    "Effective rank / 8 ↑": {
        "random": 5.2232536569,
        "High LR (1e-3)": [7.79070, 7.88270, 7.90724, 7.92748],
        "Middle LR (4e-4)": [7.85774, 7.90371, 7.86463, 7.90340],
        "Low LR (1e-4)": [7.90203, 7.85282, 7.84232, 7.83992],
        "ylim": (4.8, 8.1),
    },
    "AA purity ↑": {
        "random": 0.8209466667,
        "High LR (1e-3)": [0.07593, 0.07665, 0.07839, 0.07757],
        "Middle LR (4e-4)": [0.08533, 0.07770, 0.07811, 0.07597],
        "Low LR (1e-4)": [0.08776, 0.08265, 0.08135, 0.07998],
        "ylim": (0.0, 0.88),
    },
    "Secondary-structure purity ↑": {
        "random": 0.43486,
        "High LR (1e-3)": [0.52344, 0.52329, 0.51780, 0.51600],
        "Middle LR (4e-4)": [0.51074, 0.50305, 0.50239, 0.50449],
        "Low LR (1e-4)": [0.52435, 0.52091, 0.51561, 0.51539],
        "ylim": (0.41, 0.54),
    },
    "RSA Spearman ↑": {
        "random": 0.5130903939,
        "High LR (1e-3)": [0.16642, 0.15030, 0.17005, 0.14963],
        "Middle LR (4e-4)": [0.16535, 0.14479, 0.14701, 0.15882],
        "Low LR (1e-4)": [0.18732, 0.17867, 0.17544, 0.17111],
        "ylim": (0.10, 0.55),
    },
    "Chain-length shortcut ↓": {
        "random": 0.1923410246,
        "High LR (1e-3)": [0.31254, 0.37476, 0.39447, 0.41265],
        "Middle LR (4e-4)": [0.36204, 0.39502, 0.40081, 0.41889],
        "Low LR (1e-4)": [0.33009, 0.34311, 0.33963, 0.34165],
        "ylim": (0.15, 0.44),
    },
    "Normalized-position shortcut ↓": {
        "random": 0.0531022532,
        "High LR (1e-3)": [0.21279, 0.23826, 0.24637, 0.23144],
        "Middle LR (4e-4)": [0.20953, 0.21694, 0.21899, 0.22039],
        "Low LR (1e-4)": [0.16974, 0.18688, 0.19194, 0.17608],
        "ylim": (0.02, 0.27),
    },
}

STB_ROWS = [
    "BindShake\nAUROC",
    "CatBio fold\nAUROC",
    "CatBio superfamily\nAUROC",
    "FlexRMSF fold\nSpearman",
    "FlexRMSF superfamily\nSpearman",
    "Homo family\nAUROC",
    "Homo fold\nAUROC",
    "Homo superfamily\nAUROC",
]
STB_COLUMNS = ["Random init", "High 10k", "Middle 10k", "Low 10k"]
STB = np.array(
    [
        [0.6470, 0.5888, 0.5881, 0.5826],
        [0.6434, 0.5707, 0.5667, 0.5798],
        [0.7043, 0.6199, 0.6218, 0.6408],
        [0.2060, 0.1826, 0.1901, 0.1616],
        [0.1778, 0.2375, 0.2546, 0.2106],
        [0.0764, 0.0818, 0.0817, 0.0768],
        [0.1131, 0.1429, 0.1434, 0.1357],
        [0.1055, 0.1208, 0.1193, 0.1126],
    ],
    dtype=float,
)


def plot_intrinsic(ax: plt.Axes, title: str, data: dict[str, object]) -> None:
    baseline = float(data["random"])
    for label, color in COLORS.items():
        values = np.asarray(data[label], dtype=float)
        # The dashed segment emphasizes that 0->10k was not sampled.
        ax.plot(
            [0, STEPS[0]], [baseline, values[0]],
            color=color, linewidth=1.7, linestyle=(0, (3, 3)), alpha=0.75,
        )
        ax.plot(
            STEPS, values, color=color, marker="o", markersize=4.5,
            linewidth=2.1, label=label,
        )
    ax.scatter([0], [baseline], marker="*", s=115, color="#202020", zorder=5)
    ax.annotate(
        f"random\n{baseline:.3f}", (0, baseline), xytext=(7, 0),
        textcoords="offset points", ha="left", va="center", fontsize=7.8,
    )
    ax.set_title(title, fontsize=11, weight="semibold")
    ax.set_xlim(-1_500, 31_000)
    ax.set_ylim(*data["ylim"])
    ax.set_xticks([0, 10_000, 20_000, 30_000], ["0", "10k", "20k", "30k"])
    ax.grid(True, alpha=0.20, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_stb(ax: plt.Axes) -> None:
    row_min = STB.min(axis=1, keepdims=True)
    row_span = np.maximum(STB.max(axis=1, keepdims=True) - row_min, 1e-12)
    normalized = (STB - row_min) / row_span
    cmap = LinearSegmentedColormap.from_list(
        "muted_red_green", ["#F2C6C2", "#F7F4E9", "#9BD3AE"]
    )
    image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row in range(STB.shape[0]):
        best = int(np.argmax(STB[row]))
        for column in range(STB.shape[1]):
            ax.text(
                column, row, f"{STB[row, column]:.4f}",
                ha="center", va="center", fontsize=9.2,
                weight="bold" if column == best else "normal",
                color="#111111",
            )
        ax.add_patch(
            plt.Rectangle(
                (best - 0.49, row - 0.49), 0.98, 0.98,
                fill=False, edgecolor="#176B3A", linewidth=2.0,
            )
        )
    ax.set_xticks(range(len(STB_COLUMNS)), STB_COLUMNS, fontsize=10)
    ax.set_yticks(range(len(STB_ROWS)), STB_ROWS, fontsize=9)
    ax.tick_params(length=0)
    ax.set_title(
        "Matched STB at 10k — raw encoder (higher is better)",
        fontsize=12, weight="semibold", pad=12,
    )
    ax.set_xlabel(
        "Cell color is normalized within each task/split; labels are raw scores",
        fontsize=9, labelpad=10,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["row worst", "row best"])
    colorbar.ax.tick_params(labelsize=8)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14.2, 14.5), facecolor="white")
    grid = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.55], hspace=0.42, wspace=0.28)

    for index, (title, data) in enumerate(INTRINSIC.items()):
        plot_intrinsic(fig.add_subplot(grid[index // 3, index % 3]), title, data)

    handles = [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=2, label=label)
        for label, color in COLORS.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color="#202020", marker="*", linewidth=0,
                   markersize=10, label="Random init (seed 42)")
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.942),
               ncol=4, frameon=False, fontsize=9.5)

    plot_stb(fig.add_subplot(grid[2, :]))
    fig.suptitle(
        "Canonical LeJEPA learning-rate trajectories vs exact random initialization",
        fontsize=16, weight="bold", y=0.982,
    )
    fig.text(
        0.5, 0.952,
        "Intrinsic diagnostics: same 5,224 proteins and fixed 15,000 residues. "
        "Dashed 0→10k segments have no intermediate checkpoints.",
        ha="center", va="center", fontsize=9.5, color="#444444",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
