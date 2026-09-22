#!/usr/bin/env python3
"""Create dependency-free SVG plots for the canonical LeJEPA STB LR sweep."""

from __future__ import annotations

import csv
import math
from html import escape
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "results" / "stb_lr_trajectories"
STEPS = [0, 10_000, 16_667, 20_000, 30_000]
STEP_LABELS = ["0 (random)", "10k", "16.7k", "20k", "30k"]
LRS = ("Low LR", "Middle LR", "High LR")
COLORS = {"Low LR": "#269E74", "Middle LR": "#377EB8", "High LR": "#E24A33"}
LR_LABELS = {
    "Low LR": "Low LR (1e-4)",
    "Middle LR": "Middle LR (4e-4)",
    "High LR": "High LR (1e-3)",
}

DATA = {
    ("BindShake", "test", "AUROC"): {
        "random": 0.6470,
        "Low LR": [0.5826, 0.5833, math.nan, 0.5844],
        "Middle LR": [0.5881, 0.5957, 0.5890, 0.5859],
        "High LR": [0.5888, 0.5958, 0.5869, 0.5890],
    },
    ("CatBio", "fold", "AUROC"): {
        "random": 0.6434,
        "Low LR": [0.5798, 0.5677, math.nan, 0.5644],
        "Middle LR": [0.5667, 0.5711, 0.5690, 0.5695],
        "High LR": [0.5707, 0.5593, 0.5566, 0.5591],
    },
    ("CatBio", "superfamily", "AUROC"): {
        "random": 0.7043,
        "Low LR": [0.6408, 0.6393, math.nan, 0.6345],
        "Middle LR": [0.6218, 0.6181, 0.6121, 0.6144],
        "High LR": [0.6199, 0.6333, 0.6271, 0.6281],
    },
    ("FlexRMSF", "fold", "Spearman rho"): {
        "random": 0.2060,
        "Low LR": [0.1616, 0.1612, math.nan, 0.1703],
        "Middle LR": [0.1901, 0.1710, 0.1808, 0.1710],
        "High LR": [0.1826, 0.2062, 0.2154, 0.2035],
    },
    ("FlexRMSF", "superfamily", "Spearman rho"): {
        "random": 0.1778,
        "Low LR": [0.2106, 0.2212, math.nan, 0.2299],
        "Middle LR": [0.2546, 0.2327, 0.2412, 0.2220],
        "High LR": [0.2375, 0.2651, 0.2610, 0.2551],
    },
    ("Homo", "family", "AUROC"): {
        "random": 0.0764,
        "Low LR": [0.0768, 0.0800, math.nan, 0.0797],
        "Middle LR": [0.0817, 0.0813, 0.0814, 0.0809],
        "High LR": [0.0818, 0.0816, 0.0825, 0.0819],
    },
    ("Homo", "fold", "AUROC"): {
        "random": 0.1131,
        "Low LR": [0.1357, 0.1441, math.nan, 0.1458],
        "Middle LR": [0.1434, 0.1461, 0.1465, 0.1456],
        "High LR": [0.1429, 0.1415, 0.1408, 0.1418],
    },
    ("Homo", "superfamily", "AUROC"): {
        "random": 0.1055,
        "Low LR": [0.1126, 0.1173, math.nan, 0.1174],
        "Middle LR": [0.1193, 0.1197, 0.1202, 0.1195],
        "High LR": [0.1208, 0.1201, 0.1204, 0.1202],
    },
    ("BindInt", "fold", "AUROC"): {
        "random": 0.4364,
        "Low LR": [0.4937, 0.4712, math.nan, math.nan],
        "Middle LR": [0.4832, math.nan, 0.4710, math.nan],
        "High LR": [0.4584, 0.5029, 0.5030, 0.4695],
    },
    ("BindInt", "superfamily", "AUROC"): {
        "random": 0.8003,
        "Low LR": [0.7775, 0.8600, math.nan, math.nan],
        "Middle LR": [0.8659, math.nan, 0.8694, math.nan],
        "High LR": [0.7952, 0.8610, 0.8627, 0.7916],
    },
    ("BindBio", "fold", "AUROC"): {
        "random": 0.6108,
        "Low LR": [0.6066, 0.5961, math.nan, math.nan],
        "Middle LR": [0.5766, math.nan, 0.5754, math.nan],
        "High LR": [0.5889, 0.5953, 0.5909, math.nan],
    },
    ("BindBio", "superfamily", "AUROC"): {
        "random": 0.6276,
        "Low LR": [0.6184, 0.6080, math.nan, math.nan],
        "Middle LR": [0.5895, math.nan, 0.5929, math.nan],
        "High LR": [0.6032, 0.6057, 0.6015, math.nan],
    },
    ("CatInt", "fold", "AUROC"): {
        "random": 0.5401,
        "Low LR": [0.5538, 0.5532, math.nan, math.nan],
        "Middle LR": [0.5647, math.nan, 0.5379, math.nan],
        "High LR": [0.5770, 0.5763, 0.5531, math.nan],
    },
    ("CatInt", "superfamily", "AUROC"): {
        "random": 0.7761,
        "Low LR": [0.7979, 0.7659, math.nan, math.nan],
        "Middle LR": [0.8100, math.nan, 0.8074, math.nan],
        "High LR": [0.8081, 0.8119, 0.8133, math.nan],
    },
    ("Con", "fold", "AUROC"): {
        "random": 0.5776,
        "Low LR": [0.5481, 0.5391, math.nan, math.nan],
        "Middle LR": [0.5504, math.nan, math.nan, math.nan],
        "High LR": [0.5637, 0.5467, 0.5430, math.nan],
    },
    ("Con", "superfamily", "AUROC"): {
        "random": 0.7205,
        "Low LR": [0.7405, 0.7397, math.nan, math.nan],
        "Middle LR": [0.7363, math.nan, math.nan, math.nan],
        "High LR": [0.7419, 0.7415, 0.7447, math.nan],
    },
    ("Rep", "fold", "AUROC"): {
        "random": 0.7560,
        "Low LR": [0.7333, 0.7354, math.nan, math.nan],
        "Middle LR": [0.7430, math.nan, math.nan, math.nan],
        "High LR": [0.7467, 0.7432, 0.7431, math.nan],
    },
    ("Rep", "superfamily", "AUROC"): {
        "random": 0.7928,
        "Low LR": [0.8299, 0.8316, math.nan, math.nan],
        "Middle LR": [0.8189, math.nan, math.nan, math.nan],
        "High LR": [0.8353, 0.8404, 0.8376, math.nan],
    },
    ("Ept", "fold", "AUROC"): {
        "random": 0.5402,
        "Low LR": [0.4602, 0.4781, 0.4797, math.nan],
        "Middle LR": [0.4704, math.nan, math.nan, math.nan],
        "High LR": [0.4325, 0.4348, 0.4963, math.nan],
    },
    ("Ept", "superfamily", "AUROC"): {
        "random": 0.5532,
        "Low LR": [0.5423, 0.5898, 0.5862, math.nan],
        "Middle LR": [0.5822, math.nan, math.nan, math.nan],
        "High LR": [0.5598, 0.5645, 0.6230, math.nan],
    },
    ("FlexBFactor", "fold", "Spearman rho"): {
        "random": 0.1394,
        "Low LR": [0.0531, 0.0519, math.nan, math.nan],
        "Middle LR": [0.0973, math.nan, math.nan, math.nan],
        "High LR": [0.1069, 0.1127, 0.1229, math.nan],
    },
    ("FlexBFactor", "superfamily", "Spearman rho"): {
        "random": 0.1142,
        "Low LR": [0.0875, 0.0978, math.nan, math.nan],
        "Middle LR": [0.1018, math.nan, math.nan, math.nan],
        "High LR": [0.1287, 0.1407, 0.1283, math.nan],
    },
    ("FlexNEQ", "fold", "Spearman rho"): {
        "random": 0.0696,
        "Low LR": [0.1847, 0.1823, math.nan, math.nan],
        "Middle LR": [0.2000, math.nan, math.nan, math.nan],
        "High LR": [0.2423, 0.2146, 0.1941, math.nan],
    },
    ("FlexNEQ", "superfamily", "Spearman rho"): {
        "random": 0.0655,
        "Low LR": [0.1629, 0.1669, math.nan, math.nan],
        "Middle LR": [0.1916, math.nan, math.nan, math.nan],
        "High LR": [0.2333, 0.2156, 0.1974, math.nan],
    },
}

STB_ORDER = [
    key
    for task in ("BindInt", "BindBio", "BindShake", "CatInt", "CatBio", "Con", "Rep", "Ept", "FlexBFactor", "FlexNEQ", "FlexRMSF", "Homo")
    for key in DATA
    if key[0] == task
]

DIAGNOSTIC_STEPS = [0, 10_000, 16_667, 20_000, 30_000]
DIAGNOSTIC_STEP_LABELS = ["0 (random)", "10k", "16.7k", "20k", "30k"]
DIAGNOSTICS = {
    "Effective rank / 8 ↑": {
        "random": 5.2232536569,
        "High LR": [7.79070, 7.88270, 7.90724, 7.92748],
        "Middle LR": [7.85774, 7.90371, 7.86463, 7.90340],
        "Low LR": [7.90203, 7.85282, 7.84232, 7.83992],
    },
    "AA purity ↑": {
        "random": 0.8209466667,
        "High LR": [0.07593, 0.07665, 0.07839, 0.07757],
        "Middle LR": [0.08533, 0.07770, 0.07811, 0.07597],
        "Low LR": [0.08776, 0.08265, 0.08135, 0.07998],
    },
    "Secondary-structure purity ↑": {
        "random": 0.43486,
        "High LR": [0.52344, 0.52329, 0.51780, 0.51600],
        "Middle LR": [0.51074, 0.50305, 0.50239, 0.50449],
        "Low LR": [0.52435, 0.52091, 0.51561, 0.51539],
    },
    "RSA Spearman ↑": {
        "random": 0.5130903939,
        "High LR": [0.16642, 0.15030, 0.17005, 0.14963],
        "Middle LR": [0.16535, 0.14479, 0.14701, 0.15882],
        "Low LR": [0.18732, 0.17867, 0.17544, 0.17111],
    },
    "Chain-length shortcut ↓": {
        "random": 0.1923410246,
        "High LR": [0.31254, 0.37476, 0.39447, 0.41265],
        "Middle LR": [0.36204, 0.39502, 0.40081, 0.41889],
        "Low LR": [0.33009, 0.34311, 0.33963, 0.34165],
    },
    "Normalized-position shortcut ↓": {
        "random": 0.0531022532,
        "High LR": [0.21279, 0.23826, 0.24637, 0.23144],
        "Middle LR": [0.20953, 0.21694, 0.21899, 0.22039],
        "Low LR": [0.16974, 0.18688, 0.19194, 0.17608],
    },
}


def marker(x: float, y: float, color: str, kind: int) -> str:
    if kind == 0:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>'
    if kind == 1:
        return f'<rect x="{x - 4.3:.1f}" y="{y - 4.3:.1f}" width="8.6" height="8.6" fill="{color}"/>'
    points = f"{x:.1f},{y-5:.1f} {x-5:.1f},{y+4:.1f} {x+5:.1f},{y+4:.1f}"
    return f'<polygon points="{points}" fill="{color}"/>'


def panel(key: tuple[str, str, str], x0: float, y0: float, width: float, height: float) -> str:
    task, split, metric = key
    datum = DATA[key]
    all_values = [datum["random"]]
    for lr in LRS:
        all_values.extend(value for value in datum[lr] if not math.isnan(value))
    lo, hi = min(all_values), max(all_values)
    pad = max((hi - lo) * 0.16, 0.001)
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 58, 16, 39, 48
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom

    def sx(step: int) -> float:
        return px0 + (px1 - px0) * step / 30_000

    def sy(value: float) -> float:
        return py1 - (py1 - py0) * (value - lo) / (hi - lo)

    out = [f'<g aria-label="{escape(task)} {escape(split)}">']
    out.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="5" fill="#FFFFFF" stroke="#D7DCE2"/>')
    out.append(f'<text x="{x0 + width/2:.1f}" y="{y0 + 23}" text-anchor="middle" class="panel-title">{escape(task)} — {escape(split)}</text>')
    for i in range(5):
        value = lo + (hi - lo) * i / 4
        yy = sy(value)
        out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#E7E9EC"/>')
        out.append(f'<text x="{px0 - 8}" y="{yy + 3:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for step, label in zip(STEPS, STEP_LABELS):
        xx = sx(step)
        out.append(f'<text x="{xx:.1f}" y="{py1 + 20}" text-anchor="middle" class="tick">{label}</text>')
    random_y = sy(datum["random"])
    out.append(f'<line x1="{px0}" y1="{random_y:.1f}" x2="{px1}" y2="{random_y:.1f}" stroke="#555" stroke-dasharray="3 4"/>')
    out.append(f'<text transform="translate({x0 + 14:.1f},{y0 + height/2:.1f}) rotate(-90)" text-anchor="middle" class="axis-label">{escape(metric)}</text>')
    for lr_index, lr in enumerate(LRS):
        series = [datum["random"], *datum[lr]]
        points = [(sx(step), sy(value)) for step, value in zip(STEPS, series) if not math.isnan(value)]
        point_string = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        out.append(f'<polyline points="{point_string}" fill="none" stroke="{COLORS[lr]}" stroke-width="2.4" stroke-linejoin="round"/>')
        for xx, yy in points[1:]:
            out.append(marker(xx, yy, COLORS[lr], lr_index))
    out.append(f'<circle cx="{sx(0):.1f}" cy="{random_y:.1f}" r="6" fill="#222" stroke="#FFF" stroke-width="1.5"/>')
    if any(math.isnan(value) for lr in LRS for value in datum[lr]):
        out.append(f'<text x="{px1:.1f}" y="{py0 + 10:.1f}" text-anchor="end" class="pending">partial</text>')
    out.append("</g>")
    return "".join(out)


def diagnostic_panel(title: str, x0: float, y0: float, width: float, height: float) -> str:
    datum = DIAGNOSTICS[title]
    values = [datum["random"]]
    for lr in LRS:
        values.extend(datum[lr])
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.12, 0.001)
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 62, 16, 39, 48
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom

    def sx(step: int) -> float:
        return px0 + (px1 - px0) * step / 30_000

    def sy(value: float) -> float:
        return py1 - (py1 - py0) * (value - lo) / (hi - lo)

    out = [f'<g aria-label="{escape(title)}">']
    out.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="5" fill="#FFFFFF" stroke="#D7DCE2"/>')
    out.append(f'<text x="{x0 + width/2:.1f}" y="{y0 + 23}" text-anchor="middle" class="panel-title">{escape(title)}</text>')
    for i in range(5):
        value = lo + (hi - lo) * i / 4
        yy = sy(value)
        out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#E7E9EC"/>')
        out.append(f'<text x="{px0 - 8}" y="{yy + 3:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for step, label in zip(DIAGNOSTIC_STEPS, DIAGNOSTIC_STEP_LABELS):
        out.append(f'<text x="{sx(step):.1f}" y="{py1 + 20}" text-anchor="middle" class="tick">{label}</text>')
    random_y = sy(datum["random"])
    out.append(f'<line x1="{px0}" y1="{random_y:.1f}" x2="{px1}" y2="{random_y:.1f}" stroke="#555" stroke-dasharray="3 4"/>')
    for lr_index, lr in enumerate(LRS):
        series = [datum["random"], *datum[lr]]
        points = [(sx(step), sy(value)) for step, value in zip(DIAGNOSTIC_STEPS, series)]
        point_string = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        out.append(f'<polyline points="{point_string}" fill="none" stroke="{COLORS[lr]}" stroke-width="2.4" stroke-linejoin="round"/>')
        for xx, yy in points[1:]:
            out.append(marker(xx, yy, COLORS[lr], lr_index))
    out.append(f'<circle cx="{sx(0):.1f}" cy="{random_y:.1f}" r="6" fill="#222" stroke="#FFF" stroke-width="1.5"/>')
    out.append("</g>")
    return "".join(out)


def save_combined_figure() -> None:
    gap = 18
    width = 1650
    header = 105
    section_h = 34
    stb_panel_w, stb_panel_h = 390, 280
    diagnostic_panel_w, diagnostic_panel_h = stb_panel_w, stb_panel_h
    stb_y = header + section_h
    stb_rows = math.ceil(len(STB_ORDER) / 4)
    diagnostic_heading_y = stb_y + stb_rows * (stb_panel_h + gap) + 8
    diagnostic_y = diagnostic_heading_y + section_h
    height = diagnostic_y + 2 * (diagnostic_panel_h + gap) + gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """<style>
        text { font-family: Inter, DejaVu Sans, Arial, sans-serif; fill: #20242A; }
        .title { font-size: 22px; font-weight: 700; }
        .subtitle { font-size: 12px; fill: #5A626D; }
        .section { font-size: 17px; font-weight: 700; fill: #333A44; }
        .panel-title { font-size: 14px; font-weight: 600; }
        .axis-label { font-size: 11px; }
        .tick { font-size: 10px; fill: #59616C; }
        .legend { font-size: 12px; }
        .pending { font-size: 10px; fill: #269E74; font-style: italic; }
        </style>""",
        '<rect width="100%" height="100%" fill="#F5F7F9"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Canonical LeJEPA: downstream transfer and latent-space diagnostics</text>',
        f'<text x="{width/2}" y="50" text-anchor="middle" class="subtitle">Shared exact random initialization at step 0 · independent y-axis scales within panels</text>',
    ]
    legend_x = width / 2 - 210
    for i, lr in enumerate(LRS):
        xx = legend_x + i * 150
        parts.append(f'<line x1="{xx}" y1="75" x2="{xx + 25}" y2="75" stroke="{COLORS[lr]}" stroke-width="3"/>')
        parts.append(marker(xx + 12.5, 75, COLORS[lr], i))
        parts.append(f'<text x="{xx + 33}" y="79" class="legend">{LR_LABELS[lr]}</text>')
    parts.append(f'<text x="{gap}" y="{header + 22}" class="section">Canonical STB tasks — random init, 10k, 16.7k, 20k, 30k (higher is better)</text>')
    for index, key in enumerate(STB_ORDER):
        row, column = divmod(index, 4)
        x0 = gap + column * (stb_panel_w + gap)
        y0 = stb_y + row * (stb_panel_h + gap)
        parts.append(panel(key, x0, y0, stb_panel_w, stb_panel_h))
    parts.append(f'<text x="{gap}" y="{diagnostic_heading_y + 22}" class="section">Latent-space diagnostics — random init, 10k, 16.7k, 20k, 30k</text>')
    for index, title in enumerate(DIAGNOSTICS):
        row, column = divmod(index, 4)
        if row == 1:
            column += 1
        x0 = gap + column * (diagnostic_panel_w + gap)
        y0 = diagnostic_y + row * (diagnostic_panel_h + gap)
        parts.append(diagnostic_panel(title, x0, y0, diagnostic_panel_w, diagnostic_panel_h))
    parts.append("</svg>")
    (OUT_DIR / "canonical_stb_and_latent_diagnostics.svg").write_text("".join(parts))


def svg_document(keys: list[tuple[str, str, str]], columns: int, title: str, filename: str) -> None:
    panel_w, panel_h, gap = 390, 300, 18
    rows = math.ceil(len(keys) / columns)
    width = columns * panel_w + (columns + 1) * gap
    header = 105
    height = header + rows * panel_h + (rows + 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """<style>
        text { font-family: Inter, DejaVu Sans, Arial, sans-serif; fill: #20242A; }
        .title { font-size: 22px; font-weight: 700; }
        .subtitle { font-size: 12px; fill: #5A626D; }
        .panel-title { font-size: 14px; font-weight: 600; }
        .axis-label { font-size: 11px; }
        .tick { font-size: 10px; fill: #59616C; }
        .legend { font-size: 12px; }
        .pending { font-size: 10px; fill: #269E74; font-style: italic; }
        </style>""",
        '<rect width="100%" height="100%" fill="#F5F7F9"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<text x="{width/2:.1f}" y="50" text-anchor="middle" class="subtitle">Shared random initialization at step 0 · intermediate checkpoint is 16,667 steps · higher is better</text>',
    ]
    legend_x = width / 2 - 210
    for i, lr in enumerate(LRS):
        xx = legend_x + i * 150
        parts.append(f'<line x1="{xx}" y1="75" x2="{xx + 25}" y2="75" stroke="{COLORS[lr]}" stroke-width="3"/>')
        parts.append(marker(xx + 12.5, 75, COLORS[lr], i))
        parts.append(f'<text x="{xx + 33}" y="79" class="legend">{LR_LABELS[lr]}</text>')
    for index, key in enumerate(keys):
        row, column = divmod(index, columns)
        x0 = gap + column * (panel_w + gap)
        y0 = header + gap + row * (panel_h + gap)
        parts.append(panel(key, x0, y0, panel_w, panel_h))
    parts.append("</svg>")
    (OUT_DIR / filename).write_text("".join(parts))


def save_source_csv() -> None:
    with (OUT_DIR / "canonical_stb_and_latent_diagnostics_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "task_or_diagnostic", "split", "metric", "learning_rate", "step", "value", "status"])
        for task, split, metric in STB_ORDER:
            datum = DATA[(task, split, metric)]
            for lr in LRS:
                for step, value in zip(STEPS, [datum["random"], *datum[lr]]):
                    writer.writerow(["STB", task, split, metric, lr, step, "" if math.isnan(value) else value, "pending" if math.isnan(value) else "complete"])
        for title, datum in DIAGNOSTICS.items():
            for lr in LRS:
                for step, value in zip(DIAGNOSTIC_STEPS, [datum["random"], *datum[lr]]):
                    writer.writerow(["latent diagnostic", title, "", "", lr, step, value, "complete"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_combined_figure()
    save_source_csv()
    print(f"Wrote combined SVG and source data to {OUT_DIR}")


if __name__ == "__main__":
    main()
