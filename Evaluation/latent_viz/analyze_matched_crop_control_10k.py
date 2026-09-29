"""Chain-length audit for the matched cropped/uncropped 10k control."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
CANONICAL = REPO / "Evaluation/canonical_lejepa"
for path in (REPO, LINEAR, KNN, CANONICAL, HERE):
    sys.path.insert(0, str(path))

from knn import common_protein_ids, load_encoder_latents  # noqa: E402
from run_tsne import coordinate_path, load_or_fit  # noqa: E402
from thesis_plot_style import BLACK, apply_thesis_style  # noqa: E402
from viz import SEED, sample_indices  # noqa: E402


SPECS = {
    "canonical-lejepa-crop-control-uncropped-raw-step10k": {
        "run": "canonical_lejepa_crop_control_uncropped_local80_min96_b16_nw0_10k",
        "label": "Uncropped LeJEPA",
        "cropping": False,
    },
    "canonical-lejepa-crop-control-fixed96-raw-step10k": {
        "run": "canonical_lejepa_crop_control_fixed96_local80_min96_b16_nw0_10k",
        "label": "Cropped LeJEPA",
        "cropping": True,
    },
}
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
OUTPUT_TAG = "canonical-matched-crop-control-raw10k"
RESULT = HERE / "results/canonical_matched_crop_control_raw10k.csv"
MANIFEST = HERE / "results/canonical_matched_crop_control_raw10k.json"
FIGURES = (
    REPO / "images" / "latent-spaces" / "chain-length-crop-control-paper.png",
)
README_FIGURE = (
    REPO / "images" / "latent-spaces" / "chain-length-crop-control-transparent.png"
)
LATENTS_DIR = LINEAR / "latents"


def flatten(value, prefix="") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, name))
        return result
    return {prefix: value}


def register() -> list[dict]:
    import checkpoints
    import run_tsne as run_tsne_module
    from checkpoint import stage_structtokenbench_checkpoint

    configs = []
    for checkpoint_id, spec in SPECS.items():
        run = str(spec["run"])
        root = REPO / "store" / run
        source = root / "checkpoints" / "step-00010000.ckpt"
        config_path = root / "checkpoints" / f"exp_config_{run}.json"
        staged = root / "evaluation" / f"{checkpoint_id}.ckpt"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs.append(config)
        assert config["views"]["cropping"] is spec["cropping"]
        assert config["views"]["global_mask_pct"] == 0.20
        assert config["views"]["local_mask_pct"] == 0.80
        assert config["alignment_level"] == "protein"
        assert config["sigreg"]["input"] == "protein_mean"
        assert config["sigreg"]["joint_views"] is False
        assert config["dataset"]["datamodule"]["dataselector"]["min_length"] == 96
        assert config["dataset"]["datamodule"]["batch_size"] == 8
        assert config["trainer"]["devices"] == 2
        checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        assert checkpoint.get("global_step") == 10_000
        stage_structtokenbench_checkpoint(source, staged, source="raw")
        entry = (str(staged), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry
        run_tsne_module.ALL_CHECKPOINTS[checkpoint_id] = entry
        run_tsne_module.SHORT_LABEL[checkpoint_id] = str(spec["label"])

    left, right = map(flatten, configs)
    differences = {
        key: [left.get(key), right.get(key)]
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    expected = {"run_name_", "views.cropping"}
    assert set(differences) == expected, differences
    return [{"field": key, "values": value} for key, value in differences.items()]


def extract_chain_latents(checkpoint_ids: list[str]) -> None:
    """Extract full-context residue latents without computing unused SS/RSA labels."""
    from checkpoints import load_frozen_encoder
    from data_pool import build_protein_pool, eval_transform_compose
    from proteinfoundation.utils.dense_padding_data_loader import (
        dense_padded_from_data_list,
    )

    LATENTS_DIR.mkdir(parents=True, exist_ok=True)
    pending = [
        checkpoint_id
        for checkpoint_id in checkpoint_ids
        if not (LATENTS_DIR / f"{checkpoint_id}.pt").is_file()
    ]
    if not pending:
        return
    pool = build_protein_pool()
    transform = eval_transform_compose()
    encoders = {
        checkpoint_id: load_frozen_encoder(checkpoint_id, device="cuda")
        for checkpoint_id in pending
    }
    caches: dict[str, dict[str, torch.Tensor]] = {
        checkpoint_id: {} for checkpoint_id in pending
    }
    for index in tqdm(range(len(pool.dataset)), desc="Extracting chain-length latents"):
        protein_id = pool.protein_ids[index]
        graph = transform(pool.dataset[index])
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
        batch = batch.to("cuda")
        for checkpoint_id, encoder in encoders.items():
            with torch.no_grad():
                latent = encoder(batch)["mean"]
            caches[checkpoint_id][protein_id] = latent[0][batch["mask"][0]].cpu()
    for checkpoint_id in pending:
        torch.save(caches[checkpoint_id], LATENTS_DIR / f"{checkpoint_id}.pt")


class AllResidueSelection:
    def __init__(self, protein_ids: list[str], reference: dict[str, torch.Tensor]):
        self.keys: list[tuple[str, np.ndarray]] = []
        protein_indices = []
        for protein_index, protein_id in enumerate(protein_ids):
            indices = np.arange(len(reference[protein_id]), dtype=np.int64)
            self.keys.append((protein_id, indices))
            protein_indices.append(
                np.full(len(indices), protein_index, dtype=np.int64)
            )
        self.protein_idx = np.concatenate(protein_indices)
        self.n = len(self.protein_idx)

    def build_X(self, per_protein: dict[str, torch.Tensor]) -> np.ndarray:
        return torch.cat(
            [per_protein[protein_id][indices] for protein_id, indices in self.keys],
            dim=0,
        ).numpy()


def render_figure(
    embeddings: dict[str, np.ndarray],
    lengths: np.ndarray,
    *,
    figures: tuple[pathlib.Path, ...] = FIGURES,
    transparent: bool = False,
) -> None:
    apply_thesis_style()
    checkpoint_ids = list(SPECS)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), squeeze=False)
    image = None
    for column, checkpoint_id in enumerate(checkpoint_ids):
        ax = axes[0, column]
        image = ax.scatter(
            embeddings[checkpoint_id][:, 0],
            embeddings[checkpoint_id][:, 1],
            c=lengths,
            cmap="magma",
            vmin=50,
            vmax=256,
            s=3,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(
            str(SPECS[checkpoint_id]["label"]),
            fontsize=17,
            color="white" if transparent else BLACK,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.91, bottom=0.02, top=0.94, wspace=0.03)
    color_ax = fig.add_axes([0.925, 0.14, 0.015, 0.70])
    colorbar = fig.colorbar(image, cax=color_ax)
    text_color = "white" if transparent else BLACK
    colorbar.set_label("chain length L", fontsize=14, color=text_color)
    colorbar.ax.tick_params(labelsize=12, colors=text_color)
    if transparent:
        colorbar.outline.set_edgecolor("white")
        colorbar.outline.set_linewidth(1.5)
    for figure in figures:
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            figure,
            dpi=600,
            bbox_inches="tight",
            facecolor="none" if transparent else "white",
            transparent=transparent,
        )
        print(f"wrote {figure}")
    plt.close(fig)


def plot_from_caches(*, transparent_readme: bool = False) -> None:
    checkpoint_ids = list(SPECS)
    caches = {name: load_encoder_latents(name) for name in checkpoint_ids}
    protein_ids = common_protein_ids(caches)
    selection = AllResidueSelection(protein_ids, caches[checkpoint_ids[0]])
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    lengths = np.concatenate(
        [
            np.full(len(indices), len(indices), dtype=np.float64)
            for _, indices in selection.keys
        ]
    )[sampled]
    embeddings = {}
    for checkpoint_id in checkpoint_ids:
        path = coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED)
        cached = np.load(path, allow_pickle=False)
        embeddings[checkpoint_id] = cached["embedding"]
    render_figure(
        embeddings,
        lengths,
        figures=(README_FIGURE,) if transparent_readme else FIGURES,
        transparent=transparent_readme,
    )


def main() -> None:
    from analyze_current_four_health import (
        external_neighbours,
        neighbour_spearman,
        spectrum_metrics,
    )

    config_differences = register()
    checkpoint_ids = list(SPECS)
    extract_chain_latents(checkpoint_ids)
    caches = {name: load_encoder_latents(name) for name in checkpoint_ids}
    protein_ids = common_protein_ids(caches)
    selection = AllResidueSelection(protein_ids, caches[checkpoint_ids[0]])
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    protein_index = selection.protein_idx[sampled]
    lengths = np.concatenate(
        [
            np.full(len(indices), len(indices), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    log_length = np.log(lengths)
    valid = np.ones(N_POINTS, dtype=bool)

    rows: list[dict[str, object]] = []
    embeddings: dict[str, np.ndarray] = {}
    for checkpoint_id in checkpoint_ids:
        latent = selection.build_X(caches[checkpoint_id])[sampled].astype(np.float32)
        embedding, _, _ = load_or_fit(
            checkpoint_id,
            latent,
            perplexity=PERPLEXITY,
            seed=SEED,
            max_iter=MAX_ITER,
            force=False,
        )
        neighbours = external_neighbours(latent, protein_index)
        rho, n = neighbour_spearman(log_length, valid, neighbours)
        rows.append(
            {
                "checkpoint": checkpoint_id,
                "label": SPECS[checkpoint_id]["label"],
                "cropping": SPECS[checkpoint_id]["cropping"],
                "proteins": len(protein_ids),
                "sampled_residues": N_POINTS,
                "log_chain_length_knn10_spearman": rho,
                "log_chain_length_n": n,
                **spectrum_metrics(latent),
            }
        )
        embeddings[checkpoint_id] = embedding

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    render_figure(embeddings, lengths)

    MANIFEST.write_text(
        json.dumps(
            {
                "encoder_source": "raw",
                "source_step": 10_000,
                "config_differences": config_differences,
                "rows": rows,
                "figures": [str(figure) for figure in FIGURES],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {RESULT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from existing latent and t-SNE caches.",
    )
    parser.add_argument(
        "--transparent-readme",
        action="store_true",
        help="Export only the transparent-background README figure.",
    )
    args = parser.parse_args()
    if args.transparent_readme and not args.plot_only:
        parser.error("--transparent-readme requires --plot-only")
    plot_from_caches(transparent_readme=args.transparent_readme) if args.plot_only else main()
