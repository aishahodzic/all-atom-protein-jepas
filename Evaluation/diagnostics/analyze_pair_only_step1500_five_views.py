"""Measure view identity and collapse in the stopped pair-only raw checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = pathlib.Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "Evaluation/linear_probing"):
    sys.path.insert(0, str(path))

from data_pool import build_protein_pool, eval_transform_compose  # noqa: E402
from proteinfoundation.jepa.canonical_lejepa_model import (  # noqa: E402
    CanonicalLeJEPAModel,
)
from proteinfoundation.utils.dense_padding_data_loader import (  # noqa: E402
    dense_padded_from_data_list,
)


RUN_NAME = (
    "canonical_lejepa_v1_residue_k4_nocrop_"
    "pair_only_position_minvalid4_10k"
)
RUN_ROOT = REPO / "store" / RUN_NAME
DEFAULT_CHECKPOINT = (
    RUN_ROOT / "evaluation/snapshots/raw-step00001500-full-state.ckpt"
)
DEFAULT_CONFIG = (
    RUN_ROOT / "checkpoints" / f"exp_config_{RUN_NAME}.json"
)
DEFAULT_OUTPUT = REPO / (
    "Evaluation/diagnostics/results/"
    "canonical_residue_k4_pair_only_raw1500_five_view.json"
)
DEFAULT_FIGURE = REPO / (
    "Evaluation/latent_viz/figures/canonical_lejepa/"
    "tsne_canonical_residue_k4_pair_only_raw_1500_five-views.png"
)
VIEW_NAMES = ("global", "local 1", "local 2", "local 3", "local 4")
VIEW_COLORS = ("#111111", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3")


def to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if hasattr(value, "to"):
        return value.to(device)
    return value


def effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values.astype(np.float64), rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / np.maximum(eigenvalues.sum(), 1e-30)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def vector_health(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    dimension_std = values.std(axis=0)
    rms_spread = float(np.sqrt(np.square(dimension_std).sum()))
    mean_norm = float(np.linalg.norm(values, axis=1).mean())
    return {
        "effective_rank": effective_rank(values),
        "dimension_std_min": float(dimension_std.min()),
        "dimension_std_max": float(dimension_std.max()),
        "rms_spread": rms_spread,
        "mean_norm": mean_norm,
        "relative_rms_spread": rms_spread / max(mean_norm, 1e-30),
    }


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float((np.sum(a * b, axis=1) / np.maximum(denominator, 1e-30)).mean())


def grouped_classifier_scores(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    binary: bool = False,
) -> dict[str, object]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracy = []
    fold_auc = []
    for train, test in splitter.split(values, labels, groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2_000,
                class_weight="balanced" if binary else None,
                solver="lbfgs",
            ),
        )
        classifier.fit(values[train].astype(np.float64), labels[train])
        prediction = classifier.predict(values[test].astype(np.float64))
        if binary:
            fold_accuracy.append(
                balanced_accuracy_score(labels[test], prediction)
            )
            probability = classifier.predict_proba(
                values[test].astype(np.float64)
            )[:, 1]
            fold_auc.append(roc_auc_score(labels[test], probability))
        else:
            fold_accuracy.append(float((prediction == labels[test]).mean()))
    result: dict[str, object] = {
        "fold_accuracy": [float(value) for value in fold_accuracy],
        "mean_accuracy": float(np.mean(fold_accuracy)),
        "std_accuracy": float(np.std(fold_accuracy, ddof=1)),
    }
    if binary:
        result["fold_auroc"] = [float(value) for value in fold_auc]
        result["mean_auroc"] = float(np.mean(fold_auc))
    return result


def render_joint_tsne(
    values: np.ndarray,
    view_labels: np.ndarray,
    visible: np.ndarray,
    figure_path: pathlib.Path,
) -> None:
    embedding = TSNE(
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1_000,
        metric="cosine",
        init="random",
        random_state=42,
        method="barnes_hut",
        angle=0.5,
        n_jobs=-1,
        verbose=1,
    ).fit_transform(values.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    for view, (name, color) in enumerate(zip(VIEW_NAMES, VIEW_COLORS)):
        mask = view_labels == view
        axes[0].scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=2.8, alpha=0.58, linewidths=0, c=color,
            label=name, rasterized=True,
        )
    axes[0].legend(frameon=False, fontsize=7, markerscale=2)
    axes[0].set_title("five training views")

    global_mask = view_labels == 0
    for mask, name, color in (
        (global_mask, "global", "#111111"),
        (~global_mask, "four locals", "#e6550d"),
    ):
        axes[1].scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=2.8, alpha=0.56, linewidths=0, c=color,
            label=name, rasterized=True,
        )
    axes[1].legend(frameon=False, fontsize=8, markerscale=2)
    axes[1].set_title("global versus local")

    for mask, name, color in (
        (visible, "residue visible", "#2ca25f"),
        (~visible, "residue masked", "#756bb1"),
    ):
        axes[2].scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=2.8, alpha=0.56, linewidths=0, c=color,
            label=name, rasterized=True,
        )
    axes[2].legend(frameon=False, fontsize=8, markerscale=2)
    axes[2].set_title("was this residue itself visible?")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Pair-only residue K=4, raw step 1,500 — joint view t-SNE\n"
        "same physical residues repeated once per view",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=pathlib.Path, default=DEFAULT_FIGURE)
    parser.add_argument("--proteins", type=int, default=64)
    parser.add_argument("--residues", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cfg = OmegaConf.create(json.loads(args.config.read_text()))
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if int(checkpoint["global_step"]) != 1_500:
        raise RuntimeError(f"expected step 1500, got {checkpoint['global_step']}")
    model = CanonicalLeJEPAModel(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    pool = build_protein_pool(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    protein_order = rng.permutation(len(pool.dataset))
    transform = eval_transform_compose()
    residue_views = []
    visibility = []
    protein_groups = []
    protein_ids = []
    used_proteins = 0
    for pool_index in protein_order:
        if used_proteins >= args.proteins:
            break
        protein_id = pool.protein_ids[int(pool_index)]
        graph = transform(pool.dataset[int(pool_index)])
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = (
            graph.chain_breaks_per_residue.unsqueeze(0)
        )
        batch = to_device(batch, device)
        torch.manual_seed(args.seed * 100_000 + int(pool_index))
        torch.cuda.manual_seed_all(args.seed * 100_000 + int(pool_index))
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            encoded, _, view_masks, real_mask = model.encode_residue_views(batch)
        valid = real_mask[0].bool()
        if int(valid.sum()) < 4:
            continue
        residue_views.append(
            encoded[0, :, valid].permute(1, 0, 2).float().cpu().numpy()
        )
        visibility.append(
            view_masks[0, :, valid].permute(1, 0).bool().cpu().numpy()
        )
        protein_groups.append(
            np.full(int(valid.sum()), used_proteins, dtype=np.int64)
        )
        protein_ids.append(protein_id)
        used_proteins += 1
        if used_proteins % 16 == 0:
            print(f"encoded {used_proteins}/{args.proteins} proteins", flush=True)

    views = np.concatenate(residue_views, axis=0)
    visible = np.concatenate(visibility, axis=0)
    groups = np.concatenate(protein_groups, axis=0)
    if len(views) > args.residues:
        selected = np.sort(rng.choice(len(views), size=args.residues, replace=False))
        views = views[selected]
        visible = visible[selected]
        groups = groups[selected]

    residue_count, view_count, latent_dim = views.shape
    flat = views.reshape(residue_count * view_count, latent_dim)
    view_labels = np.tile(np.arange(view_count, dtype=np.int64), residue_count)
    flat_visible = visible.reshape(-1)
    flat_groups = np.repeat(groups, view_count)

    five_way = grouped_classifier_scores(flat, view_labels, flat_groups)
    is_global = (view_labels == 0).astype(np.int64)
    global_local = grouped_classifier_scores(
        flat, is_global, flat_groups, binary=True
    )
    local_rows = view_labels > 0
    four_local = grouped_classifier_scores(
        flat[local_rows], view_labels[local_rows] - 1,
        flat_groups[local_rows],
    )
    visible_classifier = grouped_classifier_scores(
        flat, flat_visible.astype(np.int64), flat_groups, binary=True
    )

    same_residue_mse = [
        float(np.square(views[:, 0] - views[:, local]).mean())
        for local in range(1, 5)
    ]
    same_residue_cosine = [
        mean_cosine(views[:, 0], views[:, local])
        for local in range(1, 5)
    ]
    shuffled = rng.permutation(residue_count)
    different_a = shuffled
    different_b = np.roll(shuffled, 1)
    different_residue_global_mse = float(
        np.square(views[different_a, 0] - views[different_b, 0]).mean()
    )

    grand = views.astype(np.float64).mean(axis=(0, 1), keepdims=True)
    residue_effect = views.astype(np.float64).mean(axis=1, keepdims=True) - grand
    view_effect = views.astype(np.float64).mean(axis=0, keepdims=True) - grand
    interaction = views.astype(np.float64) - grand - residue_effect - view_effect
    total_ss = float(np.square(views.astype(np.float64) - grand).sum())
    residue_ss = float(view_count * np.square(residue_effect).sum())
    view_ss = float(residue_count * np.square(view_effect).sum())
    interaction_ss = float(np.square(interaction).sum())

    render_joint_tsne(flat, view_labels, flat_visible, args.figure)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": int(checkpoint["global_step"]),
        "protocol": {
            "protein_pool": "frozen 5224-protein held-out val+test pool",
            "seed": args.seed,
            "proteins": used_proteins,
            "protein_ids": protein_ids,
            "physical_residues": residue_count,
            "view_points": len(flat),
            "views": list(VIEW_NAMES),
            "same_physical_residue_repeated_across_views": True,
            "global_visible_fraction": float(visible[:, 0].mean()),
            "local_visible_fraction_by_view": [
                float(visible[:, local].mean()) for local in range(1, 5)
            ],
        },
        "same_residue_global_local_mse_by_local": same_residue_mse,
        "same_residue_global_local_mse_mean": float(np.mean(same_residue_mse)),
        "same_residue_global_local_cosine_by_local": same_residue_cosine,
        "same_residue_global_local_cosine_mean": float(
            np.mean(same_residue_cosine)
        ),
        "different_residue_global_mse": different_residue_global_mse,
        "same_view_mse_to_different_residue_mse_ratio": float(
            np.mean(same_residue_mse)
            / max(different_residue_global_mse, 1e-30)
        ),
        "variance_decomposition": {
            "total_sum_squares": total_ss,
            "physical_residue_fraction": residue_ss / max(total_ss, 1e-30),
            "view_identity_fraction": view_ss / max(total_ss, 1e-30),
            "residue_by_view_interaction_fraction": (
                interaction_ss / max(total_ss, 1e-30)
            ),
            "fraction_sum": (
                residue_ss + view_ss + interaction_ss
            ) / max(total_ss, 1e-30),
        },
        "per_view_health": {
            VIEW_NAMES[view]: vector_health(views[:, view])
            for view in range(5)
        },
        "classifiers_grouped_by_held_out_protein": {
            "five_way_view_identity_chance_0.20": five_way,
            "global_vs_local_balanced_chance_0.50": global_local,
            "four_local_identity_chance_0.25": four_local,
            "residue_visible_vs_masked_balanced_chance_0.50": visible_classifier,
        },
        "joint_tsne": {
            "figure": str(args.figure.resolve()),
            "metric": "cosine",
            "perplexity": 30.0,
            "seed": 42,
            "max_iter": 1_000,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
