"""Standalone diagnostic for representation collapse in a trained JEPAModel checkpoint.

Loads a checkpoint, runs it on a handful of real proteins, and reports the signals we
used throughout this investigation to distinguish genuine learning from collapse:
  - predicted/target latent std (collapse -> std shrinks toward 0)
  - cosine similarity between predicted and target (collapse -> stays near 1, but that
    alone is NOT sufficient evidence -- see std)
  - cosine similarity BETWEEN DIFFERENT PROTEINS' predictions (collapse -> different
    proteins produce near-identical predictions, since the model has stopped being
    input-dependent; healthy -> different proteins predict differently)
  - raw predicted-vector norm ("confidence"): collapse -> the model settles on
    similar-magnitude outputs everywhere, since it isn't differentiating per input

Does not modify any existing files.

Usage:
  python -m proteinfoundation.jepa.inspect_collapse \\
      --checkpoint /path/to/last.ckpt \\
      --data_dir /mnt/hopper-raid/aishod/data

  # compare two checkpoints side by side:
  python -m proteinfoundation.jepa.inspect_collapse \\
      --checkpoint /path/v3/last.ckpt --checkpoint2 /path/v4/last.ckpt \\
      --data_dir /mnt/hopper-raid/aishod/data
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available on a compute node
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch_geometric.transforms import Compose

from proteinfoundation.datasets.pdb_data import PDBDataset
from proteinfoundation.datasets.transforms import (
    ChainBreakPerResidueTransform,
    CenterStructureTransform,
    CoordsToNanometers,
    GlobalRotationTransform,
)
from proteinfoundation.jepa.jepa_model import JEPAModel, mask_batch_for_encoder
from proteinfoundation.jepa.masking import block_residue_mask, random_residue_mask
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader

def log_print(log_lines, *args):
    """print() that also buffers everything for the .txt log file."""
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_lines.append(msg)


def load_model(checkpoint_path: str) -> JEPAModel:
    """Reconstructs JEPAModel from a checkpoint.

    JEPAModel.__init__ explicitly excludes `cfg` from Lightning's auto-saved
    hyperparameters (`save_hyperparameters(ignore=["cfg"])`), so `.load_from_checkpoint`
    can't rebuild it alone -- we load the exp_config_<run_name>.json that train.py
    writes next to the checkpoints (the fully resolved hydra config, including any CLI
    overrides like +block_mask=true) and construct the model with that.
    """
    ckpt_path = Path(checkpoint_path)
    exp_config_candidates = list(ckpt_path.parent.glob("exp_config_*.json"))
    if not exp_config_candidates:
        raise FileNotFoundError(
            f"No exp_config_*.json found next to {ckpt_path} -- can't reconstruct cfg."
        )
    cfg = OmegaConf.create(json.loads(exp_config_candidates[0].read_text()))

    model = JEPAModel(cfg, use_le_jepa=cfg.get("use_le_jepa", False))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, cfg


def load_proteins(data_dir: str, n: int, log_lines):
    """Loads n real processed proteins into one dense, padded batch dict, using the
    same PDBDataset + DensePaddingDataLoader + transforms train.py uses, so the model
    sees exactly the input format/preprocessing it was trained on."""
    processed_dir = Path(data_dir) / "pdb_train" / "processed"
    all_files = sorted(p.stem for p in processed_dir.glob("*.pt"))[:n]
    log_print(log_lines, f"Loading {len(all_files)} proteins from {processed_dir}:")
    log_print(log_lines, "  " + ", ".join(all_files))

    transform = Compose(
        [
            CoordsToNanometers(),
            CenterStructureTransform(),
            GlobalRotationTransform(),
            ChainBreakPerResidueTransform(),
        ]
    )
    dataset = PDBDataset(
        pdb_codes=[f.rsplit("_", 1)[0] for f in all_files],
        chains=[f.rsplit("_", 1)[1] for f in all_files],
        data_dir=str(data_dir) + "/pdb_train",
        transform=transform,
        file_names=all_files,
    )
    loader = DensePaddingDataLoader(dataset, batch_size=len(all_files), shuffle=False)
    batch = next(iter(loader))
    return batch, all_files


@torch.no_grad()
def run_forward(model: JEPAModel, batch, use_block_mask: bool):
    """Replicates the forward-pass half of JEPAModel.compute_loss (masking -> online
    encoder -> predictor -> ema encoder), without the wandb .log(...) calls (which
    require an attached Trainer and would fail here)."""
    mask = batch["mask_dict"]["coords"][..., 0, 0]
    batch["mask"] = mask

    mask_fn = block_residue_mask if use_block_mask else random_residue_mask
    context_mask, target_mask = mask_fn(batch)

    masked_batch = mask_batch_for_encoder(batch, context_mask)
    online_mean = model._encode_mean(model.online_encoder, masked_batch)  # [B, L, 8]

    from proteinfoundation.jepa.masking import apply_mask_to_latents

    masked_latents = apply_mask_to_latents(online_mean, context_mask, target_mask)
    predicted_latent = model.predictor(
        masked_latents["context_latents"],
        masked_latents["context_indices"],
        masked_latents["target_indices"],
        key_padding_mask=masked_latents["context_padding"],
    )  # [B, L_tgt, 8], raw, pre-normalization

    ema_mean = model._encode_mean(model.ema_encoder, batch)
    ema_masked_latents = apply_mask_to_latents(ema_mean, context_mask, target_mask)
    ema_target = ema_masked_latents["target_latents"]  # [B, L_tgt, 8], raw

    # Raw encoder output on the FULL, unmasked protein -- no context/target split, no
    # predictor involved. Isolates whether the collapse lives in the encoder itself
    # (as opposed to being introduced/amplified by masking or the predictor).
    online_mean_full = model._encode_mean(model.online_encoder, batch)  # [B, L, 8]

    target_padding = masked_latents["target_padding"]
    real_target_residues = torch.logical_not(target_padding)

    return {
        "context_mask": context_mask,
        "target_mask": target_mask,
        "predicted_latent": predicted_latent,
        "ema_target": ema_target,
        "real_target_residues": real_target_residues,
        "online_mean_full": online_mean_full,
        "ema_mean_full": ema_mean,
    }


def inspect_checkpoint(checkpoint_path, data_dir, n_proteins, use_block_mask, out_dir, log_lines, seed=None):
    run_name = Path(checkpoint_path).parent.parent.name  # e.g. "jepa_block_masking_v3"
    ckpt_stem = Path(checkpoint_path).stem  # e.g. "last" or "chk_epoch=..._step=000000010000"
    if ckpt_stem.startswith("chk_") and "step=" in ckpt_stem:
        ckpt_id = "step" + ckpt_stem.split("step=")[1].lstrip("0").zfill(1)
    else:
        ckpt_id = ckpt_stem
    # distinct label even when two checkpoints come from the same run (e.g. 10k vs last)
    label = f"{run_name}_{ckpt_id}"
    if seed is not None:
        label += f"_seed{seed}"
    log_print(log_lines, "=" * 70)
    log_print(log_lines, f"Inspecting checkpoint: {checkpoint_path}")
    log_print(log_lines, f"Run: {label}")

    model, cfg = load_model(checkpoint_path)
    log_print(log_lines, f"Loaded model (cfg.run_name_={cfg.get('run_name_')}, cfg.block_mask={cfg.get('block_mask', False)})")
    log_print(log_lines, f"Inspection masking used: {'block_residue_mask' if use_block_mask else 'random_residue_mask'}")
    if seed is not None:
        log_print(log_lines, f"Seed: {seed}")
        torch.manual_seed(seed)

    batch, protein_ids = load_proteins(data_dir, n_proteins, log_lines)
    out = run_forward(model, batch, use_block_mask)

    context_mask, target_mask = out["context_mask"], out["target_mask"]
    predicted_latent, ema_target = out["predicted_latent"], out["ema_target"]
    real_target_residues = out["real_target_residues"]
    online_mean_full, ema_mean_full = out["online_mean_full"], out["ema_mean_full"]

    # ---- verbose single-example protein ----
    i = 0
    n_valid = int(batch["mask_dict"]["coords"][i, :, 0, 0].sum())
    n_ctx = int(context_mask[i].sum())
    n_tgt = int(target_mask[i].sum())
    log_print(log_lines, "\n--- Example protein (verbose) ---")
    log_print(log_lines, f"Protein ID: {protein_ids[i]}")
    log_print(log_lines, f"Sequence length (valid residues): {n_valid}")
    log_print(log_lines, f"Context residues: {n_ctx}   Target residues: {n_tgt}")

    real_i = real_target_residues[i]
    pred_i = predicted_latent[i][real_i]  # [n_tgt_i, 8]
    targ_i = ema_target[i][real_i]  # [n_tgt_i, 8]

    # ---- evenly-spaced target residues (front/~25%/middle/~75%/end) ----
    # Deliberately NOT "first 5 in sequence order" -- those can land on adjacent
    # residues, and real proteins have local structural correlation, so adjacent
    # residues looking similar isn't itself a collapse signal. Spacing picks across
    # the full target set removes that confound.
    target_idx_i = torch.where(target_mask[i])[0]  # sequence positions of target residues, sorted
    n_t = pred_i.shape[0]
    if n_t >= 2:
        pick = sorted(set(int(round(x.item())) for x in torch.linspace(0, n_t - 1, min(5, n_t))))
    else:
        pick = list(range(n_t))
    log_print(log_lines, f"\n--- Evenly-spaced target residues (front/~25%/middle/~75%/end), not just adjacent ---")
    log_print(log_lines, f"Sequence positions picked: {[int(target_idx_i[p]) for p in pick]} (out of {n_valid} valid residues)")
    for p in pick:
        seq_pos = int(target_idx_i[p])
        pred_row, targ_row = pred_i[p], targ_i[p]
        cos = F.cosine_similarity(pred_row.unsqueeze(0), targ_row.unsqueeze(0)).item()
        log_print(log_lines, f"  seq_pos={seq_pos}:")
        log_print(
            log_lines,
            "    predicted: " + " ".join(f"{v:+.3f}" for v in pred_row.tolist()) + f"   ||pred||={pred_row.norm().item():.3f}",
        )
        log_print(
            log_lines,
            "    target:    " + " ".join(f"{v:+.3f}" for v in targ_row.tolist()) + f"   ||target||={targ_row.norm().item():.3f}",
        )
        log_print(log_lines, f"    cos(pred,target)={cos:.4f}")
    picked_preds = pred_i[pick]  # [k, 8]
    sim_among_picked = F.cosine_similarity(picked_preds.unsqueeze(1), picked_preds.unsqueeze(0), dim=-1)
    k = sim_among_picked.shape[0]
    if k > 1:
        off_diag_picked = ~torch.eye(k, dtype=torch.bool)
        mean_sim_far = sim_among_picked[off_diag_picked].mean().item()
        log_print(log_lines, f"Mean cosine sim AMONG these {k} widely-spaced predicted latents: {mean_sim_far:.4f}")

    # ---- raw encoder output, no masking, no predictor ----
    # Answers: is the target degenerate because the encoder itself is degenerate,
    # independent of anything masking/predictor-related?
    log_print(log_lines, "\n--- Raw encoder output on FULL unmasked protein (no masking, no predictor) ---")
    log_print(log_lines, f"Raw ONLINE encoder output [first 5 of {n_valid} residues]:")
    for row in online_mean_full[i, :5]:
        log_print(log_lines, "  " + " ".join(f"{v:+.3f}" for v in row.tolist()))
    log_print(log_lines, f"Raw EMA encoder output [first 5]:")
    for row in ema_mean_full[i, :5]:
        log_print(log_lines, "  " + " ".join(f"{v:+.3f}" for v in row.tolist()))
    cos_consecutive = F.cosine_similarity(online_mean_full[i, :5], online_mean_full[i, 1:6], dim=-1)
    log_print(log_lines, f"Cosine sim, consecutive residues (online encoder): {[f'{c:.4f}' for c in cos_consecutive.tolist()]}")
    far_idx = min(100, n_valid - 1)
    cos_near_vs_far = F.cosine_similarity(online_mean_full[i, 0], online_mean_full[i, far_idx], dim=-1).item()
    log_print(log_lines, f"Cosine sim, residue 0 vs residue {far_idx} (same protein, far apart): {cos_near_vs_far:.4f}")

    # ---- aggregate stats across all proteins ----
    log_print(log_lines, "\n--- Aggregate stats across all proteins ---")

    all_pred = predicted_latent[real_target_residues]  # [N_total_target, 8], raw
    all_targ = ema_target[real_target_residues]  # [N_total_target, 8], raw

    pred_std = all_pred.std(dim=0)  # [8]
    targ_std = all_targ.std(dim=0)  # [8]
    log_print(log_lines, f"predicted.std(dim=0)  [8 dims]: {[f'{v:.3f}' for v in pred_std.tolist()]}")
    log_print(log_lines, f"target.std(dim=0)     [8 dims]: {[f'{v:.3f}' for v in targ_std.tolist()]}")

    mean_cos_pred_target = F.cosine_similarity(all_pred, all_targ, dim=-1).mean().item()
    log_print(log_lines, f"Mean cosine(predicted, target), same residue: {mean_cos_pred_target:.4f}")

    # per-protein mean predicted vector, then pairwise cosine sim across proteins (i != j)
    per_protein_mean_pred = []
    for b in range(predicted_latent.shape[0]):
        real_b = real_target_residues[b]
        if real_b.any():
            per_protein_mean_pred.append(predicted_latent[b][real_b].mean(dim=0))
    per_protein_mean_pred = torch.stack(per_protein_mean_pred)  # [n_proteins, 8]
    sim_matrix = F.cosine_similarity(
        per_protein_mean_pred.unsqueeze(1), per_protein_mean_pred.unsqueeze(0), dim=-1
    )  # [n_proteins, n_proteins]
    n_p = sim_matrix.shape[0]
    off_diag_mask = ~torch.eye(n_p, dtype=torch.bool)
    inter_protein_cos = sim_matrix[off_diag_mask].mean().item()
    log_print(log_lines, f"Mean cosine(predicted protein_i, predicted protein_j), i!=j: {inter_protein_cos:.4f}")

    # same inter-protein check, but on raw ENCODER output (no masking, no predictor) --
    # isolates whether the collapse is already present in the encoder alone.
    valid_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
    per_protein_mean_enc = []
    for b in range(online_mean_full.shape[0]):
        v = valid_mask[b]
        if v.any():
            per_protein_mean_enc.append(online_mean_full[b][v].mean(dim=0))
    per_protein_mean_enc = torch.stack(per_protein_mean_enc)
    sim_matrix_enc = F.cosine_similarity(
        per_protein_mean_enc.unsqueeze(1), per_protein_mean_enc.unsqueeze(0), dim=-1
    )
    inter_protein_cos_encoder = sim_matrix_enc[~torch.eye(sim_matrix_enc.shape[0], dtype=torch.bool)].mean().item()
    log_print(
        log_lines,
        f"Mean cosine(ONLINE ENCODER protein_i, protein_j), i!=j, no masking/predictor: "
        f"{inter_protein_cos_encoder:.4f}",
    )

    # confidence: raw predicted-vector norm, before normalization
    pred_norms = all_pred.norm(dim=-1)  # [N_total_target]
    mean_norm, std_norm = pred_norms.mean().item(), pred_norms.std().item()
    log_print(log_lines, f"||predicted|| (raw norm): mean={mean_norm:.4f}  std={std_norm:.4f}")

    # ---- PCA plot ----
    # L2-normalize first: raw latent scale (mean_norm) differs wildly across
    # checkpoints/runs, and PCA + matplotlib autoscaling would stretch each plot to
    # fill the figure regardless of true spread, making a collapsed run look just as
    # "spread out" as a healthy one. Normalizing puts every run on the same unit-ball
    # scale, so a fixed axis range is meaningful and plots are comparable across runs.
    pca = PCA(n_components=2)
    normed_pred = F.normalize(all_pred, dim=-1)
    coords_2d = pca.fit_transform(normed_pred.numpy())
    explained = pca.explained_variance_ratio_
    log_print(log_lines, f"PCA explained variance ratio (PC1, PC2): {explained[0]:.4f}, {explained[1]:.4f}")
    # build a per-point protein-index label to color by
    point_protein_idx = []
    for b in range(predicted_latent.shape[0]):
        point_protein_idx += [b] * int(real_target_residues[b].sum())
    point_protein_idx = point_protein_idx[: coords_2d.shape[0]]

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=point_protein_idx, cmap="tab20", s=8, alpha=0.7)
    ax.set_title(f"PCA of L2-normalized predicted latents -- {label}")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    plt.colorbar(scatter, ax=ax, label="protein index")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"collapse_inspection_{label}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_print(log_lines, f"\nSaved PCA plot to {plot_path}")

    return {
        "label": label,
        "pred_std": pred_std.mean().item(),
        "target_std": targ_std.mean().item(),
        "mean_cos_pred_target": mean_cos_pred_target,
        "inter_protein_cos": inter_protein_cos,
        "inter_protein_cos_encoder": inter_protein_cos_encoder,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt file")
    parser.add_argument("--checkpoint2", default=None, help="Optional second .ckpt for comparison")
    parser.add_argument("--data_dir", required=True, help="DATA_PATH, i.e. dir containing pdb_train/processed")
    parser.add_argument("--n_proteins", type=int, default=20)
    parser.add_argument(
        "--block_mask",
        action="store_true",
        help="Use block_residue_mask for the inspection instead of random_residue_mask.",
    )
    parser.add_argument("--out_dir", default=".")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the random seed before loading proteins/masking, so the target/context "
        "split (and picked positions) are identical across checkpoints.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ckpt in [args.checkpoint, args.checkpoint2]:
        if ckpt is None:
            continue
        log_lines = []  # separate log per checkpoint, not shared
        result = inspect_checkpoint(ckpt, args.data_dir, args.n_proteins, args.block_mask, args.out_dir, log_lines, seed=args.seed)
        results.append(result)
        log_path = out_dir / f"collapse_inspection_log_{result['label']}.txt"
        log_path.write_text("\n".join(log_lines))
        print(f"\nLog for {result['label']} saved to {log_path}")

    if len(results) == 2:
        comparison_lines = []
        log_print(comparison_lines, "=" * 70)
        log_print(comparison_lines, "Cross-checkpoint comparison")
        log_print(comparison_lines, "=" * 70)
        r1, r2 = results
        header = f"{'metric':<20} | {r1['label']:<15} | {r2['label']:<15}"
        log_print(comparison_lines, header)
        log_print(comparison_lines, "-" * len(header))
        rows = [
            ("pred_std", "pred_std"),
            ("target_std", "target_std"),
            ("mean_cos_pred_target", "mean_cos_pred_target"),
            ("inter_protein_cos", "inter_protein_cos"),
            ("inter_protein_cos_encoder", "inter_protein_cos_encoder"),
            ("mean_norm", "mean_norm"),
        ]
        for label, key in rows:
            log_print(comparison_lines, f"{label:<20} | {r1[key]:<15.4f} | {r2[key]:<15.4f}")

        comparison_path = out_dir / "collapse_inspection_comparison.txt"
        comparison_path.write_text("\n".join(comparison_lines))
        print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
