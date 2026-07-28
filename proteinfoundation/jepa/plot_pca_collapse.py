"""PCA comparison of predictor output latents between an early and a late/collapsed
JEPA block-masking checkpoint.

Does not modify any existing files.

Usage:
  python -m proteinfoundation.jepa.plot_pca_collapse \\
      --checkpoint_early /path/to/step10000.ckpt \\
      --checkpoint_late /path/to/last.ckpt \\
      --data_dir /mnt/hopper-raid/aishod/data
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available on a compute node
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from proteinfoundation.jepa.inspect_collapse import load_model, load_proteins
from proteinfoundation.jepa.jepa_model import mask_batch_for_encoder
from proteinfoundation.jepa.masking import apply_mask_to_latents, random_residue_mask


def get_step_label(checkpoint_path):
    """Reads the Lightning global_step off a checkpoint, for an approximate '~Nk' title."""
    step = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["global_step"]
    return f"~{round(step / 1000)}k"


@torch.no_grad()
def get_predicted_latents(checkpoint_path, batch, seed):
    model, _ = load_model(checkpoint_path)
    model.eval()

    torch.manual_seed(seed)
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
    context_mask, target_mask = random_residue_mask(batch)

    masked_batch = mask_batch_for_encoder(batch, context_mask)
    online_mean = model._encode_mean(model.online_encoder, masked_batch)
    masked_latents = apply_mask_to_latents(online_mean, context_mask, target_mask)
    predicted_latent = model.predictor(
        masked_latents["context_latents"],
        masked_latents["context_indices"],
        masked_latents["target_indices"],
        key_padding_mask=masked_latents["context_padding"],
    )  # [B, L_tgt, 8]

    real_target_residues = ~masked_latents["target_padding"]  # [B, L_tgt]
    point_protein_idx = []
    for b in range(predicted_latent.shape[0]):
        point_protein_idx += [b] * int(real_target_residues[b].sum())

    return predicted_latent[real_target_residues], point_protein_idx  # [N_total_target, 8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_early", required=True, help="Path to early checkpoint (e.g. 10k steps)")
    parser.add_argument("--checkpoint_late", required=True, help="Path to late/collapsed checkpoint")
    parser.add_argument("--data_dir", required=True, help="DATA_PATH, i.e. dir containing pdb_train/processed")
    parser.add_argument("--num_proteins", type=int, default=30)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    batch, _ = load_proteins(args.data_dir, args.num_proteins, log_lines)

    latents_early, colors_early = get_predicted_latents(args.checkpoint_early, dict(batch), args.seed)
    latents_late, colors_late = get_predicted_latents(args.checkpoint_late, dict(batch), args.seed)

    title_early = f"step {get_step_label(args.checkpoint_early)}"
    title_late = f"last ({get_step_label(args.checkpoint_late)})"

    pca_early = PCA(n_components=2)
    coords_early = pca_early.fit_transform(latents_early.numpy())
    pca_late = PCA(n_components=2)
    coords_late = pca_late.fit_transform(latents_late.numpy())

    # same axis limits on both subplots, computed from the early checkpoint
    xlim = (coords_early[:, 0].min(), coords_early[:, 0].max())
    ylim = (coords_early[:, 1].min(), coords_early[:, 1].max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coords, colors, title in [
        (axes[0], coords_early, colors_early, title_early),
        (axes[1], coords_late, colors_late, title_late),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="nipy_spectral", s=8, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    run_name = Path(args.checkpoint_late).parent.parent.name
    plot_path = output_dir / f"pca_collapse_comparison_{run_name}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
