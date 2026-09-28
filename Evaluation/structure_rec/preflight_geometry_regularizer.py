"""One-batch loss, gradient, and memory audit for decoder geometry training."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder

from stereochemistry import stereochemical_loss
from train_decoder import (
    CA_SLOT,
    OUT_DIR,
    build_train_dataloader,
    load_donor_cfg_ae,
    load_frozen_encoder,
)


def gradient_norm(loss: torch.Tensor, parameters, retain_graph: bool) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.float().square().sum()
    return float(torch.sqrt(squared).detach())


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("geometry preflight requires a CUDA GPU")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    encoder = load_frozen_encoder(
        args.encoder, device=str(device), encoder_source=args.encoder_source
    )
    model = AutoEncoder(
        cfg_ae=load_donor_cfg_ae(args.decoder_style),
        store_dir=str(OUT_DIR / "geometry_preflight_model"),
    )
    model.encoder = encoder
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    if args.decoder_checkpoint is not None:
        model.decoder.load_state_dict(
            torch.load(args.decoder_checkpoint, map_location=device, weights_only=True),
            strict=True,
        )
    model = model.to(device)
    model.encoder.eval()
    model.decoder.train()

    dataloader = build_train_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    batch = next(iter(dataloader)).to(device)
    mask = batch["mask_dict"]["coords"][..., 0, 0]
    batch["mask"] = mask
    ca_coors_nm = batch["coords_nm"][..., CA_SLOT, :] * mask[..., None]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        latent_mean = model.encoder(batch)["mean"]
    output = model.decoder(
        {
            "z_latent": latent_mean,
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "mask": mask,
        }
    )
    reconstruction = model.compute_struct_rec_loss(
        output, batch, reduce_mode="sum", loss_ty="l2", weight=1.0
    )
    sequence = model.compute_seq_rec_loss(output, batch, weight=1.0)
    base_loss = sum(
        value.mean()
        for losses in (reconstruction, sequence)
        for name, value in losses.items()
        if "_justlog" not in name
    )
    geometry = stereochemical_loss(
        output,
        batch,
        predicted_sequence_weight=args.predicted_sequence_weight,
        tolerance_factor=args.tolerance_factor,
        clash_overlap_tolerance=args.clash_overlap,
    )

    parameters = [p for p in model.decoder.parameters() if p.requires_grad]
    base_gradient_norm = gradient_norm(base_loss, parameters, retain_graph=True)
    geometry_gradient_norm = gradient_norm(
        geometry["total"], parameters, retain_graph=False
    )
    torch.cuda.synchronize()

    result = {
        "batch_size": args.batch_size,
        "max_residues": int(mask.shape[1]),
        "visible_residues": int(mask.sum()),
        "base_loss": float(base_loss.detach()),
        "geometry": {
            name: float(value.detach()) for name, value in geometry.items()
        },
        "base_gradient_norm": base_gradient_norm,
        "geometry_gradient_norm_unweighted": geometry_gradient_norm,
        "equal_gradient_weight": (
            base_gradient_norm / geometry_gradient_norm
            if geometry_gradient_norm > 0
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "decoder_initialization": (
            str(args.decoder_checkpoint.resolve())
            if args.decoder_checkpoint is not None
            else "fresh_seeded_initialization"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="jepa-rand-mask-70")
    parser.add_argument("--encoder-source", default="target")
    parser.add_argument("--decoder-style", default="fully-latent")
    parser.add_argument(
        "--decoder-checkpoint",
        type=pathlib.Path,
        default=None,
        help="Optional decoder state_dict; omit to audit fresh seeded initialization.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--predicted-sequence-weight", type=float, default=0.5)
    parser.add_argument("--tolerance-factor", type=float, default=12.0)
    parser.add_argument("--clash-overlap", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUT_DIR / "geometry_preflight.json",
    )
    main(parser.parse_args())
