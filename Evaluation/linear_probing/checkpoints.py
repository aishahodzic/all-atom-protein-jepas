"""Frozen encoder registry + loading.

Mirrors the checkpoint-loading logic in
StructTokenBench/src/tokenizer.py:WrappedLaProteinaTokenizer.__init__
(lines 604-621) without the rest of that class (StructTokenBench-specific
data plumbing we don't need here).
"""

import json
import pathlib

import torch

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO_ROOT / "StructTokenBench" / "100k-checkpoints"
JEPA_ENCODER_CFG = CKPT_ROOT / "shared" / "exp_config_jepa_nn130m.json"
EXTERNAL_CKPT_ROOT = REPO_ROOT / "checkpoints_laproteina"

CHECKPOINTS = {
    "ca-cond": ("ae/ca-cond/100k-EMA.ckpt", "ae"),
    "fully-latent": ("ae/fully_latent/100k-EMA.ckpt", "ae"),
    "jepa-mutation": ("jepa/mutation/100k.ckpt", "jepa"),
    "jepa-rand-mask-50": ("jepa/random_masking_50pct/100k.ckpt", "jepa"),
    "jepa-rand-mask-70": ("jepa/random_masking_70pct/100k.ckpt", "jepa"),
    "le-jepa-mutation": ("le-jepa/mutation/100k.ckpt", "jepa"),
    "le-jepa-rand-mask-50": ("le-jepa/random_masking_50pct/100k.ckpt", "jepa"),
    "le-jepa-rand-mask-70": ("le-jepa/random_masking_70pct/100k.ckpt", "jepa"),
    "le-jepa-rand70-relative-position-rotation": (
        "le-jepa/relative-position-and-rotation/100k.ckpt", "jepa"
    ),
    "jepa-rand50-control-30k": ("baseline_ablations/control/last.ckpt", "jepa"),
    "jepa-rand50-ema-30k": ("baseline_ablations/ema/last.ckpt", "jepa"),
    "jepa-rand50-layernorm-30k": ("baseline_ablations/layernorm/last.ckpt", "jepa"),
    "jepa-rand50-predictor-30k": ("baseline_ablations/predictor/last.ckpt", "jepa"),
}

EXTERNAL_CHECKPOINTS = {
    "ae1-external": (str(EXTERNAL_CKPT_ROOT / "AE1_ucond_512.ckpt"), "ae"),
}

RANDOM_INIT_CHECKPOINTS = {
    "random-init": (None, "random"),
}
RANDOM_INIT_SEED = 42
RANDOM_INIT_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parent / "latents" / "random_init_weights_seed42.pt"

ALL_CHECKPOINTS = {**CHECKPOINTS, **EXTERNAL_CHECKPOINTS, **RANDOM_INIT_CHECKPOINTS}


def _infer_latent_z_dim(encoder_weights: dict) -> int:
    """Reads latent_z_dim straight from the checkpoint's own weights, rather than
    trusting a naming convention (a folder/run name saying "64dim" can drift out
    of sync with the actual saved weights; the weights themselves can't).

    latent_decoder_mean_n_log_scale.1 is the encoder's final projection to
    [mean, log_scale] concatenated, so its output width is 2 * latent_z_dim --
    confirmed against a real 64-dim checkpoint (weight shape (128, 768)).
    """
    weight = encoder_weights["latent_decoder_mean_n_log_scale.1.weight"]
    out_dim = weight.shape[0]
    assert out_dim % 2 == 0, f"latent_decoder_mean_n_log_scale.1.weight has odd output dim {out_dim}, expected 2 * latent_z_dim"
    return out_dim // 2


def load_frozen_encoder(
    checkpoint_id: str,
    device: str = "cpu",
    encoder_source: str = "online",
) -> EncoderTransformer:
    """Loads a checkpoint (from CHECKPOINTS, EXTERNAL_CHECKPOINTS, or
    RANDOM_INIT_CHECKPOINTS) into a frozen, eval-mode EncoderTransformer."""
    path, checkpoint_type = ALL_CHECKPOINTS[checkpoint_id]
    if checkpoint_type != "jepa" and encoder_source != "online":
        raise ValueError(
            f"encoder_source={encoder_source!r} is only valid for JEPA checkpoints"
        )

    if checkpoint_type == "random":
        cfg = json.load(open(JEPA_ENCODER_CFG))
        encoder = EncoderTransformer(**cfg["nn_ae"])
        if RANDOM_INIT_WEIGHTS_PATH.exists():
            encoder.load_state_dict(torch.load(RANDOM_INIT_WEIGHTS_PATH, map_location=device))
        else:
            torch.manual_seed(RANDOM_INIT_SEED)
            encoder = EncoderTransformer(**cfg["nn_ae"])  # re-init under the fixed seed
            RANDOM_INIT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), RANDOM_INIT_WEIGHTS_PATH)
    else:
        ckpt_path = pathlib.Path(path)
        if not ckpt_path.is_absolute():
            ckpt_path = CKPT_ROOT / ckpt_path

        if checkpoint_type == "ae":
            model = AutoEncoder.load_from_checkpoint(
                str(ckpt_path), map_location=device, weights_only=False
            )
            encoder = model.encoder
        elif checkpoint_type == "jepa":
            cfg = json.load(open(JEPA_ENCODER_CFG))
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            source_prefixes = {
                "online": ("online_encoder.",),
                "target": ("target_encoder.", "ema_encoder."),
            }
            if encoder_source not in source_prefixes:
                raise ValueError(
                    f"Unknown encoder_source {encoder_source!r}; expected 'online' or 'target'"
                )
            encoder_weights = {}
            encoder_prefix = None
            for prefix in source_prefixes[encoder_source]:
                encoder_weights = {
                    k[len(prefix) :]: v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith(prefix)
                }
                if encoder_weights:
                    encoder_prefix = prefix[:-1]
                    break
            if not encoder_weights:
                raise KeyError(
                    f"Checkpoint {ckpt_path} has no {encoder_source!r} encoder branch; "
                    f"tried {source_prefixes[encoder_source]}"
                )
            latent_z_dim = _infer_latent_z_dim(encoder_weights)
            print(
                f"[checkpoints] {checkpoint_id}: encoder_source={encoder_source} "
                f"({encoder_prefix}), latent_z_dim={latent_z_dim}"
            )
            cfg["nn_ae"]["latent_z_dim"] = latent_z_dim
            cfg["nn_ae"]["encoder"]["latent_z_dim"] = latent_z_dim
            cfg["nn_ae"]["decoder"]["latent_z_dim"] = latent_z_dim
            encoder = EncoderTransformer(**cfg["nn_ae"])
            encoder.load_state_dict(encoder_weights)
        else:
            raise ValueError(f"Unknown checkpoint_type: {checkpoint_type!r}")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device).eval()
    return encoder
