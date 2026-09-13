"""Canonical-LeJEPA checkpoint loading, separate from historical JEPA loaders."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


EncoderSource = Literal["ema", "raw"]
_ENCODER_PREFIX = "encoder."
_LATENT_HEAD_WEIGHT = "latent_decoder_mean_n_log_scale.1.weight"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_canonical_encoder_state_dict(
    checkpoint: Mapping, source: EncoderSource = "ema"
) -> dict[str, torch.Tensor]:
    """Extract strict eight-dimensional canonical encoder weights.

    The default never falls back from missing EMA weights to raw weights.
    ``ema`` reads the checkpoint's separate top-level ``state_dict_ema``;
    ``raw`` is an explicit diagnostic reading ``state_dict``.
    """
    if source not in ("ema", "raw"):
        raise ValueError(f"source must be 'ema' or 'raw', got {source!r}")
    root_key = "state_dict_ema" if source == "ema" else "state_dict"
    state = checkpoint.get(root_key)
    if not isinstance(state, Mapping) or not state:
        raise KeyError(
            f"canonical checkpoint has no non-empty {root_key!r} mapping for "
            f"source={source!r}"
        )

    if source == "ema":
        unexpected = [key for key in state if not str(key).startswith(_ENCODER_PREFIX)]
        if unexpected:
            raise ValueError(
                "state_dict_ema must contain encoder parameters only; unexpected "
                f"keys include {unexpected[:3]}"
            )

    encoder_state = {
        str(key)[len(_ENCODER_PREFIX) :]: value
        for key, value in state.items()
        if str(key).startswith(_ENCODER_PREFIX)
    }
    if not encoder_state:
        raise KeyError(f"{root_key!r} contains no {_ENCODER_PREFIX!r} weights")
    if _LATENT_HEAD_WEIGHT not in encoder_state:
        raise KeyError(
            f"canonical encoder is missing {_LATENT_HEAD_WEIGHT!r}"
        )
    head = encoder_state[_LATENT_HEAD_WEIGHT]
    if not isinstance(head, torch.Tensor) or head.ndim != 2:
        raise ValueError("canonical latent-head weight is malformed")
    latent_dim = int(head.shape[0] // 2)
    if head.shape[0] % 2 != 0 or latent_dim != 8:
        raise ValueError(
            f"canonical LeJEPA evaluation requires an 8D encoder, got {latent_dim}D"
        )
    return encoder_state


def load_canonical_encoder(
    checkpoint_path: str | Path,
    encoder_config_path: str | Path,
    *,
    source: EncoderSource = "ema",
    device: str | torch.device = "cpu",
) -> EncoderTransformer:
    """Instantiate a frozen full-protein encoder from EMA (default) or raw weights."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    with Path(encoder_config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    nn_ae = config.get("nn_ae")
    if not isinstance(nn_ae, dict):
        raise KeyError("encoder config must contain an 'nn_ae' mapping")
    if int(nn_ae.get("latent_z_dim", -1)) != 8:
        raise ValueError("canonical encoder config must declare latent_z_dim=8")

    encoder = EncoderTransformer(**nn_ae)
    encoder.load_state_dict(
        extract_canonical_encoder_state_dict(checkpoint, source=source), strict=True
    )
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(device)


def stage_structtokenbench_checkpoint(
    canonical_checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    source: EncoderSource = "ema",
) -> Path:
    """Create an isolated, STB-compatible frozen-encoder artifact.

    [REPO INFRASTRUCTURE] Existing StructTokenBench code remains unchanged.  Its
    historical ``jepa`` loader can consume this deliberately staged artifact as
    an ``online_encoder`` checkpoint.  EMA is the default; selecting raw is
    explicit, and the two staged files have distinct contents/cache identities.
    """
    canonical_checkpoint_path = Path(canonical_checkpoint_path)
    checkpoint = torch.load(
        canonical_checkpoint_path, map_location="cpu", weights_only=False
    )
    encoder_state = extract_canonical_encoder_state_dict(checkpoint, source=source)
    staged = {
        "state_dict": {
            f"online_encoder.{name}": value.detach().cpu()
            for name, value in encoder_state.items()
        },
        "canonical_lejepa_evaluation": {
            "schema_version": 1,
            "encoder_source": source,
            "source_checkpoint_sha256": _sha256_file(canonical_checkpoint_path),
            "source_global_step": checkpoint.get("global_step"),
            "latent_dim": 8,
            "projector": None,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(staged, output_path)
    return output_path
