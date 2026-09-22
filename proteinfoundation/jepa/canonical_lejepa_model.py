"""Projector-free canonical LeJEPA adapted to La-Proteina residue encoders.

This module is intentionally isolated from all historical JEPA experiments.
It contains no predictor, target/teacher encoder, stop-gradient branch, or
projector.  Five independently masked views of each protein are encoded by one
shared :class:`EncoderTransformer`; visible residue means are pooled directly
to the native eight-dimensional loss space.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from proteinfoundation.jepa.canonical_sigreg import CanonicalSIGReg
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.coors_utils import sample_uniform_rotation
from proteinfoundation.vendor.stable_pretraining import (
    LinearWarmupCosineAnnealing,
)


CANONICAL_LATENT_DIM = 8
CANONICAL_NUM_LOCAL_VIEWS = 4
CANONICAL_NUM_VIEWS = 1 + CANONICAL_NUM_LOCAL_VIEWS
CANONICAL_ALIGNMENT_LEVELS = ("protein", "residue")
CANONICAL_MASKED_KV_MODES = ("iterative", "visible_only")
CANONICAL_SIGREG_INPUTS = ("protein_mean", "protein_centered_residue")

CANONICAL_PACKED_SAFE_SEQ_FEATURES = (
    "chain_break_per_res",
    "x1_aatype",
    "x1_a37coors_nm",
    "x1_a37coors_nm_rel",
    "chain_idx_seq",
)
CANONICAL_PACKED_SAFE_COND_FEATURES: tuple[str, ...] = ()
CANONICAL_PACKED_SAFE_PAIR_FEATURES = (
    "rel_seq_sep",
    "x1_bb_pair_dists_nm",
    "x1_bb_pair_orientation",
    "chain_idx_pair",
)
CANONICAL_PACKED_REQUIRED_FIELDS = (
    "coords",
    "coords_nm",
    "coord_mask",
    "residue_type",
    "residue_pdb_idx",
    "chains",
    "chain_breaks_per_residue",
)


@dataclass(frozen=True)
class PackedCanonicalView:
    """One canonical view packed in original residue order.

    ``original_indices`` maps packed rows back to the unmodified, full-protein
    layout and is ``-1`` on padding. The original full-layout view mask remains
    the model's public diagnostic output.
    """

    batch: dict[str, Any]
    valid_mask: torch.Tensor
    original_indices: torch.Tensor


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _feature_tuple(features: Any) -> tuple[str, ...]:
    if features is None:
        return ()
    return tuple(str(feature) for feature in features)


def validate_canonical_packed_encoder_config(encoder_cfg: Any) -> None:
    """Fail closed unless the exact audited encoder feature path is active."""
    actual_seq = _feature_tuple(_config_get(encoder_cfg, "feats_seq"))
    actual_cond = _feature_tuple(_config_get(encoder_cfg, "feats_cond_seq"))
    actual_pair = _feature_tuple(_config_get(encoder_cfg, "feats_pair_repr"))
    errors = []
    if actual_seq != CANONICAL_PACKED_SAFE_SEQ_FEATURES:
        errors.append(
            "feats_seq must exactly equal "
            f"{CANONICAL_PACKED_SAFE_SEQ_FEATURES}, got {actual_seq}"
        )
    if actual_cond != CANONICAL_PACKED_SAFE_COND_FEATURES:
        errors.append(f"feats_cond_seq must be empty, got {actual_cond}")
    if actual_pair != CANONICAL_PACKED_SAFE_PAIR_FEATURES:
        errors.append(
            "feats_pair_repr must exactly equal "
            f"{CANONICAL_PACKED_SAFE_PAIR_FEATURES}, got {actual_pair}"
        )
    if bool(_config_get(encoder_cfg, "update_pair_repr", False)):
        errors.append("update_pair_repr must be false")
    if errors:
        raise ValueError(
            "canonical packed views are only valid for the exact audited "
            "encoder configuration: " + "; ".join(errors)
        )


def _validate_residue_tensor(
    batch: Mapping[str, Any], key: str, batch_size: int, length: int
) -> torch.Tensor:
    if key not in batch or batch[key] is None:
        raise ValueError(
            f"canonical packed views require non-null batch['{key}']; "
            "contiguous packed-index fallbacks are forbidden"
        )
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch['{key}'] must be a tensor, got {type(value).__name__}")
    if value.ndim < 2 or tuple(value.shape[:2]) != (batch_size, length):
        raise ValueError(
            f"batch['{key}'] must begin with [B, L]=[{batch_size}, {length}], "
            f"got {tuple(value.shape)}"
        )
    return value


def _gather_and_pad_rows(
    tensor: torch.Tensor,
    retained_indices: list[torch.Tensor],
    packed_length: int,
    *,
    padding_value: bool | int | float,
) -> torch.Tensor:
    packed_rows = []
    for row, indices in enumerate(retained_indices):
        selected = tensor[row].index_select(0, indices)
        padding_length = packed_length - int(indices.numel())
        if padding_length:
            padding = torch.full(
                (padding_length, *tensor.shape[2:]),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            selected = torch.cat((selected, padding), dim=0)
        packed_rows.append(selected)
    return torch.stack(packed_rows, dim=0)


def pack_batch_for_canonical_view(
    batch: Mapping[str, Any],
    visible_mask: torch.Tensor,
    *,
    row_valid_mask: torch.Tensor | None = None,
) -> PackedCanonicalView:
    """Physically retain one view without changing its protein semantics.

    This canonical-only execution helper must be called after the dataset's
    full-protein centering and chain-break transforms. It gathers, but never
    renumbers or recomputes, original residue metadata. Pair features remain
    the encoder's responsibility and are recomputed from this retained set.
    """
    if "mask_dict" not in batch or not isinstance(batch["mask_dict"], Mapping):
        raise ValueError("canonical packed views require a mapping batch['mask_dict']")
    mask_dict = batch["mask_dict"]
    if "coords" not in mask_dict or mask_dict["coords"] is None:
        raise ValueError("canonical packed views require mask_dict['coords']")
    original_coord_padding_mask = mask_dict["coords"]
    if not isinstance(original_coord_padding_mask, torch.Tensor):
        raise TypeError("mask_dict['coords'] must be a tensor")
    if original_coord_padding_mask.ndim != 4:
        raise ValueError(
            "mask_dict['coords'] must have shape [B, L, 37, 3], got "
            f"{tuple(original_coord_padding_mask.shape)}"
        )

    batch_size, length = original_coord_padding_mask.shape[:2]
    if visible_mask.dtype != torch.bool or tuple(visible_mask.shape) != (
        batch_size,
        length,
    ):
        raise ValueError(
            "visible_mask must be boolean [B, L] matching mask_dict['coords'], "
            f"got {tuple(visible_mask.shape)} {visible_mask.dtype}"
        )
    if visible_mask.device != original_coord_padding_mask.device:
        raise ValueError(
            "visible_mask and mask_dict['coords'] must be on the same device"
        )
    coordinate_valid_mask = original_coord_padding_mask[..., 0, 0].bool()
    valid_mask = coordinate_valid_mask if row_valid_mask is None else row_valid_mask
    if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != (
        batch_size,
        length,
    ):
        raise ValueError("row_valid_mask must be boolean [B, L]")
    if valid_mask.device != visible_mask.device:
        raise ValueError("row_valid_mask and visible_mask must be on the same device")
    if (visible_mask & ~valid_mask).any():
        raise ValueError("canonical packed view selected padded residues")
    visible_counts = visible_mask.sum(dim=1)
    if (visible_counts == 0).any():
        rows = torch.where(visible_counts == 0)[0].tolist()
        raise ValueError(
            f"canonical packed view has zero visible residues: rows={rows}"
        )

    tensors = {
        key: _validate_residue_tensor(batch, key, batch_size, length)
        for key in CANONICAL_PACKED_REQUIRED_FIELDS
    }
    expected_shapes = {
        "coords": (batch_size, length, 37, 3),
        "coords_nm": (batch_size, length, 37, 3),
        "coord_mask": (batch_size, length, 37),
        "residue_type": (batch_size, length),
        "residue_pdb_idx": (batch_size, length),
        "chains": (batch_size, length),
        "chain_breaks_per_residue": (batch_size, length),
    }
    for key, value in tensors.items():
        if tuple(value.shape) != expected_shapes[key]:
            raise ValueError(
                f"batch['{key}'] must have exact audited shape "
                f"{expected_shapes[key]}, got {tuple(value.shape)}"
            )
        if value.device != visible_mask.device:
            raise ValueError(f"batch['{key}'] is not on visible_mask's device")
    if not torch.is_floating_point(tensors["coords"]) or not torch.is_floating_point(
        tensors["coords_nm"]
    ):
        raise TypeError("coords and coords_nm must be floating-point tensors")
    if tensors["coord_mask"].dtype != torch.bool:
        raise TypeError("coord_mask must be boolean")
    for key in ("residue_type", "residue_pdb_idx", "chains"):
        if tensors[key].dtype != torch.long:
            raise TypeError(f"{key} must have dtype torch.long")
    if tensors["chain_breaks_per_residue"].dtype != torch.bool:
        raise TypeError("chain_breaks_per_residue must be boolean")

    for key in CANONICAL_PACKED_REQUIRED_FIELDS:
        if key not in mask_dict or mask_dict[key] is None:
            raise ValueError(
                f"canonical packed views require mask_dict['{key}'] so padding "
                "semantics cannot silently change"
            )
        mask_value = mask_dict[key]
        if not isinstance(mask_value, torch.Tensor):
            raise TypeError(f"mask_dict['{key}'] must be a tensor")
        if mask_value.ndim < 2 or tuple(mask_value.shape[:2]) != (
            batch_size,
            length,
        ):
            raise ValueError(
                f"mask_dict['{key}'] must begin with [B, L]="
                f"[{batch_size}, {length}], got {tuple(mask_value.shape)}"
            )
        if mask_value.dtype != torch.bool:
            raise TypeError(f"mask_dict['{key}'] must be boolean")
        if tuple(mask_value.shape) != tuple(tensors[key].shape):
            raise ValueError(
                f"mask_dict['{key}'] must exactly match batch['{key}'] shape "
                f"{tuple(tensors[key].shape)}, got {tuple(mask_value.shape)}"
            )
        if mask_value.device != visible_mask.device:
            raise ValueError(f"mask_dict['{key}'] is not on visible_mask's device")

    retained_indices = [torch.where(visible_mask[row])[0] for row in range(batch_size)]
    packed_length = int(visible_counts.max().item())
    packed_valid = (
        torch.arange(packed_length, device=visible_mask.device)[None, :]
        < visible_counts[:, None]
    )

    padding_values: dict[str, bool | int | float] = {
        "coords": 0.0,
        "coords_nm": 0.0,
        "coord_mask": False,
        "residue_type": -1,
        "residue_pdb_idx": -1,
        "chains": -1,
        "chain_breaks_per_residue": False,
    }
    packed_batch = {
        key: _gather_and_pad_rows(
            value,
            retained_indices,
            packed_length,
            padding_value=padding_values[key],
        )
        for key, value in tensors.items()
    }
    packed_batch["mask_dict"] = {
        key: _gather_and_pad_rows(
            mask_dict[key],
            retained_indices,
            packed_length,
            padding_value=False,
        )
        for key in CANONICAL_PACKED_REQUIRED_FIELDS
    }
    packed_batch["mask"] = packed_valid

    packed_attention_mask = packed_batch["mask_dict"]["coords"][..., 0, 0].bool()
    if row_valid_mask is None and not torch.equal(packed_attention_mask, packed_valid):
        raise ValueError(
            "packed mask_dict['coords'] does not match retained-residue validity"
        )

    original_indices = torch.full(
        (batch_size, packed_length),
        -1,
        dtype=torch.long,
        device=visible_mask.device,
    )
    for row, indices in enumerate(retained_indices):
        original_indices[row, : indices.numel()] = indices

    return PackedCanonicalView(
        batch=packed_batch,
        valid_mask=packed_valid,
        original_indices=original_indices,
    )


def gather_mask_for_packed_canonical_view(
    dense_mask: torch.Tensor, packed_view: PackedCanonicalView
) -> torch.Tensor:
    """Gather a dense residue mask into a packed view's original-row order."""
    if dense_mask.ndim != 2 or dense_mask.dtype != torch.bool:
        raise ValueError("dense_mask must be boolean [B, L]")
    if dense_mask.shape[0] != packed_view.original_indices.shape[0]:
        raise ValueError("dense_mask and packed view must have the same batch size")
    source_rows = packed_view.original_indices.clamp_min(0)
    if source_rows.numel() and int(source_rows.max()) >= dense_mask.shape[1]:
        raise ValueError("packed view contains an out-of-range original residue row")
    gathered = dense_mask.gather(1, source_rows)
    return gathered & packed_view.valid_mask


def scatter_packed_canonical_residues(
    packed_values: torch.Tensor,
    packed_view: PackedCanonicalView,
    *,
    dense_length: int,
) -> torch.Tensor:
    """Scatter packed residue outputs back to their original dense rows.

    The dense result exists only to align independently shifted crops in the
    loss. Feature construction and every transformer layer remain packed.
    """
    if packed_values.ndim != 3:
        raise ValueError("packed_values must have shape [B, P, D]")
    if packed_values.shape[:2] != packed_view.original_indices.shape:
        raise ValueError("packed_values must match the packed view layout")
    if dense_length < 1:
        raise ValueError("dense_length must be positive")
    source_rows = packed_view.original_indices.clamp_min(0)
    if source_rows.numel() and int(source_rows.max()) >= dense_length:
        raise ValueError("packed view contains an out-of-range original residue row")
    values = packed_values * packed_view.valid_mask[..., None].to(
        packed_values.dtype
    )
    dense = packed_values.new_zeros(
        packed_values.shape[0], dense_length, packed_values.shape[-1]
    )
    return dense.scatter_add(
        1, source_rows[..., None].expand_as(packed_values), values
    )


def random_visible_residue_mask(
    valid_mask: torch.Tensor, masked_pct: float
) -> torch.Tensor:
    """Sample visible residues using the historical JEPA split semantics.

    [REPO INFRASTRUCTURE] This reproduces ``random_residue_mask`` from the
    repository's previous JEPA experiments: for each protein, floor
    ``n_valid * masked_pct`` randomly permuted valid residues are hidden and all
    remaining valid residues are visible.  Padding is therefore never visible.
    """
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean [B, L] tensor")
    if not 0.0 <= masked_pct < 1.0:
        raise ValueError(f"masked_pct must be in [0, 1), got {masked_pct}")

    visible = torch.zeros_like(valid_mask)
    for protein_idx in range(valid_mask.shape[0]):
        valid_indices = torch.where(valid_mask[protein_idx])[0]
        n_valid = int(valid_indices.numel())
        if n_valid == 0:
            raise ValueError(
                f"protein {protein_idx} has zero valid residues; cannot form a view"
            )
        permutation = valid_indices[torch.randperm(n_valid, device=valid_mask.device)]
        n_hidden = int(n_valid * masked_pct)
        visible[protein_idx, permutation[n_hidden:]] = True

    if not visible.any(dim=1).all():
        raise RuntimeError("canonical masking produced a view with no visible residue")
    return visible


def build_canonical_view_masks(
    batch: Mapping[str, Any],
    *,
    num_local_views: int = CANONICAL_NUM_LOCAL_VIEWS,
    global_mask_pct: float = 0.20,
    local_mask_pct: float = 0.80,
    crop_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build independently sampled masks ordered global, local1, ..., local4.

    [PROTEIN DEVIATION] Protein views are defined by residue masks rather than
    crops.  The dense reference applies them in place; the optional audited
    packed execution path gathers the same visible residues afterward.  All
    masks are sampled before any encoder forward, so the encoder's unused
    latent-noise draw cannot perturb later view masks.
    """
    if num_local_views != CANONICAL_NUM_LOCAL_VIEWS:
        raise ValueError(
            "canonical LeJEPA V1 requires exactly four local views, got "
            f"{num_local_views}"
        )
    valid_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
    mask_rates = [global_mask_pct] + [local_mask_pct] * num_local_views
    if crop_masks is not None:
        crop_row_mask = batch["mask_dict"]["residue_type"].bool()
        expected = (valid_mask.shape[0], CANONICAL_NUM_VIEWS, valid_mask.shape[1])
        if crop_masks.dtype != torch.bool or tuple(crop_masks.shape) != expected:
            raise ValueError(f"crop_masks must be boolean {expected}")
        if (crop_masks & ~crop_row_mask[:, None]).any():
            raise ValueError("crop_masks may contain only real residues")
        bases = crop_masks.unbind(dim=1)
    else:
        bases = (valid_mask,) * CANONICAL_NUM_VIEWS
    masks = [
        random_visible_residue_mask(base, masked_pct=rate)
        for base, rate in zip(bases, mask_rates)
    ]
    return torch.stack(masks, dim=1)


def build_canonical_crop_masks(
    batch: Mapping[str, Any],
    *,
    crop_length: int | None = None,
    global_crop_length: int | None = None,
    local_crop_length: int | None = None,
    global_crop_fraction_min: float | None = None,
    global_crop_fraction_max: float | None = None,
    global_crop_min_length: int = 48,
    global_crop_max_length: int | None = None,
    locals_within_global: bool = False,
    shared_crop_anchor: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample asymmetric fixed-length crops with optional shared anchoring.

    Crops are represented on the original dense rows solely so physical-residue
    correspondence remains exact. The encoder receives only each crop's query
    and key masks, and no crop offset is added as an input feature.
    """
    valid = batch["mask_dict"]["residue_type"].bool()
    if global_crop_length is None and global_crop_fraction_min is None:
        global_crop_length = crop_length
    if local_crop_length is None:
        local_crop_length = crop_length
    fractional_global = global_crop_fraction_min is not None
    if fractional_global != (global_crop_fraction_max is not None):
        raise ValueError("both global crop fraction bounds must be set together")
    if fractional_global and global_crop_length is not None:
        raise ValueError("global crop length and fraction bounds are mutually exclusive")
    if local_crop_length is None or (
        global_crop_length is None and not fractional_global
    ):
        raise ValueError("global and local crop specifications are required")
    if fractional_global and not (
        0.0 < float(global_crop_fraction_min)
        <= float(global_crop_fraction_max)
        <= 1.0
    ):
        raise ValueError("global crop fractions must satisfy 0 < min <= max <= 1")
    if min(int(local_crop_length), int(global_crop_min_length)) < 2:
        raise ValueError("canonical crop lengths must be at least 2")
    crops = torch.zeros(
        valid.shape[0], CANONICAL_NUM_VIEWS, valid.shape[1],
        dtype=torch.bool, device=valid.device,
    )
    anchors = torch.empty(valid.shape[0], dtype=torch.long, device=valid.device)
    for protein_index in range(valid.shape[0]):
        rows = torch.where(valid[protein_index])[0]
        count = int(rows.numel())
        if fractional_global:
            fraction = float(global_crop_fraction_min) + float(
                torch.rand((), device=valid.device)
            ) * (
                float(global_crop_fraction_max) - float(global_crop_fraction_min)
            )
            sampled_global_length = int(round(fraction * count))
            maximum = count if global_crop_max_length is None else min(
                count, int(global_crop_max_length)
            )
            view_global_length = min(
                maximum, max(int(global_crop_min_length), sampled_global_length)
            )
        else:
            view_global_length = int(global_crop_length)
        view_local_length = int(local_crop_length)
        if locals_within_global and view_local_length > view_global_length:
            raise ValueError("local crops cannot exceed a containing global crop")
        if count < max(view_global_length, view_local_length):
            raise ValueError(
                f"protein {protein_index} has {count} real residues, fewer than "
                "the requested crop lengths"
            )
        anchor_ordinal = int(torch.randint(count, (), device=valid.device))
        anchors[protein_index] = rows[anchor_ordinal]
        global_start_low = max(0, anchor_ordinal - view_global_length + 1)
        global_start_high = min(anchor_ordinal, count - view_global_length)
        global_start = int(
            torch.randint(
                global_start_low, global_start_high + 1, (), device=valid.device
            )
        )
        global_end = global_start + view_global_length
        crops[protein_index, 0, rows[global_start:global_end]] = True
        for view_index in range(1, CANONICAL_NUM_VIEWS):
            if not shared_crop_anchor and locals_within_global:
                start_low = global_start
                start_high = global_end - view_local_length
            elif not shared_crop_anchor:
                start_low = 0
                start_high = count - view_local_length
            elif locals_within_global:
                start_low = max(
                    global_start, anchor_ordinal - view_local_length + 1
                )
                start_high = min(anchor_ordinal, global_end - view_local_length)
            else:
                start_low = max(0, anchor_ordinal - view_local_length + 1)
                start_high = min(anchor_ordinal, count - view_local_length)
            start = int(torch.randint(start_low, start_high + 1, (), device=valid.device))
            crops[protein_index, view_index, rows[start : start + view_local_length]] = True
    return crops, anchors


def recenter_batch_for_canonical_crop(
    batch: Mapping[str, Any], crop_mask: torch.Tensor
) -> dict[str, Any]:
    """Recenter coordinate inputs on the residues visible to the crop."""
    if crop_mask.dtype != torch.bool or crop_mask.shape != batch["coords_nm"].shape[:2]:
        raise ValueError("crop_mask must be boolean [B, L]")
    recentered = dict(batch)
    ca_available = batch["coord_mask"][:, :, 1].bool() & crop_mask
    ca_counts = ca_available.sum(dim=1, keepdim=True)
    if (ca_counts == 0).any():
        rows = torch.where(ca_counts.squeeze(1) == 0)[0].tolist()
        raise ValueError(f"canonical crop has no available CA coordinates: rows={rows}")
    for key in ("coords", "coords_nm"):
        coordinates = batch[key]
        ca = coordinates[:, :, 1, :]
        center = (ca * ca_available[..., None]).sum(dim=1) / ca_counts.to(ca.dtype)
        shifted = coordinates - center[:, None, None, :]
        atom_mask = batch["coord_mask"][..., None].to(shifted.dtype)
        recentered[key] = shifted * atom_mask
    return recentered


def mask_batch_for_canonical_view(
    batch: Mapping[str, Any], visible_mask: torch.Tensor
) -> dict[str, Any]:
    """Apply the established JEPA mask-in-place semantics to one view.

    [REPO INFRASTRUCTURE] Coordinates are zeroed and the encoder's authoritative
    residue/attention mask is cleared at hidden positions.  Other tensors remain
    shared exactly as in the previous JEPA implementation; features derived from
    them are gated by the cleared encoder mask.  Full-protein centering and
    precomputed chain-break features are intentionally retained.
    """
    expected_shape = batch["mask_dict"]["coords"].shape[:2]
    if visible_mask.shape != expected_shape or visible_mask.dtype != torch.bool:
        raise ValueError(
            f"visible_mask must be boolean {tuple(expected_shape)}, got "
            f"{tuple(visible_mask.shape)} {visible_mask.dtype}"
        )

    atom_keep = visible_mask[..., None, None]
    masked_batch = dict(batch)
    for key in ("coords", "coords_nm"):
        if key in batch:
            masked_batch[key] = batch[key] * atom_keep.to(batch[key].dtype)

    masked_batch["mask_dict"] = dict(batch["mask_dict"])
    masked_batch["mask_dict"]["coords"] = batch["mask_dict"]["coords"] & atom_keep
    masked_batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0].bool()
    return masked_batch


def mask_batch_for_canonical_residue_view(
    batch: Mapping[str, Any], visible_mask: torch.Tensor
) -> dict[str, Any]:
    """Create a dense contextual-query view for residue-level alignment.

    Hidden real residues retain their original rows and sequence/chain metadata,
    but their own amino-acid identity, coordinates, and Atom37 availability are
    removed. The returned batch keeps the full real-residue padding mask for
    feature construction; the encoder receives separate real-query and
    visible-key masks at the call site.
    """
    masked_batch = mask_batch_for_canonical_view(batch, visible_mask)
    real_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()

    masked_batch["residue_type"] = batch["residue_type"].masked_fill(~visible_mask, -1)
    masked_batch["coord_mask"] = batch["coord_mask"] & visible_mask[..., None]

    masked_batch["mask_dict"] = dict(masked_batch["mask_dict"])
    for key in ("coords_nm", "coord_mask", "residue_type"):
        if key not in batch["mask_dict"]:
            continue
        field_mask = visible_mask
        while field_mask.ndim < batch["mask_dict"][key].ndim:
            field_mask = field_mask.unsqueeze(-1)
        masked_batch["mask_dict"][key] = batch["mask_dict"][key] & field_mask
    masked_batch["mask"] = real_mask
    return masked_batch


def rotate_batch_for_canonical_view(
    batch: Mapping[str, Any], rotations: torch.Tensor
) -> dict[str, Any]:
    """Rotate each protein in one canonical view without mutating the batch.

    ``rotations[b]`` is shared by every residue and Atom37 coordinate of protein
    ``b``.  The repository uses row-vector coordinates, so the operation is
    ``coordinates @ rotations``.  Both stored coordinate representations are
    transformed from their existing frames; neither is re-derived or re-centred.
    """
    for key in ("coords", "coords_nm"):
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            raise ValueError(f"canonical view rotation requires batch['{key}']")
        if batch[key].ndim != 4 or batch[key].shape[-2:] != (37, 3):
            raise ValueError(
                f"batch['{key}'] must have shape [B, L, 37, 3], got "
                f"{tuple(batch[key].shape)}"
            )
    batch_size = int(batch["coords_nm"].shape[0])
    if tuple(batch["coords"].shape[:2]) != tuple(batch["coords_nm"].shape[:2]):
        raise ValueError("coords and coords_nm must share their [B, L] axes")
    if rotations.shape != (batch_size, 3, 3):
        raise ValueError(
            f"rotations must have shape [{batch_size}, 3, 3], got "
            f"{tuple(rotations.shape)}"
        )
    if rotations.device != batch["coords_nm"].device:
        raise ValueError("rotations and coordinates must be on the same device")

    rotated_batch = dict(batch)
    for key in ("coords", "coords_nm"):
        coordinates = batch[key]
        if coordinates.device != rotations.device:
            raise ValueError(f"batch['{key}'] and rotations must be on the same device")
        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            rotated_batch[key] = torch.einsum(
                "blac,bcd->blad",
                coordinates,
                rotations.to(dtype=coordinates.dtype),
            )
    return rotated_batch


def _sample_canonical_view_rotations(batch: Mapping[str, Any]) -> torch.Tensor:
    """Sample independent ``[local_B, 5, 3, 3]`` SO(3) view rotations.

    Lightning seeds NumPy identically on externally launched DDP ranks, while
    the repository sampler is NumPy/SciPy-backed.  Sampling the complete
    rank-by-protein bank on every rank and selecting a disjoint rank slice keeps
    all protein-view draws independent without changing global RNG state across
    ranks or introducing a collective.
    """
    coordinates = batch["coords_nm"]
    batch_size = int(coordinates.shape[0])
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    rotation_bank = sample_uniform_rotation(
        shape=(world_size, batch_size, CANONICAL_NUM_VIEWS),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    return rotation_bank[rank]


def mean_pool_visible_residues(
    residue_latents: torch.Tensor, visible_mask: torch.Tensor
) -> torch.Tensor:
    """Exactly mean-pool visible, non-padding residue embeddings."""
    if residue_latents.ndim != 3:
        raise ValueError("residue_latents must have shape [B, L, D]")
    if visible_mask.shape != residue_latents.shape[:2]:
        raise ValueError("visible_mask must match residue_latents' [B, L] axes")
    if visible_mask.dtype != torch.bool:
        raise ValueError("visible_mask must be boolean")

    denominator = visible_mask.sum(dim=1, keepdim=True)
    if (denominator == 0).any():
        rows = torch.where(denominator.squeeze(1) == 0)[0].tolist()
        raise ValueError(f"cannot pool views with zero visible residues: rows={rows}")
    numerator = (
        residue_latents * visible_mask.unsqueeze(-1).to(residue_latents.dtype)
    ).sum(dim=1)
    return numerator / denominator.to(residue_latents.dtype)


def canonical_invariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Official LeVJEPA global/local squared-error computation."""
    if embeddings.ndim != 3 or embeddings.shape[1] != CANONICAL_NUM_VIEWS:
        raise ValueError(
            f"embeddings must have shape [B, {CANONICAL_NUM_VIEWS}, D], got "
            f"{tuple(embeddings.shape)}"
        )
    global_embedding = embeddings[:, :1]
    return (global_embedding - embeddings).pow(2).mean()


def canonical_residue_invariance_loss(
    residue_embeddings: torch.Tensor, real_mask: torch.Tensor
) -> torch.Tensor:
    """Align identical residue rows from the global view to all five views.

    Each protein contributes equally: the squared error is averaged over that
    protein's real residues and latent dimensions, then over batch and views.
    Padding alone is excluded; visibility never gates this objective.
    """
    if (
        residue_embeddings.ndim != 4
        or residue_embeddings.shape[1] != CANONICAL_NUM_VIEWS
    ):
        raise ValueError(
            "residue_embeddings must have shape "
            f"[B, {CANONICAL_NUM_VIEWS}, L, D], got {tuple(residue_embeddings.shape)}"
        )
    if real_mask.dtype != torch.bool:
        raise ValueError("real_mask must be boolean")
    if real_mask.ndim == 2:
        alignment_mask = real_mask[:, None].expand(-1, CANONICAL_NUM_VIEWS, -1)
    elif real_mask.shape == residue_embeddings.shape[:3]:
        alignment_mask = real_mask[:, :1] & real_mask
    else:
        raise ValueError("real_mask must be [B, L] or [B, 5, L]")
    real_counts = alignment_mask.sum(dim=2)
    if (real_counts == 0).any():
        rows = torch.where(real_counts == 0)[0].tolist()
        raise ValueError(f"residue invariance has no real residues: rows={rows}")

    squared_error = (residue_embeddings[:, :1] - residue_embeddings).pow(2)
    masked_sum = (
        squared_error * alignment_mask[..., None].to(squared_error.dtype)
    ).sum(dim=(2, 3))
    per_protein_view = masked_sum / (
        real_counts.to(squared_error.dtype) * residue_embeddings.shape[-1]
    )
    return per_protein_view.mean()


def protein_centered_residue_sigreg_input(
    residue_embeddings: torch.Tensor,
    real_mask: torch.Tensor,
    *,
    sampled_indices: torch.Tensor | None = None,
    samples_per_protein: int = 1,
    independent_views: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a protein-balanced, protein-centred residue sample for SIGReg.

    Each view is centred independently using all real (non-padding) residue
    outputs from that protein. The historical/default path samples exactly one
    physical row per protein and shares it across all five views.

    With ``independent_views=True``, each protein/view independently samples
    ``samples_per_protein`` distinct real rows. The result has axes
    ``[view * draw, protein, latent]``: every draw remains a separate SIGReg
    distribution and the protein-balanced sample axis remains ``B``. Residue
    vectors are never averaged together.

    ``sampled_indices`` is an explicit deterministic test hook. Production
    calls leave it unset and sample uniformly from the real rows.
    """
    if (
        residue_embeddings.ndim != 4
        or residue_embeddings.shape[1] != CANONICAL_NUM_VIEWS
    ):
        raise ValueError(
            "residue_embeddings must have shape "
            f"[B, {CANONICAL_NUM_VIEWS}, L, D], got "
            f"{tuple(residue_embeddings.shape)}"
        )
    batch_size, _, length, latent_dim = residue_embeddings.shape
    if real_mask.dtype != torch.bool:
        raise ValueError("real_mask must be boolean")
    if real_mask.shape == (batch_size, length):
        view_real = real_mask[:, None].expand(-1, CANONICAL_NUM_VIEWS, -1)
    elif real_mask.shape == (batch_size, CANONICAL_NUM_VIEWS, length):
        view_real = real_mask
    else:
        raise ValueError("real_mask must be [B, L] or [B, 5, L]")
    if not isinstance(samples_per_protein, int) or samples_per_protein < 1:
        raise ValueError("samples_per_protein must be a positive integer")
    if samples_per_protein != 1 and not independent_views:
        raise ValueError("samples_per_protein>1 requires independent_views=True")

    if not independent_views:
        shared_real = view_real.all(dim=1)
        real_counts = shared_real.sum(dim=1)
        if (real_counts == 0).any():
            rows = torch.where(real_counts == 0)[0].tolist()
            raise ValueError(
                f"protein-centred residue SIGReg has no real residues: rows={rows}"
            )

        if sampled_indices is None:
            sampled_ordinals = torch.floor(
                torch.rand(batch_size, device=residue_embeddings.device)
                * real_counts.to(dtype=torch.float32)
            ).to(dtype=torch.long)
            real_ordinals = shared_real.to(dtype=torch.long).cumsum(dim=1) - 1
            selected_rows = shared_real & (
                real_ordinals == sampled_ordinals.unsqueeze(1)
            )
            sampled_indices = selected_rows.to(dtype=torch.long).argmax(dim=1)
        else:
            if (
                sampled_indices.dtype not in (torch.int32, torch.int64)
                or sampled_indices.shape != (batch_size,)
            ):
                raise ValueError(
                    "sampled_indices must be an integer tensor with shape [B]"
                )
            sampled_indices = sampled_indices.to(
                device=residue_embeddings.device, dtype=torch.long
            )
            if ((sampled_indices < 0) | (sampled_indices >= length)).any():
                raise ValueError("sampled_indices contains an out-of-range residue row")
            if not shared_real.gather(1, sampled_indices.unsqueeze(1)).all():
                raise ValueError("sampled_indices may select only real/non-padding rows")
    else:
        per_view_counts = view_real.sum(dim=2)
        if (per_view_counts < samples_per_protein).any():
            protein_rows, view_rows = torch.where(
                per_view_counts < samples_per_protein
            )
            locations = list(zip(protein_rows.tolist(), view_rows.tolist()))
            raise ValueError(
                "protein-centred residue SIGReg has fewer than "
                f"{samples_per_protein} real residues at protein/view rows: "
                f"{locations}"
            )
        expected_shape = (
            batch_size,
            CANONICAL_NUM_VIEWS,
            samples_per_protein,
        )
        if sampled_indices is None:
            scores = torch.rand(
                batch_size,
                CANONICAL_NUM_VIEWS,
                length,
                device=residue_embeddings.device,
            ).masked_fill(~view_real, float("inf"))
            sampled_indices = scores.topk(
                samples_per_protein, dim=2, largest=False
            ).indices
        else:
            if (
                sampled_indices.dtype not in (torch.int32, torch.int64)
                or tuple(sampled_indices.shape) != expected_shape
            ):
                raise ValueError(
                    "sampled_indices must be an integer tensor with shape "
                    f"{expected_shape}"
                )
            sampled_indices = sampled_indices.to(
                device=residue_embeddings.device, dtype=torch.long
            )
            if ((sampled_indices < 0) | (sampled_indices >= length)).any():
                raise ValueError("sampled_indices contains an out-of-range residue row")
            if not view_real.gather(2, sampled_indices).all():
                raise ValueError("sampled_indices may select only real/non-padding rows")
            if samples_per_protein > 1:
                sorted_indices = sampled_indices.sort(dim=2).values
                if (sorted_indices[:, :, 1:] == sorted_indices[:, :, :-1]).any():
                    raise ValueError(
                        "sampled_indices must be distinct within each protein/view"
                    )

    mask = view_real[..., None].to(dtype=residue_embeddings.dtype)
    protein_means = (residue_embeddings * mask).sum(dim=2, keepdim=True) / (
        view_real.sum(dim=2)[..., None, None].to(dtype=residue_embeddings.dtype)
    )
    centered = residue_embeddings - protein_means
    if not independent_views:
        gather_indices = sampled_indices[:, None, None, None].expand(
            batch_size, CANONICAL_NUM_VIEWS, 1, latent_dim
        )
        selected = centered.gather(dim=2, index=gather_indices).squeeze(2)
        return selected.permute(1, 0, 2), sampled_indices

    gather_indices = sampled_indices[..., None].expand(
        batch_size, CANONICAL_NUM_VIEWS, samples_per_protein, latent_dim
    )
    selected = centered.gather(dim=2, index=gather_indices)
    return (
        selected.permute(1, 2, 0, 3).reshape(
            CANONICAL_NUM_VIEWS * samples_per_protein,
            batch_size,
            latent_dim,
        ),
        sampled_indices,
    )


def residue_invariance_visibility_mse(
    residue_embeddings: torch.Tensor,
    real_mask: torch.Tensor,
    view_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Diagnostic residue MSE for four global/local visibility combinations."""
    global_visible = view_masks[:, :1]
    local_visible = view_masks[:, 1:]
    squared_error = (residue_embeddings[:, :1] - residue_embeddings[:, 1:]).pow(2)
    if real_mask.ndim == 2:
        real = real_mask[:, None, :]
    elif real_mask.shape == residue_embeddings.shape[:3]:
        real = real_mask[:, :1] & real_mask[:, 1:]
    else:
        raise ValueError("real_mask must be [B, L] or [B, 5, L]")
    combinations = {
        "visible_visible": real & global_visible & local_visible,
        "visible_masked": real & global_visible & ~local_visible,
        "masked_visible": real & ~global_visible & local_visible,
        "masked_masked": real & ~global_visible & ~local_visible,
    }
    metrics = {}
    for name, mask in combinations.items():
        count = mask.sum()
        if int(count) == 0:
            metrics[name] = squared_error.new_zeros(())
        else:
            metrics[name] = (
                squared_error * mask[..., None].to(squared_error.dtype)
            ).sum() / (count.to(squared_error.dtype) * squared_error.shape[-1])
    return metrics


def covariance_effective_rank(values: torch.Tensor) -> torch.Tensor:
    """Effective rank of a centered ``[N, D]`` embedding matrix."""
    if values.ndim != 2:
        raise ValueError("effective rank expects [N, D]")
    if values.shape[0] < 2:
        return values.new_zeros((), dtype=torch.float32)
    centered = values.detach().float() - values.detach().float().mean(
        dim=0, keepdim=True
    )
    variance = torch.linalg.svdvals(centered).square()
    total = variance.sum()
    if total <= 0:
        return total
    probabilities = variance / total
    return (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()


def build_adamw_parameter_groups(
    module: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Apply official LeVJEPA bias/norm/one-dimensional WD exclusion."""
    decay_parameters: list[torch.nn.Parameter] = []
    excluded_parameters: list[torch.nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if "bias" in lowered or "norm" in lowered or parameter.ndim == 1:
            excluded_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    if not decay_parameters:
        raise ValueError("optimizer has no weight-decayed parameters")
    return [
        {"params": decay_parameters, "weight_decay": float(weight_decay)},
        {"params": excluded_parameters, "weight_decay": 0.0},
    ]


class CanonicalLeJEPAModel(L.LightningModule):
    """One-encoder, projector-free canonical protein LeJEPA V1."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        nn_ae = _config_get(cfg, "nn_ae")
        latent_dim = int(_config_get(nn_ae, "latent_z_dim"))
        encoder_cfg = _config_get(nn_ae, "encoder")
        encoder_latent_dim = int(_config_get(encoder_cfg, "latent_z_dim"))
        if (
            latent_dim != CANONICAL_LATENT_DIM
            or encoder_latent_dim != CANONICAL_LATENT_DIM
        ):
            raise ValueError(
                "canonical LeJEPA V1 requires top-level and encoder "
                "latent_z_dim == 8, got "
                f"{latent_dim} and {encoder_latent_dim}"
            )

        views = _config_get(cfg, "views")
        self.alignment_level = str(_config_get(cfg, "alignment_level", "protein"))
        if self.alignment_level not in CANONICAL_ALIGNMENT_LEVELS:
            raise ValueError(
                "canonical LeJEPA alignment_level must be one of "
                f"{CANONICAL_ALIGNMENT_LEVELS}, got {self.alignment_level!r}"
            )
        self.masked_kv_mode = str(_config_get(cfg, "masked_kv_mode", "visible_only"))
        if self.masked_kv_mode not in CANONICAL_MASKED_KV_MODES:
            raise ValueError(
                "canonical LeJEPA masked_kv_mode must be one of "
                f"{CANONICAL_MASKED_KV_MODES}, got {self.masked_kv_mode!r}"
            )
        self.residue_view_checkpointing = bool(
            _config_get(cfg, "residue_view_checkpointing", False)
        )
        self.residue_compact_visible_kv = bool(
            _config_get(cfg, "residue_compact_visible_kv", False)
        )
        if self.residue_compact_visible_kv and (
            self.alignment_level != "residue"
            or self.masked_kv_mode != "visible_only"
        ):
            raise ValueError(
                "residue_compact_visible_kv requires alignment_level=residue "
                "and masked_kv_mode=visible_only"
            )
        self.num_local_views = int(_config_get(views, "num_local_views"))
        self.global_mask_pct = float(_config_get(views, "global_mask_pct"))
        self.local_mask_pct = float(_config_get(views, "local_mask_pct"))
        self.cropping = bool(_config_get(views, "cropping", False))
        if self.num_local_views != CANONICAL_NUM_LOCAL_VIEWS:
            raise ValueError("canonical LeJEPA V1 requires exactly four local views")
        if self.global_mask_pct != 0.20:
            raise ValueError("canonical LeJEPA V1 requires global_mask_pct=0.20")
        if self.local_mask_pct != 0.80 and not (
            self.cropping and self.local_mask_pct == 0.50
        ):
            raise ValueError(
                "canonical LeJEPA requires local_mask_pct=0.80, or 0.50 for "
                "the cropped-view ablation"
            )
        readout = _config_get(cfg, "readout")
        if str(_config_get(readout, "type")) != "mean_visible_residues":
            raise ValueError(
                "canonical LeJEPA V1 requires readout.type=mean_visible_residues"
            )
        legacy_crop_length = _config_get(views, "crop_length", None)
        self.global_crop_fraction_min = _config_get(
            views, "global_crop_fraction_min", None
        )
        self.global_crop_fraction_max = _config_get(
            views, "global_crop_fraction_max", None
        )
        global_crop_length = _config_get(views, "global_crop_length", None)
        if global_crop_length is None and self.global_crop_fraction_min is None:
            global_crop_length = 48 if legacy_crop_length is None else legacy_crop_length
        self.global_crop_length = (
            None if global_crop_length is None else int(global_crop_length)
        )
        self.local_crop_length = int(
            _config_get(
                views,
                "local_crop_length",
                48 if legacy_crop_length is None else legacy_crop_length,
            )
        )
        self.global_crop_min_length = int(
            _config_get(views, "global_crop_min_length", 48)
        )
        global_crop_max_length = _config_get(views, "global_crop_max_length", None)
        self.global_crop_max_length = (
            None if global_crop_max_length is None else int(global_crop_max_length)
        )
        self.locals_within_global = bool(
            _config_get(views, "locals_within_global", False)
        )
        self.shared_crop_anchor = bool(
            _config_get(views, "shared_crop_anchor", True)
        )
        for forbidden, expected in (
            ("rotation", False),
            ("physical_token_drop", False),
        ):
            if bool(_config_get(views, forbidden, expected)) is not expected:
                raise ValueError(
                    f"canonical LeJEPA V1 requires views.{forbidden}=false"
                )
        self.packed_views = bool(_config_get(views, "packed_views", False))
        if self.alignment_level == "residue" and self.packed_views and not self.cropping:
            raise ValueError(
                "canonical residue alignment permits views.packed_views=true only "
                "with views.cropping=true"
            )
        if self.packed_views:
            validate_canonical_packed_encoder_config(encoder_cfg)
        self.independent_view_rotations = bool(
            _config_get(cfg, "independent_view_rotations", False)
        )

        sigreg_cfg = _config_get(cfg, "sigreg")
        self.sigreg_weight = float(_config_get(sigreg_cfg, "weight"))
        self.sigreg_input = str(_config_get(sigreg_cfg, "input", "protein_mean"))
        self.sigreg_joint_views = bool(
            _config_get(sigreg_cfg, "joint_views", False)
        )
        self.residue_samples_per_protein = int(
            _config_get(sigreg_cfg, "residue_samples_per_protein", 1)
        )
        self.independent_residue_samples = bool(
            _config_get(sigreg_cfg, "independent_residue_samples", False)
        )
        if self.sigreg_input not in CANONICAL_SIGREG_INPUTS:
            raise ValueError(
                "canonical LeJEPA sigreg.input must be one of "
                f"{CANONICAL_SIGREG_INPUTS}, got {self.sigreg_input!r}"
            )
        if (
            self.sigreg_input == "protein_centered_residue"
            and self.alignment_level != "residue"
        ):
            raise ValueError(
                "sigreg.input=protein_centered_residue requires "
                "alignment_level=residue"
            )
        if self.residue_samples_per_protein < 1:
            raise ValueError("sigreg.residue_samples_per_protein must be positive")
        if (
            self.residue_samples_per_protein != 1
            and not self.independent_residue_samples
        ):
            raise ValueError(
                "sigreg.residue_samples_per_protein>1 requires "
                "sigreg.independent_residue_samples=true"
            )
        if (
            self.independent_residue_samples
            or self.residue_samples_per_protein != 1
        ) and self.sigreg_input != "protein_centered_residue":
            raise ValueError(
                "residue SIGReg sampling options require "
                "sigreg.input=protein_centered_residue"
            )
        if (
            not self.shared_crop_anchor
            and self.sigreg_input == "protein_centered_residue"
            and not self.independent_residue_samples
        ):
            raise ValueError(
                "independent local crop starts with protein-centred residue "
                "SIGReg require sigreg.independent_residue_samples=true"
            )
        self.encoder = EncoderTransformer(**nn_ae)
        self.sigreg = CanonicalSIGReg(
            knots=int(_config_get(sigreg_cfg, "knots")),
            num_proj=int(_config_get(sigreg_cfg, "num_proj")),
            normalize_by_n=bool(_config_get(sigreg_cfg, "normalize_by_n")),
        )
        self._first_batch_sanity_reported = False

    def encode_views(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return direct pooled 8D embeddings and their visibility masks."""
        full_real_mask = (
            batch["mask_dict"]["residue_type"].bool()
            if self.cropping
            else batch["mask_dict"]["coords"][..., 0, 0].bool()
        )
        crop_masks = (
            build_canonical_crop_masks(
                batch,
                global_crop_length=self.global_crop_length,
                local_crop_length=self.local_crop_length,
                global_crop_fraction_min=self.global_crop_fraction_min,
                global_crop_fraction_max=self.global_crop_fraction_max,
                global_crop_min_length=self.global_crop_min_length,
                global_crop_max_length=self.global_crop_max_length,
                locals_within_global=self.locals_within_global,
                shared_crop_anchor=self.shared_crop_anchor,
            )[0]
            if self.cropping
            else None
        )
        view_masks = build_canonical_view_masks(
            batch,
            num_local_views=self.num_local_views,
            global_mask_pct=self.global_mask_pct,
            local_mask_pct=self.local_mask_pct,
            crop_masks=crop_masks,
        )
        rotations = (
            _sample_canonical_view_rotations(batch)
            if self.independent_view_rotations
            else None
        )
        pooled_views = []
        for view_index in range(CANONICAL_NUM_VIEWS):
            visible_mask = view_masks[:, view_index]
            view_batch = (
                rotate_batch_for_canonical_view(batch, rotations[:, view_index])
                if rotations is not None
                else batch
            )
            if self.cropping:
                view_batch = recenter_batch_for_canonical_crop(
                    view_batch, crop_masks[:, view_index]
                )
            if self.packed_views:
                packed_view = pack_batch_for_canonical_view(
                    view_batch,
                    visible_mask,
                    row_valid_mask=full_real_mask if self.cropping else None,
                )
                residue_latents = self.encoder(packed_view.batch)["mean"]
                pooling_mask = packed_view.valid_mask
            else:
                # Dense reference/fallback path: intentionally retain the
                # established mask-in-place implementation unchanged.
                masked_batch = mask_batch_for_canonical_view(view_batch, visible_mask)
                if self.cropping:
                    masked_batch["mask"] = visible_mask
                residue_latents = self.encoder(masked_batch)["mean"]
                pooling_mask = visible_mask
            if residue_latents.shape[-1] != CANONICAL_LATENT_DIM:
                raise RuntimeError(
                    "encoder returned an unexpected latent dimension: "
                    f"{tuple(residue_latents.shape)}"
                )
            pooled_views.append(
                mean_pool_visible_residues(residue_latents, pooling_mask)
            )
        embeddings = torch.stack(pooled_views, dim=1)
        return embeddings, view_masks

    def _encode_residue_mean(
        self,
        batch: Mapping[str, Any],
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one residue view through the shared encoder."""
        encoder_kwargs = {"query_mask": query_mask, "key_mask": key_mask}
        if self.residue_compact_visible_kv:
            encoder_kwargs["compact_kv"] = True
        return self.encoder(batch, **encoder_kwargs)["mean"]

    def encode_residue_views(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dense residue outputs and all-real protein embeddings."""
        if self.packed_views and not self.cropping:
            raise ValueError(
                "canonical residue alignment permits views.packed_views=true only "
                "with views.cropping=true"
            )
        coordinate_real_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
        full_real_mask = (
            batch["mask_dict"]["residue_type"].bool()
            if self.cropping
            else coordinate_real_mask
        )
        crop_masks = (
            build_canonical_crop_masks(
                batch,
                global_crop_length=self.global_crop_length,
                local_crop_length=self.local_crop_length,
                global_crop_fraction_min=self.global_crop_fraction_min,
                global_crop_fraction_max=self.global_crop_fraction_max,
                global_crop_min_length=self.global_crop_min_length,
                global_crop_max_length=self.global_crop_max_length,
                locals_within_global=self.locals_within_global,
                shared_crop_anchor=self.shared_crop_anchor,
            )[0]
            if self.cropping
            else full_real_mask[:, None].expand(-1, CANONICAL_NUM_VIEWS, -1)
        )
        view_masks = build_canonical_view_masks(
            batch,
            num_local_views=self.num_local_views,
            global_mask_pct=self.global_mask_pct,
            local_mask_pct=self.local_mask_pct,
            crop_masks=crop_masks,
        )
        rotations = (
            _sample_canonical_view_rotations(batch)
            if self.independent_view_rotations
            else None
        )
        residue_views = []
        protein_views = []
        for view_index in range(CANONICAL_NUM_VIEWS):
            real_mask = crop_masks[:, view_index]
            visible_mask = view_masks[:, view_index]
            key_mask = (
                real_mask if self.masked_kv_mode == "iterative" else visible_mask
            )
            view_batch = (
                rotate_batch_for_canonical_view(batch, rotations[:, view_index])
                if rotations is not None
                else batch
            )
            if self.cropping:
                view_batch = recenter_batch_for_canonical_crop(view_batch, real_mask)
            packed_view = None
            if self.cropping and self.packed_views:
                packed_view = pack_batch_for_canonical_view(
                    view_batch,
                    real_mask,
                    row_valid_mask=full_real_mask,
                )
                packed_visible_mask = gather_mask_for_packed_canonical_view(
                    visible_mask, packed_view
                )
                masked_batch = mask_batch_for_canonical_residue_view(
                    packed_view.batch, packed_visible_mask
                )
                query_mask = packed_view.valid_mask
                masked_batch["mask"] = query_mask
                key_mask = (
                    query_mask
                    if self.masked_kv_mode == "iterative"
                    else packed_visible_mask
                )
            else:
                masked_batch = mask_batch_for_canonical_residue_view(
                    view_batch, visible_mask
                )
                query_mask = real_mask
                if self.cropping:
                    masked_batch["mask"] = real_mask
            if self.residue_view_checkpointing and torch.is_grad_enabled():
                residue_latents = activation_checkpoint(
                    self._encode_residue_mean,
                    masked_batch,
                    query_mask,
                    key_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                residue_latents = self._encode_residue_mean(
                    masked_batch,
                    query_mask,
                    key_mask,
                )
            if residue_latents.shape[-1] != CANONICAL_LATENT_DIM:
                raise RuntimeError(
                    "encoder returned an unexpected latent dimension: "
                    f"{tuple(residue_latents.shape)}"
                )
            dense_residue_latents = (
                scatter_packed_canonical_residues(
                    residue_latents,
                    packed_view,
                    dense_length=full_real_mask.shape[1],
                )
                if packed_view is not None
                else residue_latents
            )
            residue_views.append(dense_residue_latents)
            protein_views.append(
                mean_pool_visible_residues(residue_latents, query_mask)
            )
        return (
            torch.stack(residue_views, dim=1),
            torch.stack(protein_views, dim=1),
            view_masks,
            crop_masks if self.cropping else full_real_mask,
        )

    def compute_loss(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Compute direct invariance plus SIGReg in native 8D encoder space."""
        if self.alignment_level == "protein":
            embeddings, view_masks = self.encode_views(batch)
            inv_loss = canonical_invariance_loss(embeddings)
            residue_embeddings = None
            real_mask = None
        else:
            (
                residue_embeddings,
                embeddings,
                view_masks,
                real_mask,
            ) = self.encode_residue_views(batch)
            inv_loss = canonical_residue_invariance_loss(
                residue_embeddings, real_mask
            )
        if self.sigreg_input == "protein_mean":
            sigreg_input = embeddings.permute(1, 0, 2)
        else:
            if residue_embeddings is None or real_mask is None:
                raise RuntimeError(
                    "protein-centred residue SIGReg requires residue embeddings"
                )
            if (
                self.residue_samples_per_protein == 1
                and not self.independent_residue_samples
            ):
                # Keep the historical default call and RNG path unchanged.
                sigreg_input, _ = protein_centered_residue_sigreg_input(
                    residue_embeddings,
                    real_mask,
                )
            else:
                sigreg_input, _ = protein_centered_residue_sigreg_input(
                    residue_embeddings,
                    real_mask,
                    samples_per_protein=self.residue_samples_per_protein,
                    independent_views=self.independent_residue_samples,
                )
        if self.sigreg_joint_views:
            sigreg_input = sigreg_input.reshape(
                1, -1, sigreg_input.shape[-1]
            )
        sigreg_loss = self.sigreg(sigreg_input)
        total_loss = inv_loss + self.sigreg_weight * sigreg_loss
        output = {
            "loss": total_loss,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "embeddings": embeddings,
            "view_masks": view_masks,
        }
        if real_mask is not None:
            output["alignment_masks"] = real_mask
        if residue_embeddings is not None:
            output["residue_embeddings"] = residue_embeddings
        return output

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        output = self.compute_loss(batch)
        embeddings = output["embeddings"].detach()
        view_masks = output["view_masks"]
        full_valid = batch["mask_dict"]["coords"][..., 0, 0].bool()
        valid = output.get("alignment_masks", full_valid)
        valid_counts = valid.sum(dim=-1)
        visible_counts = view_masks.sum(dim=2)
        visible_denominator = (
            valid_counts if valid.ndim == 3 else valid_counts[:, None]
        )
        visible_fractions = visible_counts / visible_denominator.clamp_min(1)

        if not self._first_batch_sanity_reported:
            local_masks = view_masks[:, 1:]
            identical_local_mask_pairs = sum(
                int((local_masks[:, left] == local_masks[:, right]).all(dim=1).sum())
                for left in range(CANONICAL_NUM_LOCAL_VIEWS)
                for right in range(left + 1, CANONICAL_NUM_LOCAL_VIEWS)
            )
            local_packed_max_lengths = [
                int(value) for value in visible_counts[:, 1:].max(dim=0).values.tolist()
            ]
            print(
                "Canonical LeJEPA first-batch sanity:\n"
                f"  packed_views={str(self.packed_views).lower()}, "
                "independent_view_rotations="
                f"{str(self.independent_view_rotations).lower()}, "
                f"cropping={str(self.cropping).lower()}, "
                f"shared_crop_anchor={str(self.shared_crop_anchor).lower()}, "
                "independent_residue_samples="
                f"{str(self.independent_residue_samples).lower()}, "
                f"residue_samples_per_protein={self.residue_samples_per_protein}, "
                f"actual_batch_size={full_valid.shape[0]}, num_views={view_masks.shape[1]}\n"
                "  valid_residue_count: "
                f"mean={full_valid.sum(dim=1).float().mean().item():.2f}, "
                f"min={int(full_valid.sum(dim=1).min())}, "
                f"max={int(full_valid.sum(dim=1).max())}\n"
                "  global: "
                f"mean_visible_count={visible_counts[:, 0].float().mean().item():.2f}, "
                f"mean_visible_fraction={visible_fractions[:, 0].mean().item():.6f}, "
                f"packed_max_length={int(visible_counts[:, 0].max())}\n"
                "  locals: "
                f"mean_visible_count={visible_counts[:, 1:].float().mean().item():.2f}, "
                f"mean_visible_fraction={visible_fractions[:, 1:].mean().item():.6f}, "
                f"packed_max_lengths={local_packed_max_lengths}\n"
                f"  exactly_identical_local_mask_pairs={identical_local_mask_pairs}",
                flush=True,
            )
            self._first_batch_sanity_reported = True

        local_cosine = F.cosine_similarity(
            embeddings[:, :1].expand_as(embeddings[:, 1:]),
            embeddings[:, 1:],
            dim=-1,
        ).mean()
        ranks = torch.stack(
            [covariance_effective_rank(embeddings[:, i]) for i in range(5)]
        )
        metrics = {
            "train/loss": output["loss"].detach(),
            "train/inv_loss": output["inv_loss"].detach(),
            "train/sigreg_loss": output["sigreg_loss"].detach(),
            "train/global_visible_fraction": visible_fractions[:, 0].mean(),
            "train/local_visible_fraction": visible_fractions[:, 1:].mean(),
            "train/embedding_std": embeddings.float().std(),
            "train/effective_rank": ranks.mean(),
            "train/global_local_cosine": local_cosine,
        }
        if self.alignment_level == "residue":
            residue_embeddings = output["residue_embeddings"].detach()
            residue_mask = (
                valid
                if valid.ndim == 3
                else valid[:, None, :].expand(-1, CANONICAL_NUM_VIEWS, -1)
            )
            residue_values = residue_embeddings[residue_mask]
            visibility_metrics = residue_invariance_visibility_mse(
                residue_embeddings, valid, view_masks
            )
            metrics.update(
                {
                    "train/residue_effective_rank": covariance_effective_rank(
                        residue_values
                    ),
                    "train/residue_embedding_std": residue_values.float().std(),
                    "train/inv_visible_visible": visibility_metrics["visible_visible"],
                    "train/inv_visible_masked": visibility_metrics["visible_masked"],
                    "train/inv_masked_visible": visibility_metrics["masked_visible"],
                    "train/inv_masked_masked": visibility_metrics["masked_masked"],
                }
            )
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True)
        return output["loss"]

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log one diagnostic L2 norm over all encoder gradients."""
        del optimizer
        squared_norms = [
            parameter.grad.detach().float().norm(2).square()
            for parameter in self.encoder.parameters()
            if parameter.grad is not None
        ]
        if squared_norms:
            gradient_norm = torch.stack(squared_norms).sum().sqrt()
            self.log(
                "train/gradient_norm",
                gradient_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        opt_cfg = _config_get(self.cfg, "optimizer")
        parameter_groups = build_adamw_parameter_groups(
            self.encoder, float(_config_get(opt_cfg, "weight_decay"))
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=float(_config_get(opt_cfg, "lr")),
            betas=tuple(float(x) for x in _config_get(opt_cfg, "betas")),
            amsgrad=bool(_config_get(opt_cfg, "amsgrad")),
        )

        schedule_cfg = _config_get(self.cfg, "schedule")
        scheduler = LinearWarmupCosineAnnealing(
            optimizer,
            total_steps=int(_config_get(schedule_cfg, "total_steps")),
            peak_step=int(_config_get(schedule_cfg, "peak_step")),
            start_factor=float(_config_get(schedule_cfg, "start_lr"))
            / float(_config_get(opt_cfg, "lr")),
            end_lr=float(_config_get(schedule_cfg, "end_lr")),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
