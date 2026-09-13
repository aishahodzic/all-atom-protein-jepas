"""Small deterministic fixtures for canonical-LeJEPA unit tests."""

from __future__ import annotations

import copy

import torch
from omegaconf import OmegaConf


def tiny_canonical_config(
    *,
    num_proj: int = 8,
    packed_views: bool = False,
    alignment_level: str | None = "protein",
    masked_kv_mode: str | None = "visible_only",
    residue_view_checkpointing: bool = False,
    residue_compact_visible_kv: bool = False,
    sigreg_input: str | None = "protein_mean",
    sigreg_joint_views: bool = False,
    residue_samples_per_protein: int = 1,
    independent_residue_samples: bool = False,
    cropping: bool = False,
    crop_length: int = 8,
    global_crop_length: int | None = None,
    local_crop_length: int | None = None,
    local_mask_pct: float = 0.80,
    global_crop_fraction_min: float | None = None,
    global_crop_fraction_max: float | None = None,
    global_crop_min_length: int = 48,
    global_crop_max_length: int | None = None,
    locals_within_global: bool = False,
    shared_crop_anchor: bool = True,
):
    config = {
        "nn_ae": {
            "latent_z_dim": 8,
            "encoder": {
                "nlayers": 1,
                "token_dim": 16,
                "nheads": 4,
                "parallel_mha_transition": False,
                "strict_feats": False,
                "feats_seq": [
                    "chain_break_per_res",
                    "x1_aatype",
                    "x1_a37coors_nm",
                    "x1_a37coors_nm_rel",
                    "chain_idx_seq",
                ],
                "feats_cond_seq": None,
                "dim_cond": 128,
                "idx_emb_dim": 8,
                "feats_pair_repr": [
                    "rel_seq_sep",
                    "x1_bb_pair_dists_nm",
                    "x1_bb_pair_orientation",
                    "chain_idx_pair",
                ],
                "seq_sep_dim": 7,
                "pair_repr_dim": 8,
                "update_pair_repr": False,
                "update_pair_repr_every_n": 1,
                "use_tri_mult": False,
                "use_qkln": True,
                "normalize_latent": False,
                "latent_z_dim": 8,
            },
            "decoder": {"latent_z_dim": 8},
        },
        "views": {
            "num_local_views": 4,
            "global_mask_pct": 0.20,
            "local_mask_pct": local_mask_pct,
            "cropping": cropping,
            "crop_length": crop_length,
            "global_crop_fraction_min": global_crop_fraction_min,
            "global_crop_fraction_max": global_crop_fraction_max,
            "global_crop_min_length": global_crop_min_length,
            "global_crop_max_length": global_crop_max_length,
            "locals_within_global": locals_within_global,
            "shared_crop_anchor": shared_crop_anchor,
            "rotation": False,
            "physical_token_drop": False,
            "packed_views": packed_views,
        },
        "readout": {"type": "mean_visible_residues"},
        "residue_view_checkpointing": residue_view_checkpointing,
        "residue_compact_visible_kv": residue_compact_visible_kv,
        "sigreg": {
            "weight": 0.02,
            "knots": 5,
            "num_proj": num_proj,
            "normalize_by_n": False,
            "joint_views": sigreg_joint_views,
            "residue_samples_per_protein": residue_samples_per_protein,
            "independent_residue_samples": independent_residue_samples,
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 4e-4,
            "weight_decay": 0.04,
            "betas": [0.9, 0.95],
            "amsgrad": False,
            "exclude_bias_norm": True,
        },
        "schedule": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": 100_000,
            "peak_step": 12_000,
            "start_lr": 1e-4,
            "end_lr": 4e-4,
        },
    }
    if global_crop_length is not None:
        config["views"]["global_crop_length"] = global_crop_length
    if local_crop_length is not None:
        config["views"]["local_crop_length"] = local_crop_length
    if alignment_level is not None:
        config["alignment_level"] = alignment_level
    if masked_kv_mode is not None:
        config["masked_kv_mode"] = masked_kv_mode
    if sigreg_input is not None:
        config["sigreg"]["input"] = sigreg_input
    return OmegaConf.create(config)


def synthetic_encoder_batch(
    *, batch_size: int = 3, length: int = 12
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1729)
    coords_nm = torch.randn(batch_size, length, 37, 3, generator=generator)
    coords = coords_nm * 10.0
    valid = torch.ones(batch_size, length, dtype=torch.bool)
    for row in range(batch_size):
        padding = min(row, length - 1)
        if padding:
            valid[row, length - padding :] = False
    coord_mask = valid[..., None].expand(-1, -1, 37).clone()
    coords_nm = coords_nm * coord_mask[..., None]
    coords = coords * coord_mask[..., None]
    residue_type = torch.randint(0, 20, (batch_size, length), generator=generator)
    residue_type = residue_type.masked_fill(~valid, 0)
    residue_pdb_idx = torch.arange(length).expand(batch_size, -1).clone()
    chains = torch.zeros(batch_size, length, dtype=torch.long)
    chain_breaks_per_residue = torch.zeros(batch_size, length, dtype=torch.bool)
    mask_coords = coord_mask[..., None].expand(-1, -1, -1, 3).clone()
    return {
        "coords": coords,
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "residue_pdb_idx": residue_pdb_idx,
        "chains": chains,
        "chain_breaks_per_residue": chain_breaks_per_residue,
        "mask_dict": {
            "coords": mask_coords,
            "coords_nm": mask_coords.clone(),
            "coord_mask": coord_mask.clone(),
            "residue_type": valid.clone(),
            "residue_pdb_idx": valid.clone(),
            "chains": valid.clone(),
            "chain_breaks_per_residue": valid.clone(),
        },
    }


def clone_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            cloned[key] = {
                nested_key: nested_value.clone()
                for nested_key, nested_value in value.items()
            }
        elif isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned
