import warnings

import torch


def mask_partial_atom_coordinates(batch, mask_pct=0.5):
    """Hide a fixed fraction of each residue's observed atom coordinates.

    Amino-acid identity and residue validity are preserved. Every observed atom
    (backbone and side chain, including C-alpha) is equally eligible, and the
    draw is independent for each residue. Hidden coordinate values are zeroed in
    addition to clearing ``coord_mask`` so an accidentally ungated coordinate
    feature cannot leak them.

    The returned batch is a shallow copy with cloned coordinate tensors. The
    input batch and ``mask_dict`` are not modified; keeping ``mask_dict['coords']``
    intact is what lets all real residues remain encoder tokens and JEPA targets.
    """
    if not 0.0 <= mask_pct <= 1.0:
        raise ValueError(f"mask_pct must be in [0, 1], got {mask_pct}")
    if "coord_mask" not in batch:
        raise KeyError("partial-coordinate masking requires batch['coord_mask']")

    coord_mask = batch["coord_mask"].bool()  # [B, L, 37]
    if coord_mask.ndim != 3:
        raise ValueError("batch['coord_mask'] must have shape [B, L, A]")

    n_observed = coord_mask.sum(dim=-1, keepdim=True)
    n_hide = (n_observed * mask_pct).round().long()
    scores = torch.rand(coord_mask.shape, device=coord_mask.device)
    scores = scores.masked_fill(~coord_mask, 2.0)
    ranks = scores.argsort(dim=-1).argsort(dim=-1)
    hidden = coord_mask & (ranks < n_hide)
    visible = coord_mask & ~hidden

    masked_batch = dict(batch)
    masked_batch["coord_mask"] = visible
    for key in ("coords", "coords_nm"):
        if key in batch:
            masked_batch[key] = batch[key] * visible[..., None].to(batch[key].dtype)

    # Some data pipelines cache CA coordinates separately. Keep that cache
    # consistent with atom-37 slot 1 so hiding CA cannot leak through it.
    if "ca_coors_nm" in batch:
        masked_batch["ca_coors_nm"] = (
            batch["ca_coors_nm"] * visible[..., 1, None].to(batch["ca_coors_nm"].dtype)
        )

    return masked_batch


def random_residue_mask(batch, masked_pct=0.5):
    """Split each protein's valid residues into context and target sets for JEPA.

    For every protein we randomly assign ``masked_pct`` of its valid
    residues to the target set and the remainder to the context set. 

    Returns two boolean masks over residues, each [n_proteins, n_residues]. They are
    mutually exclusive and only ever True on valid residues (padding is always False).
    """
    # take valid residues - residues that are not padding
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [n_proteins, n_residues]
    n_proteins, n_residues = valid_res.shape

    context_mask = torch.zeros_like(valid_res)  # [n_proteins, n_residues]
    target_mask = torch.zeros_like(valid_res)   # [n_proteins, n_residues]

    # for each protein, permute the residue indices and split into context and target sets
    for i in range(n_proteins):
        res_idx = torch.where(valid_res[i])[0]  # [n_valid_i] indices of this protein's valid residues
        n_valid = len(res_idx)

        # permute this protein's valid residue indices to randomise the split
        perm = res_idx[torch.randperm(n_valid, device=valid_res.device)]  # [n_valid_i]

        #take x% of valid residues as target, the rest as context
        n_target = int(n_valid * masked_pct)
        target_mask[i, perm[:n_target]] = True    # first n_target residues -> target (masked)
        context_mask[i, perm[n_target:]] = True   # remaining residues -> context

    return context_mask, target_mask


def block_residue_mask(batch, num_blocks=4, min_block_ratio=0.15, max_block_ratio=0.20):
    """Splits each protein's valid residues into ``num_blocks`` equal regions and
    samples one contiguous block per region, each with a length in
    ``[min_block_ratio, max_block_ratio]`` of the protein's valid residue count.

    Returns two boolean masks over residues, each [n_proteins, n_residues]. They are
    mutually exclusive and only ever True on valid residues (padding is always False).
    """
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [n_proteins, n_residues]
    n_proteins, n_residues = valid_res.shape

    context_mask = torch.zeros_like(valid_res)  # [n_proteins, n_residues]
    target_mask = torch.zeros_like(valid_res)   # [n_proteins, n_residues]

    for i in range(n_proteins):
        res_idx = torch.where(valid_res[i])[0]  # [n_valid_i] indices of this protein's valid residues
        n_valid = len(res_idx)

        min_len = max(1, int(min_block_ratio * n_valid))
        max_len = max(min_len, int(max_block_ratio * n_valid))

        block_mask_local = torch.zeros(n_valid, dtype=torch.bool, device=valid_res.device)  # [n_valid_i]
        region_size = n_valid / num_blocks
        for b in range(num_blocks if n_valid > 0 else 0):
            region_start = int(b * region_size)
            region_end = int((b + 1) * region_size) if b < num_blocks - 1 else n_valid
            region_len = region_end - region_start
            if region_len <= 0:
                continue
            block_len = min(
                torch.randint(min_len, max_len + 1, (1,), device=valid_res.device).item(), region_len
            )
            max_offset = region_len - block_len
            offset = torch.randint(0, max_offset + 1, (1,), device=valid_res.device).item()
            start = region_start + offset
            block_mask_local[start : start + block_len] = True

        target_idx = res_idx[block_mask_local]   # [n_target_i] residue positions covered by a block
        context_idx = res_idx[~block_mask_local]  # [n_context_i] remaining valid residues

        target_mask[i, target_idx] = True
        context_mask[i, context_idx] = True

    return context_mask, target_mask


def _spatial_3d_block_mask_for_single_protein(
    ca_coords_nm,
    num_seeds=4,
    min_seed_distance_nm=1.5,
    target_fraction=0.75,
    min_fraction=0.60,
    max_fraction=0.85,
    max_radius_nm=3.0,
):
    """Build one spatial target mask and return it with probe metadata."""
    n_valid = ca_coords_nm.shape[0]
    if n_valid < 2:
        raise ValueError("spatial_3d_block_mask requires at least two valid residues")

    # Very small proteins use two anchors. Larger proteins try four, reducing the
    # count only when the 15 A separation constraint cannot be satisfied.
    desired_seeds = min(2 if n_valid < 20 else num_seeds, n_valid - 1)
    pairwise = torch.cdist(ca_coords_nm.float(), ca_coords_nm.float())

    best_seed_idx = None
    n_trials = min(16, n_valid)
    starts = torch.randperm(n_valid, device=ca_coords_nm.device)[:n_trials]
    for first in starts:
        selected = [first]
        min_distance = pairwise[first].clone()
        while len(selected) < desired_seeds:
            eligible = min_distance >= min_seed_distance_nm
            eligible[torch.stack(selected)] = False
            if not eligible.any():
                break
            # Farthest-point sampling keeps the spheres spatially distributed.
            next_idx = torch.where(eligible, min_distance, -torch.ones_like(min_distance)).argmax()
            selected.append(next_idx)
            min_distance = torch.minimum(min_distance, pairwise[next_idx])
        candidate = torch.stack(selected)
        if best_seed_idx is None or candidate.numel() > best_seed_idx.numel():
            best_seed_idx = candidate
        if best_seed_idx.numel() == desired_seeds:
            break

    separation_relaxed = False
    if best_seed_idx.numel() < 2:
        # No pair satisfies 15 A. Use the farthest pair so the split remains usable.
        flat_idx = pairwise.argmax()
        first = torch.div(flat_idx, n_valid, rounding_mode="floor")
        second = flat_idx.remainder(n_valid)
        best_seed_idx = torch.stack((first, second))
        separation_relaxed = True
        warnings.warn(
            "spatial_3d_block_mask could not place two seeds 15 A apart; "
            "using the farthest available pair",
            RuntimeWarning,
            stacklevel=2,
        )
    elif best_seed_idx.numel() < desired_seeds:
        warnings.warn(
            "spatial_3d_block_mask reduced the number of seeds because four "
            "15 A-separated seeds did not fit",
            RuntimeWarning,
            stacklevel=2,
        )

    seed_coords = ca_coords_nm[best_seed_idx]
    nearest_seed_distance = torch.cdist(
        ca_coords_nm.float(), seed_coords.float()
    ).min(dim=1).values

    # Every distinct nearest-seed distance is a radius at which coverage changes.
    # Choose the radius closest to 75% among candidates respecting the 60--85%
    # band. If a 30 A cap prevents 60%, take the largest attainable coverage.
    candidate_radii = torch.unique(
        torch.cat(
            (
                nearest_seed_distance[nearest_seed_distance <= max_radius_nm],
                nearest_seed_distance.new_tensor([max_radius_nm]),
            )
        ),
        sorted=True,
    )
    fractions = torch.stack(
        [(nearest_seed_distance <= radius).float().mean() for radius in candidate_radii]
    )
    in_band = (fractions >= min_fraction) & (fractions <= max_fraction)
    if in_band.any():
        candidate_ids = torch.where(in_band)[0]
        chosen = candidate_ids[(fractions[candidate_ids] - target_fraction).abs().argmin()]
    else:
        below_cap = torch.where(fractions <= max_fraction)[0]
        if below_cap.numel() > 0:
            chosen = below_cap[fractions[below_cap].argmax()]
        else:
            chosen = fractions.argmin()

    radius_nm = candidate_radii[chosen]
    target_local = nearest_seed_distance <= radius_nm
    seed_distances = pairwise[best_seed_idx][:, best_seed_idx]
    upper_triangle = torch.triu_indices(
        best_seed_idx.numel(), best_seed_idx.numel(), offset=1, device=ca_coords_nm.device
    )
    pair_distances = seed_distances[upper_triangle[0], upper_triangle[1]]
    metadata = {
        "seed_indices": best_seed_idx,
        "seed_coords_nm": seed_coords,
        "seed_pair_distances_nm": pair_distances,
        "radius_nm": radius_nm,
        "masked_fraction": target_local.float().mean(),
        "separation_relaxed": separation_relaxed,
    }
    return target_local, metadata


def spatial_3d_block_mask(batch):
    """Mask spatially coherent residue neighbourhoods around spread C-alpha seeds.

    Each protein independently selects up to four seeds separated by at least 15 A
    and adapts a shared sphere radius (capped at 30 A) toward 75% target coverage.
    Context and target are a disjoint partition of the valid residues.

    Returns boolean ``context_mask`` and ``target_mask`` tensors with shape [B, L],
    matching :func:`random_residue_mask`.
    """
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()
    if "ca_coors_nm" in batch:
        ca_coords_nm = batch["ca_coors_nm"]
    elif "coords_nm" in batch:
        ca_coords_nm = batch["coords_nm"][:, :, 1, :]
    else:
        raise KeyError("spatial_3d_block_mask requires 'ca_coors_nm' or 'coords_nm'")

    if ca_coords_nm.shape[:2] != valid_res.shape or ca_coords_nm.shape[-1] != 3:
        raise ValueError("C-alpha coordinates must have shape [B, L, 3]")

    context_mask = torch.zeros_like(valid_res)
    target_mask = torch.zeros_like(valid_res)
    for protein_idx in range(valid_res.shape[0]):
        residue_idx = torch.where(valid_res[protein_idx])[0]
        if residue_idx.numel() < 2:
            # A rare malformed crop can retain only one valid residue despite the
            # dataset length filter. Keep it as context-only so it contributes no
            # ill-posed spatial target while the other proteins in the batch train.
            context_mask[protein_idx, residue_idx] = True
            warnings.warn(
                "spatial_3d_block_mask kept a protein with fewer than two valid "
                "residues as context-only",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        coords = ca_coords_nm[protein_idx, residue_idx]
        if not torch.isfinite(coords).all():
            raise ValueError("valid residues contain non-finite C-alpha coordinates")
        target_local, _ = _spatial_3d_block_mask_for_single_protein(coords)
        target_idx = residue_idx[target_local]
        context_idx = residue_idx[~target_local]
        target_mask[protein_idx, target_idx] = True
        context_mask[protein_idx, context_idx] = True

    return context_mask, target_mask


def apply_mask_to_latents(latents, context_mask, target_mask):
    """
    Latents stay [B, L, latent_z_dim] in shape but we select the subset where mask is True.
    The rest are zeroed out and marked in the padding mask so attention ignores them.

    Inputs:
        latents:      [B, L, latent_z_dim]  encoder mean output for all residues
        context_mask: [B, L] bool           True for context residues
        target_mask:  [B, L] bool           True for target residues

    Returns a dict with:
        context_latents:      [B, L_ctx, latent_z_dim]  padded context latents
        context_indices:      [B, L_ctx] long           context residue positions (pad=0)
        target_latents:       [B, L_tgt, latent_z_dim]  padded target latents
        target_indices:       [B, L_tgt] long           target residue positions (pad=0)
        context_padding: [B, L_ctx] bool           True where context is padding
        target_padding:  [B, L_tgt] bool           True where target is padding
    """

    B, L, latent_z_dim = latents.shape
    device = latents.device

    #Variable lengths: each protein has a different num. of context/target residues
    # gather each protein's context/target latents and indices
    context_lat_list, context_idx_list = [], []
    target_lat_list, target_idx_list = [], []
    for i in range(B):  # loop only over the batch dimension
        c_idx = torch.where(context_mask[i])[0]  # [n_ctx_i] context residue positions
        t_idx = torch.where(target_mask[i])[0]   # [n_tgt_i] target residue positions
        context_lat_list.append(latents[i, c_idx])  # [n_ctx_i, latent_z_dim]
        target_lat_list.append(latents[i, t_idx])   # [n_tgt_i, latent_z_dim]
        context_idx_list.append(c_idx)              # [n_ctx_i]
        target_idx_list.append(t_idx)               # [n_tgt_i]

    # Compute max num el. of each residue
    # then pad to the batch's maximum count so the residues can be stacked into one tensor
    L_ctx = max(idx.numel() for idx in context_idx_list)  # max context residues across batch
    L_tgt = max(idx.numel() for idx in target_idx_list)   # max target residues across batch

    context_latents = torch.zeros(B, L_ctx, latent_z_dim, device=device, dtype=latents.dtype)
    target_latents = torch.zeros(B, L_tgt, latent_z_dim, device=device, dtype=latents.dtype)
    context_indices = torch.zeros(B, L_ctx, device=device, dtype=torch.long)  # pad with 0
    target_indices = torch.zeros(B, L_tgt, device=device, dtype=torch.long)   # pad with 0
    context_padding = torch.ones(B, L_ctx, device=device, dtype=torch.bool)  # True = padding
    target_padding = torch.ones(B, L_tgt, device=device, dtype=torch.bool)   # True = padding

    #Fixed length (with padding) - populate the tensors with valid latents + padding
    for i in range(B):  # loop only over the batch dimension
        n_ctx = context_idx_list[i].numel()
        n_tgt = target_idx_list[i].numel()
        context_latents[i, :n_ctx] = context_lat_list[i]   # [n_ctx_i, latent_z_dim]
        target_latents[i, :n_tgt] = target_lat_list[i]     # [n_tgt_i, latent_z_dim]
        context_indices[i, :n_ctx] = context_idx_list[i]   # [n_ctx_i]
        target_indices[i, :n_tgt] = target_idx_list[i]     # [n_tgt_i]
        context_padding[i, :n_ctx] = False            # real entries are not padding
        target_padding[i, :n_tgt] = False

    return {
        "context_latents": context_latents,            # [B, L_ctx, latent_z_dim]
        "context_indices": context_indices,            # [B, L_ctx]
        "target_latents": target_latents,              # [B, L_tgt, latent_z_dim]
        "target_indices": target_indices,              # [B, L_tgt]
        "context_padding": context_padding,  # [B, L_ctx]
        "target_padding": target_padding,    # [B, L_tgt]
    }
