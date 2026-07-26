import torch


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


def block_residue_mask(batch, num_blocks=4, min_block_ratio=0.15, max_block_ratio=0.30):
    """Samples ``num_blocks`` independent contiguous blocks per protein, each with a
    length in ``[min_block_ratio, max_block_ratio]`` of that protein's valid residue
    count, at a random start position clipped at the sequence boundary (so edge
    blocks can end up shorter than sampled). Blocks may overlap; their union is the
    target set.

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
        for _ in range(num_blocks if n_valid > 0 else 0):
            block_len = torch.randint(min_len, max_len + 1, (1,), device=valid_res.device).item()
            start = torch.randint(0, n_valid, (1,), device=valid_res.device).item()
            end = min(start + block_len, n_valid)  # clip at boundary rather than constrain start
            block_mask_local[start:end] = True

        target_idx = res_idx[block_mask_local]   # [n_target_i] residue positions covered by a block
        context_idx = res_idx[~block_mask_local]  # [n_context_i] remaining valid residues

        target_mask[i, target_idx] = True
        context_mask[i, context_idx] = True

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
