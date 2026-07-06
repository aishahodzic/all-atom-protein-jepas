import torch


def random_residue_mask(batch, masked_pct=0.5):
    """Split each protein's valid residues into context and target sets for JEPA.

    For every protein we randomly assign ``masked_pct`` of its valid (non-padding)
    residues to the target (masked) set and the remainder to the context set. The
    predictor later uses the context residues' latents to predict the target
    residues' latents.

    Returns two boolean masks over residues, each [n_proteins, n_residues]. They are
    mutually exclusive and only ever True on valid residues (padding is always False).
    """
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [n_proteins, n_residues]
    n_proteins, n_residues = valid_res.shape

    context_mask = torch.zeros_like(valid_res)  # [n_proteins, n_residues]
    target_mask = torch.zeros_like(valid_res)   # [n_proteins, n_residues]

    # different random context/target split per protein in the batch
    for i in range(n_proteins):
        res_idx = torch.where(valid_res[i])[0]  # [n_valid_i] indices of this protein's valid residues
        n_valid = len(res_idx)

        # permute this protein's valid residue indices to randomise the split
        perm = res_idx[torch.randperm(n_valid, device=valid_res.device)]  # [n_valid_i]

        n_target = int(n_valid * masked_pct)
        target_mask[i, perm[:n_target]] = True    # first n_target residues -> target (masked)
        context_mask[i, perm[n_target:]] = True   # remaining residues -> context

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
        context_padding_mask: [B, L_ctx] bool           True where context is padding
        target_padding_mask:  [B, L_tgt] bool           True where target is padding
    """

    B, L, latent_z_dim = latents.shape
    device = latents.device

    # gather each protein's context/target latents and indices (variable length per protein)
    context_lat_list, context_idx_list = [], []
    target_lat_list, target_idx_list = [], []
    for i in range(B):  # loop only over the batch dimension
        c_idx = torch.where(context_mask[i])[0]  # [n_ctx_i] context residue positions
        t_idx = torch.where(target_mask[i])[0]   # [n_tgt_i] target residue positions
        context_lat_list.append(latents[i, c_idx])  # [n_ctx_i, latent_z_dim]
        target_lat_list.append(latents[i, t_idx])   # [n_tgt_i, latent_z_dim]
        context_idx_list.append(c_idx)              # [n_ctx_i]
        target_idx_list.append(t_idx)               # [n_tgt_i]

    # pad to the batch-wide maximum count so the sets can be stacked into one tensor
    L_ctx = max(idx.numel() for idx in context_idx_list)  # max context residues across batch
    L_tgt = max(idx.numel() for idx in target_idx_list)   # max target residues across batch

    context_latents = torch.zeros(B, L_ctx, latent_z_dim, device=device, dtype=latents.dtype)
    target_latents = torch.zeros(B, L_tgt, latent_z_dim, device=device, dtype=latents.dtype)
    context_indices = torch.zeros(B, L_ctx, device=device, dtype=torch.long)  # pad with 0
    target_indices = torch.zeros(B, L_tgt, device=device, dtype=torch.long)   # pad with 0
    context_padding_mask = torch.ones(B, L_ctx, device=device, dtype=torch.bool)  # True = padding
    target_padding_mask = torch.ones(B, L_tgt, device=device, dtype=torch.bool)   # True = padding

    for i in range(B):  # loop only over the batch dimension
        n_ctx = context_idx_list[i].numel()
        n_tgt = target_idx_list[i].numel()
        context_latents[i, :n_ctx] = context_lat_list[i]   # [n_ctx_i, latent_z_dim]
        target_latents[i, :n_tgt] = target_lat_list[i]     # [n_tgt_i, latent_z_dim]
        context_indices[i, :n_ctx] = context_idx_list[i]   # [n_ctx_i]
        target_indices[i, :n_tgt] = target_idx_list[i]     # [n_tgt_i]
        context_padding_mask[i, :n_ctx] = False            # real entries are not padding
        target_padding_mask[i, :n_tgt] = False

    return {
        "context_latents": context_latents,            # [B, L_ctx, latent_z_dim]
        "context_indices": context_indices,            # [B, L_ctx]
        "target_latents": target_latents,              # [B, L_tgt, latent_z_dim]
        "target_indices": target_indices,              # [B, L_tgt]
        "context_padding_mask": context_padding_mask,  # [B, L_ctx]
        "target_padding_mask": target_padding_mask,    # [B, L_tgt]
    }
