"""Tiny standalone test for random_residue_mask and apply_mask_to_latents.

Run with:  python -m proteinfoundation.jepa.tests.masking
"""

import torch

from proteinfoundation.jepa.masking import apply_mask_to_latents, random_residue_mask


def run(batch, masked_pct=0.5):
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [B, L]
    context_mask, target_mask = random_residue_mask(batch, masked_pct=masked_pct)

    # context and target never overlap, and together cover exactly the valid residues
    assert not (context_mask & target_mask).any()
    assert torch.equal(context_mask | target_mask, valid_res)

    for i in range(valid_res.shape[0]):
        n_valid = int(valid_res[i].sum())
        print(
            f"  protein {i}: valid={n_valid} "
            f"context={int(context_mask[i].sum())} target={int(target_mask[i].sum())}"
        )


# Scenario 1: single protein, all residues valid
B, L = 1, 128  # batch size, sequence length
batch = {
    "coords": torch.randn(B, L, 37, 3),
    "mask_dict": {"coords": torch.ones(B, L, 37, 3).bool()},  # all valid residues
}
print("scenario 1 (B=1, all valid):")
run(batch)

# Scenario 2: small batch, all residues valid
B, L = 4, 16
batch = {
    "coords": torch.randn(B, L, 37, 3),
    "mask_dict": {"coords": torch.ones(B, L, 37, 3).bool()},
}
print("scenario 2 (B=4, all valid):")
run(batch)

# Scenario 3: with padding -> padded residues must never be selected
B, L = 2, 10
valid = torch.ones(B, L, 37, 3).bool()
valid[0, 6:] = False  # protein 0: only 6 valid residues
valid[1, 8:] = False  # protein 1: only 8 valid residues
batch = {"coords": torch.randn(B, L, 37, 3), "mask_dict": {"coords": valid}}
print("scenario 3 (with padding):")
run(batch)


def run_apply(batch, latent_z_dim=8, masked_pct=0.5):
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [B, L]
    B, L = valid_res.shape
    context_mask, target_mask = random_residue_mask(batch, masked_pct=masked_pct)
    latents = torch.randn(B, L, latent_z_dim)  # stand-in for encoder mean
    out = apply_mask_to_latents(latents, context_mask, target_mask)

    L_ctx = out["context_latents"].shape[1]
    L_tgt = out["target_latents"].shape[1]

    # shapes and dtypes
    assert out["context_latents"].shape == (B, L_ctx, latent_z_dim)
    assert out["target_latents"].shape == (B, L_tgt, latent_z_dim)
    assert out["context_indices"].shape == (B, L_ctx) and out["context_indices"].dtype == torch.long
    assert out["target_indices"].shape == (B, L_tgt) and out["target_indices"].dtype == torch.long
    assert out["context_padding"].shape == (B, L_ctx)
    assert out["target_padding"].shape == (B, L_tgt)

    # L_ctx / L_tgt equal the batch-wide max count
    assert L_ctx == int(context_mask.sum(dim=1).max())
    assert L_tgt == int(target_mask.sum(dim=1).max())

    for i in range(B):
        c_real = torch.logical_not(out["context_padding"][i])
        t_real = torch.logical_not(out["target_padding"][i])

        # number of real (non-padding) entries matches the mask counts
        assert int(c_real.sum()) == int(context_mask[i].sum())
        assert int(t_real.sum()) == int(target_mask[i].sum())

        # gathered latents/indices match the originals at the recorded positions
        c_idx = out["context_indices"][i][c_real]
        t_idx = out["target_indices"][i][t_real]
        assert torch.equal(out["context_latents"][i][c_real], latents[i, c_idx])
        assert torch.equal(out["target_latents"][i][t_real], latents[i, t_idx])

        # recorded positions are actually context/target residues
        assert context_mask[i, c_idx].all()
        assert target_mask[i, t_idx].all()

        # padded latents are zeroed, padded indices are 0
        assert torch.equal(out["context_latents"][i][torch.logical_not(c_real)], torch.zeros(int((torch.logical_not(c_real)).sum()), latent_z_dim))
        assert (out["context_indices"][i][torch.logical_not(c_real)] == 0).all()

        print(f"  protein {i}: context={int(c_real.sum())}/{L_ctx} target={int(t_real.sum())}/{L_tgt}")


# Scenario 4: apply_mask_to_latents with padding -> ragged context/target counts
B, L = 2, 10
valid = torch.ones(B, L, 37, 3).bool()
valid[0, 6:] = False  # protein 0: 6 valid -> 3 context / 3 target
valid[1, 8:] = False  # protein 1: 8 valid -> 4 context / 4 target
batch = {"coords": torch.randn(B, L, 37, 3), "mask_dict": {"coords": valid}}
print("scenario 4 (apply_mask_to_latents, ragged):")
run_apply(batch)

print("all checks passed")
