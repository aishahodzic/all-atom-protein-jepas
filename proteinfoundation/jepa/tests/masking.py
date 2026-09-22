"""Tiny standalone test for random_residue_mask and apply_mask_to_latents.

Run with:  python -m proteinfoundation.jepa.tests.masking
"""

import torch

from proteinfoundation.jepa.masking import (
    apply_mask_to_latents,
    block_residue_mask,
    random_residue_mask,
)


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

def run_block(batch, num_blocks=4, min_block_ratio=0.15, max_block_ratio=0.20):
    valid_res = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [B, L]
    context_mask, target_mask = block_residue_mask(
        batch,
        num_blocks=num_blocks,
        min_block_ratio=min_block_ratio,
        max_block_ratio=max_block_ratio,
    )

    # context and target never overlap, and together cover exactly the valid residues
    assert not (context_mask & target_mask).any()
    assert torch.equal(context_mask | target_mask, valid_res)

    for i in range(valid_res.shape[0]):
        n_valid = int(valid_res[i].sum())
        tgt_idx = torch.where(target_mask[i])[0]

        # no target residue past the last valid one for this protein
        assert tgt_idx.numel() == 0 or int(tgt_idx.max()) < n_valid

        print(
            f"  protein {i}: valid={n_valid} "
            f"context={int(context_mask[i].sum())} target={int(target_mask[i].sum())}"
        )


# Scenario 5: single protein, all residues valid -> block masking
B, L = 1, 128  # batch size, sequence length
batch = {
    "coords": torch.randn(B, L, 37, 3),
    "mask_dict": {"coords": torch.ones(B, L, 37, 3).bool()},
}
print("scenario 5 (block masking, B=1, all valid):")
run_block(batch)

B, L = 3, 20
valid = torch.ones(B, L, 37, 3).bool()
valid[0, 12:] = False  # protein 0: 12 valid residues
valid[1, 5:] = False  # protein 1: 5 valid residues, small enough to force edge clipping
valid[2, :] = False  # protein 2: fully padded
batch = {"coords": torch.randn(B, L, 37, 3), "mask_dict": {"coords": valid}}
print("scenario 6 (block masking, with padding incl. fully-padded protein):")
run_block(batch)

# Scenario 7: stratified structure -> at most num_blocks distinct masked runs
B, L = 20, 200
lengths = torch.randint(50, L + 1, (B,))
valid = torch.zeros(B, L, 37, 3).bool()
for i, n in enumerate(lengths):
    valid[i, :n] = True
batch = {"coords": torch.randn(B, L, 37, 3), "mask_dict": {"coords": valid}}
print("scenario 7 (stratified structure, num_blocks upper-bounds run count):")
num_blocks = 4
context_mask, target_mask = block_residue_mask(batch, num_blocks=num_blocks)
for i in range(B):
    n = int(lengths[i])
    row = target_mask[i, :n]
    runs = int(((row[1:] & ~row[:-1]).sum() + row[0]).item()) if n > 0 else 0
    assert runs <= num_blocks, f"protein {i}: got {runs} masked runs, expected <= {num_blocks}"
print(f"  all {B} proteins had <= {num_blocks} masked runs")

# Scenario 8: n_valid < num_blocks -> some regions are empty, must not crash
B, L = 3, 10
valid = torch.zeros(B, L, 37, 3).bool()
valid[0, :2] = True   # 2 valid residues, fewer than num_blocks=4
valid[1, :1] = True   # 1 valid residue
valid[2, :4] = True   # exactly num_blocks valid residues
batch = {"coords": torch.randn(B, L, 37, 3), "mask_dict": {"coords": valid}}
print("scenario 8 (n_valid < num_blocks):")
run_block(batch)

# Scenario 9: coverage sanity check over a larger, varied-length batch
B, L = 100, 256
lengths = torch.randint(50, L + 1, (B,))
valid = torch.zeros(B, L, 37, 3).bool()
for i, n in enumerate(lengths):
    valid[i, :n] = True
batch = {"coords": torch.randn(B, L, 37, 3), "mask_dict": {"coords": valid}}
context_mask, target_mask = block_residue_mask(batch)
coverages = [
    int(target_mask[i, : int(lengths[i])].sum()) / int(lengths[i]) for i in range(B)
]
avg_coverage = sum(coverages) / len(coverages)
print(f"scenario 9 (coverage sanity, B={B}): avg_coverage={avg_coverage:.3f}")
assert 0.5 < avg_coverage < 0.85, f"average coverage {avg_coverage:.3f} outside expected range"

print("all checks passed")
