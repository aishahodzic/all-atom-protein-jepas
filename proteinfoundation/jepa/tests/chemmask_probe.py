"""One-protein probe for mask_batch_for_encoder_chemmask_aa_only (chemical masking
scheme 1: "AA only, no coords").

Every real residue's AA identity reaches the context encoder, context and target
alike. Only geometry is masked, and only on target (non-context) residues, via
coord_mask -- i.e. the masking propagates through coord_mask, not by relying on
zeroed coordinates (target residues here keep real, nonzero coordinate values in
"coords"/"coords_nm"; only coord_mask signals absence, so any feature that fails
to gate on it will show up as a nonzero leak below).

Run with:  python -m proteinfoundation.jepa.tests.chemmask_probe
"""

import torch

from proteinfoundation.jepa.jepa_model import mask_batch_for_encoder_chemmask_aa_only
from proteinfoundation.jepa.masking import random_residue_mask
from proteinfoundation.nn.feature_factory import (
    Atom37NanometersCoorsSeqFeat,
    BackbonePairDistancesNanometerPairFeat,
    RelativeResidueOrientationPairFeat,
    ResidueTypeSeqFeat,
)

torch.manual_seed(0)

B, L, A = 1, 40, 37
coords = torch.randn(B, L, A, 3) * 2.0 + 5.0  # away from origin, so [0,0,0] can't be confused with real data
coords_nm = coords / 10.0
coord_mask = torch.ones(B, L, A, dtype=torch.bool)  # every atom present in the raw structure
residue_type = torch.randint(0, 20, (B, L))
valid = torch.ones(B, L, A, 3, dtype=torch.bool)
valid_res_type = torch.ones(B, L, dtype=torch.bool)

batch = {
    "coords": coords,
    "coords_nm": coords_nm,
    "coord_mask": coord_mask,
    "residue_type": residue_type,
    "mask_dict": {"coords": valid, "residue_type": valid_res_type},
}

context_mask, target_mask = random_residue_mask(batch, masked_pct=0.7)
assert context_mask.any() and target_mask.any(), "need both context and target residues for a meaningful probe"

masked_batch = mask_batch_for_encoder_chemmask_aa_only(batch, context_mask)

ctx = context_mask[0]
tgt = target_mask[0]

# --- sanity: coord_mask stripped exactly on TARGET residues; context untouched ---
assert (masked_batch["coord_mask"][0][tgt] == False).all(), "target residues must have coord_mask fully False"
assert (masked_batch["coord_mask"][0][ctx] == coord_mask[0][ctx]).all(), "context coord_mask must be untouched"
assert torch.equal(masked_batch["coords"][0], coords[0]), "coords must be left as real values everywhere (masking is via coord_mask, not zeroing)"
assert masked_batch["mask_dict"]["coords"] is batch["mask_dict"]["coords"], "mask_dict['coords'] must be untouched -- this is what protects AA identity for target residues"
print(f"context={int(ctx.sum())} target={int(tgt.sum())} residues; coord_mask/coords/mask_dict sanity checks passed")

# --- geometry features must be exactly zero for TARGET residues, present for context ---
atom_feat = Atom37NanometersCoorsSeqFeat()(masked_batch)  # [B, L, 37*4]
atom_feat_rel = Atom37NanometersCoorsSeqFeat(rel=True)(masked_batch)  # [B, L, 37*4]
pair_dist_feat = BackbonePairDistancesNanometerPairFeat()(masked_batch)  # [B, L, L, 84]
pair_orient_feat = RelativeResidueOrientationPairFeat()(masked_batch)  # [B, L, L, 105]

assert torch.equal(atom_feat[0][tgt], torch.zeros_like(atom_feat[0][tgt])), "Atom37NanometersCoorsSeqFeat leaked geometry on a target residue"
assert torch.equal(atom_feat_rel[0][tgt], torch.zeros_like(atom_feat_rel[0][tgt])), "Atom37NanometersCoorsSeqFeat(rel=True) leaked geometry on a target residue"
assert not torch.equal(atom_feat[0][ctx], torch.zeros_like(atom_feat[0][ctx])), "Atom37NanometersCoorsSeqFeat wrongly zeroed a context residue"

tgt_pair = tgt[:, None] | tgt[None, :]  # [L, L]
ctx_only_pair = ctx[:, None] & ctx[None, :]  # [L, L], both endpoints context
assert torch.equal(pair_dist_feat[0][tgt_pair], torch.zeros_like(pair_dist_feat[0][tgt_pair])), "BackbonePairDistancesNanometerPairFeat leaked geometry on a target residue pair"
assert torch.equal(pair_orient_feat[0][tgt_pair], torch.zeros_like(pair_orient_feat[0][tgt_pair])), "RelativeResidueOrientationPairFeat leaked geometry on a target residue pair"
assert not torch.equal(pair_dist_feat[0][ctx_only_pair], torch.zeros_like(pair_dist_feat[0][ctx_only_pair])), "BackbonePairDistancesNanometerPairFeat wrongly zeroed a context-context pair"
print("geometry features (atom + pair) are exactly zero for target residues, present for context residues")

aa_feat_raw = ResidueTypeSeqFeat()(masked_batch)  # [B, L, 20]
final_mask = masked_batch["mask_dict"]["coords"][..., 0, 0]  # [B, L], True for every real (non-padding) residue
aa_feat = aa_feat_raw * final_mask[..., None]

assert (aa_feat[0].sum(dim=-1) == 1).all(), "AA one-hot must be present (sum to 1) for every real residue, context and target alike"
assert torch.equal(aa_feat[0].argmax(dim=-1), residue_type[0]), "AA identity must match the real residue_type for every real residue"
print("AA identity (x1_aatype) survives, unchanged, for EVERY real residue -- context AND target")

print("\nPROBE PASSED: only geometry is masked (via coord_mask), only on target residues; AA identity reaches the encoder for all residues.")

# --- explicit expected vs. received, for one concrete residue of each role ---
for label, idx_mask in (("context", ctx), ("target", tgt)):
    i = int(torch.where(idx_mask)[0][0])
    print(f"\n--- expected vs. received: {label} residue #{i} ---")

    print(f"raw input coords[0, {i}] CA (unmasked, real value still sitting in the tensor): {coords[0, i, 1].tolist()}")

    coord_portion = atom_feat[0, i, : A * 3].reshape(A, 3)
    expect = "all zero" if label == "target" else "real (nonzero) values"
    got = "all zero" if (coord_portion == 0).all() else "nonzero"
    print(f"Atom37NanometersCoorsSeqFeat coordinate portion: expected {expect}, received {got} (max abs = {coord_portion.abs().max().item():.3f})")

    true_aa = int(residue_type[0, i])
    seen = aa_feat[0, i]
    print(f"x1_aatype: expected one-hot at index {true_aa} (real identity, ALWAYS present), received one-hot at index {int(seen.argmax())}")

# --- tiny 4-residue example at masked_pct=0.7, side-by-side context vs. target ---
print("\n\n=== tiny 4-residue protein, masked_pct=0.7: what the online encoder sees ===\n")

torch.manual_seed(1)  # gives a mixed 2-context/2-target split for a clearer example
B4, L4 = 1, 4
coords4 = torch.randn(B4, L4, A, 3) * 2.0 + 5.0
coords4_nm = coords4 / 10.0
coord_mask4 = torch.ones(B4, L4, A, dtype=torch.bool)
residue_type4 = torch.randint(0, 20, (B4, L4))
valid4 = torch.ones(B4, L4, A, 3, dtype=torch.bool)
valid_res_type4 = torch.ones(B4, L4, dtype=torch.bool)

batch4 = {
    "coords": coords4,
    "coords_nm": coords4_nm,
    "coord_mask": coord_mask4,
    "residue_type": residue_type4,
    "mask_dict": {"coords": valid4, "residue_type": valid_res_type4},
}

context_mask4, target_mask4 = random_residue_mask(batch4, masked_pct=0.7)
masked_batch4 = mask_batch_for_encoder_chemmask_aa_only(batch4, context_mask4)

atom_feat4 = Atom37NanometersCoorsSeqFeat()(masked_batch4)  # [1, 4, 37*4]
aa_feat4_raw = ResidueTypeSeqFeat()(masked_batch4)  # [1, 4, 20]

final_mask4 = masked_batch4["mask_dict"]["coords"][..., 0, 0]  # [1, 4]
aa_feat4 = aa_feat4_raw * final_mask4[..., None]

print(f"{'residue':<8} {'role':<9} {'true AA':<8} {'encoder sees (AA)':<20} {'true CA (Å)':<28} {'encoder sees (CA)'}")
for r in range(L4):
    role = "context" if context_mask4[0, r] else "target"
    true_aa_r = int(residue_type4[0, r])
    seen_aa = aa_feat4[0, r]
    seen_aa_str = f"index {int(seen_aa.argmax())}" if seen_aa.sum() > 0 else "ZERO (no identity)"

    true_ca = coords4[0, r, 1].tolist()  # atom index 1 = CA
    seen_ca_flat = atom_feat4[0, r, : A * 3].reshape(A, 3)[1]  # CA slot of the feature
    seen_ca_str = "ZERO (hidden)" if (seen_ca_flat == 0).all() else f"[{seen_ca_flat[0]:.2f}, {seen_ca_flat[1]:.2f}, {seen_ca_flat[2]:.2f}]"

    true_ca_str = f"[{true_ca[0]:.2f}, {true_ca[1]:.2f}, {true_ca[2]:.2f}]"
    print(f"{r:<8} {role:<9} {true_aa_r:<8} {seen_aa_str:<20} {true_ca_str:<28} {seen_ca_str}")

print(
    "\ncontext residues: encoder sees real AA identity AND real coordinates"
    "\ntarget residues:  encoder sees real AA identity, coordinates hidden (zero)"
)
