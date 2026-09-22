"""Probe for mask_batch_for_encoder_chemmask_partial_coordinates (per-atom masking,
context=target=all residues, predictor self-exclusion).

Checks: (1) AA identity always present, (2) ~mask_pct of each residue's valid atoms
hidden, (3) masked atoms contribute no coord feature and no pair distance (N
specifically, via BackbonePairDistancesNanometerPairFeat's CA-N column),
(4) predictor self-exclusion: perturbing context_i doesn't change predicted_i,
but does change some predicted_j.

Run with:  python -m proteinfoundation.jepa.tests.chemmask_partial_coordinates_probe
"""

import torch

from proteinfoundation.jepa.jepa_model import (
    mask_batch_for_encoder_chemmask_partial_coordinates,
    mask_batch_for_encoder_chemmask_partial_coords_and_identity,
)
from proteinfoundation.jepa.masking import apply_mask_to_latents
from proteinfoundation.jepa.predictor import JEPAPredictor
from proteinfoundation.nn.feature_factory import (
    Atom37NanometersCoorsSeqFeat,
    BackbonePairDistancesNanometerPairFeat,
    ResidueTypeSeqFeat,
)

torch.manual_seed(0)

# --- (1) + (2): identity preserved, ~70% of each residue's valid atoms hidden ---
B, L, A = 1, 40, 37
coords = torch.randn(B, L, A, 3) * 2.0 + 5.0
coords_nm = coords / 10.0
coord_mask = torch.ones(B, L, A, dtype=torch.bool)
coord_mask[:, :, 34:] = False  # simulate residues without the last 3 sidechain slots (e.g. short AAs)
residue_type = torch.randint(0, 20, (B, L))
valid_res_type = torch.ones(B, L, dtype=torch.bool)

batch = {
    "coords": coords,
    "coords_nm": coords_nm,
    "coord_mask": coord_mask,
    "residue_type": residue_type,
    "mask_dict": {"coords": torch.ones(B, L, A, 3, dtype=torch.bool), "residue_type": valid_res_type},
}

masked_batch = mask_batch_for_encoder_chemmask_partial_coordinates(batch, mask_pct=0.7)

assert masked_batch["mask_dict"] is batch["mask_dict"], "mask_dict must be untouched (protects AA identity)"
assert torch.equal(masked_batch["coords"], coords), "coords must be left as real values (masking is via coord_mask)"
# absent atoms (already False) must stay False, never get "unmasked"
assert (masked_batch["coord_mask"][:, :, 34:] == False).all(), "absent atoms must remain False"

n_valid = coord_mask[:, :, :34].sum(dim=-1).float()  # [B, L], 34 valid slots/residue here
n_kept = masked_batch["coord_mask"][:, :, :34].sum(dim=-1).float()
frac_hidden = 1 - n_kept / n_valid
print(f"mean fraction of valid atoms hidden per residue: {frac_hidden.mean():.3f} (target 0.7)")
assert (frac_hidden - 0.7).abs().mean() < 0.02, "should be ~70% per residue on average"

aa_feat = ResidueTypeSeqFeat()(masked_batch)
assert (aa_feat[0].sum(dim=-1) == 1).all(), "AA one-hot must be present for every residue -- identity is never masked here"
assert torch.equal(aa_feat[0].argmax(dim=-1), residue_type[0]), "AA identity must match residue_type for every residue"
print("AA identity present for all 40 residues; absent atoms (slots 34-36) stayed unmasked-absent, not touched")

# --- (3): masked atoms contribute no coord feature; a masked N leaks no CA-N distance ---
atom_feat = Atom37NanometersCoorsSeqFeat()(masked_batch)  # [B, L, 37*4]
hidden = ~masked_batch["coord_mask"]  # [B, L, 37]
coord_portion = atom_feat[0, :, : A * 3].reshape(L, A, 3)
assert torch.equal(coord_portion[hidden[0]], torch.zeros_like(coord_portion[hidden[0]])), "a hidden atom leaked a nonzero coordinate feature"
print("every hidden atom slot contributes zero to Atom37NanometersCoorsSeqFeat")

i, j = 2, 5
ctrl_mask = coord_mask.clone()
ctrl_mask[:, j, 0] = False  # hide only N of residue j; CA stays valid
ctrl_batch = dict(batch)
ctrl_batch["coord_mask"] = ctrl_mask
pair_feat = BackbonePairDistancesNanometerPairFeat()(ctrl_batch)  # [B, L, L, 84]
ca_n_feat = pair_feat[0, i, j, :21]  # CA_i-N_j one-hot-binned distance, first 21 dims
assert torch.equal(ca_n_feat, torch.zeros_like(ca_n_feat)), "CA-N pair feature leaked a real distance for a masked N"
print(f"residue {j}'s N masked, CA left valid: CA_{i}-N_{j} pair-distance feature is correctly zeroed (was leaking before the fix)")

print("\nPROBE (masking) PASSED\n")

# --- (4): predictor self-exclusion ---
torch.manual_seed(0)
Bp, Lp, Dz, H = 1, 6, 8, 128
valid_mask = torch.ones(Bp, Lp, dtype=torch.bool)
latents = torch.randn(Bp, Lp, Dz)

masked_latents = apply_mask_to_latents(latents, valid_mask, valid_mask)  # context=target=all residues
assert torch.equal(masked_latents["context_indices"], masked_latents["target_indices"]), "context/target indices must coincide for the overlapping case"

predictor = JEPAPredictor(
    latent_z_dim=Dz, hidden_dim=H, num_layers=2, num_heads=4, max_len=Lp,
    use_self_attn=True,
)
predictor.eval()

def run(latents_in):
    ml = apply_mask_to_latents(latents_in, valid_mask, valid_mask)
    with torch.no_grad():
        return predictor(
            ml["context_latents"], ml["context_indices"], ml["target_indices"],
            key_padding_mask=ml["context_padding"], self_exclude=True,
        )  # [B, L, Dz]

pred_before = run(latents)
perturbed = latents.clone()
target_i = 3
perturbed[0, target_i] += 10.0  # large perturbation to context_i's latent
pred_after = run(perturbed)

diff = (pred_after - pred_before)[0]  # [L, Dz]
self_diff = diff[target_i].abs().max().item()
other_diff = diff[torch.arange(Lp) != target_i].abs().max().item()
print(f"perturbed context_{target_i}: predicted_{target_i} max abs change = {self_diff:.2e} (expect ~0)")
print(f"                              max change among other residues    = {other_diff:.2e} (expect > 0)")
assert self_diff < 1e-5, "self-exclusion failed: predicted_i changed when only context_i was perturbed"
assert other_diff > 1e-4, "neighbours' predictions should change when context_i is perturbed (sanity: mask is actually wired into attention)"

print("\nPROBE (self-exclusion) PASSED: predicted_i is invariant to context_i; neighbours' predictions do change.")

# --- (5): V2 retains V1's coordinate mask and independently hides exact-count AA identities ---
toy_coord_mask = torch.zeros(1, 3, 37, dtype=torch.bool)
toy_coord_mask[0, 0, :5] = True   # ALA: N, CA, C, O, CB
toy_coord_mask[0, 1, :4] = True   # GLY: N, CA, C, O
toy_coord_mask[0, 2, :8] = True   # ASP: N, CA, C, O, CB, CG, OD1, OD2
toy_batch = {
    "coords": torch.randn(1, 3, 37, 3),
    "coords_nm": torch.randn(1, 3, 37, 3),
    "coord_mask": toy_coord_mask,
    "residue_type": torch.tensor([[0, 7, 3]]),
    "mask_dict": {
        "coords": torch.ones(1, 3, 37, 3, dtype=torch.bool),
        "residue_type": torch.ones(1, 3, dtype=torch.bool),
    },
}

torch.manual_seed(23)
v1 = mask_batch_for_encoder_chemmask_partial_coordinates(toy_batch, mask_pct=0.7)
torch.manual_seed(23)
v2 = mask_batch_for_encoder_chemmask_partial_coords_and_identity(
    toy_batch, atom_mask_pct=0.7, identity_mask_pct=0.5
)
assert torch.equal(v1["coord_mask"], v2["coord_mask"]), "V2 coordinate draw must match V1 with the same seed"
identity_hidden = ~v2["mask_dict"]["residue_type"]
assert identity_hidden.sum().item() == 2, "round(0.5 * 3) identity masks expected"
v2_aa_feat = ResidueTypeSeqFeat()(v2)
assert (v2_aa_feat[identity_hidden] == 0).all(), "hidden identity must produce a zero AA feature"
assert (v2_aa_feat[~identity_hidden].sum(dim=-1) == 1).all(), "shown identity must remain one-hot"
print("PROBE (V2 identity): coordinate masks match V1; exactly 2/3 identities are independently zeroed")
