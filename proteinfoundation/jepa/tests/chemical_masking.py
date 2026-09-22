"""Focused checks for partial-coordinate chemical masking.

Run with: python -m proteinfoundation.jepa.tests.chemical_masking
"""

import torch

from proteinfoundation.jepa.masking import (
    BACKBONE_ATOM37_INDICES,
    mask_partial_atom_coordinates,
    mask_partial_sidechain_coordinates,
    sidechain_atom_mask,
    sidechain_chemical_context_target_masks,
)
from proteinfoundation.jepa.jepa_model import covariance_effective_rank
from proteinfoundation.nn.feature_factory import (
    Atom37NanometersCoorsSeqFeat,
    BackbonePairDistancesNanometerPairFeat,
    RelativeResidueOrientationPairFeat,
    ResidueTypeSeqFeat,
)


torch.manual_seed(4)
B, L, A = 2, 5, 37
coords = torch.randn(B, L, A, 3)
coord_mask = torch.ones(B, L, A, dtype=torch.bool)
coord_mask[1, 4] = False  # padded residue
batch = {
    "coords": coords.clone(),
    "coords_nm": coords.clone(),
    "ca_coors_nm": coords[:, :, 1].clone(),
    "coord_mask": coord_mask.clone(),
    "residue_type": torch.randint(0, 20, (B, L)),
    "mask_dict": {
        "coords": coord_mask[..., None].expand(-1, -1, -1, 3).clone(),
        "residue_type": torch.ones(B, L, dtype=torch.bool),
    },
}
original_coords = batch["coords"].clone()
original_validity = batch["mask_dict"]["coords"].clone()

torch.manual_seed(11)
masked = mask_partial_atom_coordinates(batch, mask_pct=0.5)
hidden = coord_mask & ~masked["coord_mask"]

expected_hidden = (coord_mask.sum(dim=-1).float() * 0.5).round().long()
assert torch.equal(hidden.sum(dim=-1), expected_hidden)
assert torch.equal(batch["coords"], original_coords)
assert torch.equal(batch["mask_dict"]["coords"], original_validity)
assert masked["mask_dict"] is batch["mask_dict"]
assert not masked["coords"][hidden].any()
assert not masked["coords_nm"][hidden].any()
assert not masked["ca_coors_nm"][~masked["coord_mask"][..., 1]].any()
print("scenario 1: exact per-residue atom masking is non-mutating and zeroes hidden values")

perturbed_batch = dict(batch)
perturbed_batch["coords"] = batch["coords"].clone()
perturbed_batch["coords_nm"] = batch["coords_nm"].clone()
perturbed_batch["coords"][hidden] += 10_000.0
perturbed_batch["coords_nm"][hidden] += 10_000.0
torch.manual_seed(11)
masked_perturbed = mask_partial_atom_coordinates(perturbed_batch, mask_pct=0.5)
for key in ("coords", "coords_nm", "coord_mask", "ca_coors_nm"):
    assert torch.equal(masked[key], masked_perturbed[key]), f"hidden values leaked via {key}"

features = (
    Atom37NanometersCoorsSeqFeat(rel=False),
    Atom37NanometersCoorsSeqFeat(rel=True),
    BackbonePairDistancesNanometerPairFeat(),
    RelativeResidueOrientationPairFeat(),
)
for feature in features:
    assert torch.equal(feature(masked), feature(masked_perturbed))
print("scenario 2: hidden coordinate perturbations do not affect encoder geometry features")

relative = Atom37NanometersCoorsSeqFeat(rel=True)(masked)
relative_coords = relative[..., : A * 3]
missing_ca = ~masked["coord_mask"][..., 1]
assert not relative_coords[missing_ca].any()
print("scenario 3: hidden CA never turns the global origin into a relative feature")

pair_batch = {
    "coords": torch.randn(1, 2, A, 3),
    "coords_nm": torch.randn(1, 2, A, 3),
    "coord_mask": torch.ones(1, 2, A, dtype=torch.bool),
    "residue_type": torch.zeros(1, 2, dtype=torch.long),
}
pair_batch["coord_mask"][:, 1, 0] = False  # N_j hidden
pair_dist = BackbonePairDistancesNanometerPairFeat()(pair_batch)
assert not pair_dist[:, :, 1, :21].any()  # CA_i -> N_j channel
orientation = RelativeResidueOrientationPairFeat()(pair_batch)
assert not orientation[:, :, 1].any()
assert not orientation[:, 1, :].any()
print("scenario 4: pair features suppress channels requiring hidden atoms")

# Effective rank distinguishes a one-dimensional embedding cloud from one using
# all eight latent dimensions.
one_dimensional = torch.arange(16, dtype=torch.float32)[:, None] * torch.ones(1, 8)
full_rank = torch.eye(8)
assert torch.allclose(covariance_effective_rank(one_dimensional), torch.tensor(1.0))
assert covariance_effective_rank(full_rank) > 6.9
print("scenario 5: covariance effective rank detects dimensional collapse")

chemical_mask = torch.zeros(1, 3, A, dtype=torch.bool)
backbone = list(BACKBONE_ATOM37_INDICES)
chemical_mask[0, 0, backbone + [3]] = True  # ALA: CB
chemical_mask[0, 1, backbone] = True  # GLY: no side chain
chemical_mask[0, 2, backbone + [3, 5, 11, 12]] = True  # LEU-like: four SC atoms
chemical_coords = torch.randn(1, 3, A, 3)
chemical_batch = {
    "coords": chemical_coords.clone(),
    "coords_nm": chemical_coords.clone(),
    "ca_coors_nm": chemical_coords[:, :, 1].clone(),
    "coord_mask": chemical_mask.clone(),
    "residue_type": torch.tensor([[0, 7, 10]]),
    "mask_dict": {
        "coords": chemical_mask[..., None].expand(-1, -1, -1, 3).clone(),
        "residue_type": torch.ones(1, 3, dtype=torch.bool),
    },
}
torch.manual_seed(19)
sc70 = mask_partial_sidechain_coordinates(chemical_batch, mask_pct=0.7)
sc_hidden = chemical_mask & ~sc70["coord_mask"]
assert not sc_hidden[..., backbone].any()
assert torch.equal(sc_hidden.sum(dim=-1), torch.tensor([[1, 0, 3]]))
assert not sc70["coords"][sc_hidden].any()
assert torch.equal(sc70["ca_coors_nm"], chemical_batch["ca_coors_nm"])
context, target = sidechain_chemical_context_target_masks(
    chemical_batch, mask_aa_identity=False
)
assert torch.equal(context, torch.tensor([[True, True, True]]))
assert torch.equal(target, torch.tensor([[True, False, True]]))
print("scenario 6: SC70 preserves backbone and makes GLY context-only")

torch.manual_seed(23)
noaa_sc50 = mask_partial_sidechain_coordinates(
    chemical_batch, mask_pct=0.5, mask_aa_identity=True
)
assert not noaa_sc50["mask_dict"]["residue_type"].any()
assert not noaa_sc50["residue_type"].any()
assert not ResidueTypeSeqFeat()(noaa_sc50).any()
assert chemical_batch["mask_dict"]["residue_type"].all()  # source is untouched
assert torch.equal(
    noaa_sc50["coord_mask"][..., backbone], chemical_mask[..., backbone]
)
context, target = sidechain_chemical_context_target_masks(
    chemical_batch, mask_aa_identity=True
)
assert torch.equal(context, torch.tensor([[True, True, True]]))
assert torch.equal(target, torch.tensor([[True, True, True]]))
print("scenario 7: AA-hidden SC50 suppresses explicit identity and targets GLY")

print("all checks passed")
