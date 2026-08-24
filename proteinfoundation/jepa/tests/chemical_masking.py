"""Focused checks for partial-coordinate chemical masking.

Run with: python -m proteinfoundation.jepa.tests.chemical_masking
"""

import torch

from proteinfoundation.jepa.masking import mask_partial_atom_coordinates
from proteinfoundation.jepa.jepa_model import covariance_effective_rank
from proteinfoundation.nn.feature_factory import (
    Atom37NanometersCoorsSeqFeat,
    BackbonePairDistancesNanometerPairFeat,
    RelativeResidueOrientationPairFeat,
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
    },
}
original_coords = batch["coords"].clone()
original_validity = batch["mask_dict"]["coords"].clone()

torch.manual_seed(11)
masked = mask_partial_atom_coordinates(batch, mask_pct=0.5)
hidden = coord_mask & ~masked["coord_mask"]

# Exactly round(50%) of each real residue's observed atoms is hidden, padding is
# untouched, and the source/target-validity metadata remains intact.
expected_hidden = (coord_mask.sum(dim=-1).float() * 0.5).round().long()
assert torch.equal(hidden.sum(dim=-1), expected_hidden)
assert torch.equal(batch["coords"], original_coords)
assert torch.equal(batch["mask_dict"]["coords"], original_validity)
assert masked["mask_dict"] is batch["mask_dict"]
assert not masked["coords"][hidden].any()
assert not masked["coords_nm"][hidden].any()
assert not masked["ca_coors_nm"][~masked["coord_mask"][..., 1]].any()
print("scenario 1: exact per-residue atom masking is non-mutating and zeroes hidden values")

# Hidden coordinate values cannot change any masked batch tensor or configured
# geometry feature when the random observation mask is held fixed.
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

# A missing CA makes the entire relative-coordinate value channel undefined and
# therefore zero; atom-observation bits remain available in the last 37 channels.
relative = Atom37NanometersCoorsSeqFeat(rel=True)(masked)
relative_coords = relative[..., : A * 3]
missing_ca = ~masked["coord_mask"][..., 1]
assert not relative_coords[missing_ca].any()
print("scenario 3: hidden CA never turns the global origin into a relative feature")

# Pair channels are independently gated after one-hot binning. This specifically
# catches zero-distance values becoming a nonzero first bin after an atom is hidden.
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

print("all checks passed")
