"""Hard per-residue ideal-geometry projection for decoded Atom37 coordinates.

The decoder's predicted backbone frame and torsion angles are retained, but
atom positions are rebuilt from OpenFold's literature geometry.  This keeps
C-alpha anchors fixed and guarantees residue-internal bond lengths/angles by
construction.  It is an explicit post-processing operation, not part of the
raw decoder prediction.
"""

from __future__ import annotations

import torch

from openfold.data.data_transforms import atom37_to_frames, atom37_to_torsion_angles
from openfold.np import residue_constants
from openfold.utils.all_atom_multimer import atom14_to_atom37
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.rigid_utils import Rigid

from proteinfoundation.utils.coors_utils import ang_to_nm, nm_to_ang


def idealize_atom37_geometry(
    coords_nm: torch.Tensor,
    aatype: torch.Tensor,
    atom37_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild Atom37 coordinates using ideal residue geometry.

    The returned coordinates are in nanometres.  C-alpha coordinates are
    preserved (up to floating-point precision).  Peptide connections between
    independently framed residues are not constrained by this operation.
    """

    working_dtype = torch.float64
    features = {
        "aatype": aatype.long().clamp(max=20),
        "all_atom_positions": nm_to_ang(coords_nm).to(working_dtype),
        "all_atom_mask": atom37_mask.to(working_dtype),
    }
    features = atom37_to_frames(features)
    features = atom37_to_torsion_angles()(features)

    device = coords_nm.device
    default_frames = torch.as_tensor(
        residue_constants.restype_rigid_group_default_frame,
        dtype=working_dtype,
        device=device,
    )
    group_idx = torch.as_tensor(
        residue_constants.restype_atom14_to_rigid_group,
        dtype=torch.long,
        device=device,
    )
    atom14_mask = torch.as_tensor(
        residue_constants.restype_atom14_mask,
        dtype=working_dtype,
        device=device,
    )
    literature_positions = torch.as_tensor(
        residue_constants.restype_atom14_rigid_group_positions,
        dtype=working_dtype,
        device=device,
    )

    backbone_frame = Rigid.from_tensor_4x4(
        features["rigidgroups_gt_frames"][..., 0, :, :]
    )
    torsions = features["torsion_angles_sin_cos"]
    torsion_mask = features["torsion_angles_mask"][..., None]
    identity_torsion = torch.zeros_like(torsions)
    identity_torsion[..., 1] = 1.0
    torsions = torsions * torsion_mask + identity_torsion * (1.0 - torsion_mask)

    frames = torsion_angles_to_frames(
        backbone_frame,
        torsions,
        features["aatype"],
        default_frames,
    )
    atom14 = frames_and_literature_positions_to_atom14_pos(
        frames,
        features["aatype"],
        default_frames,
        group_idx,
        atom14_mask,
        literature_positions,
    )
    atom37_ang, projected_mask = atom14_to_atom37(atom14, features["aatype"])
    projected_nm = ang_to_nm(atom37_ang).to(coords_nm.dtype)
    projected_mask = projected_mask.bool()

    # Backbone frames are CA-centred; guard that invariant explicitly.
    ca_present = projected_mask[..., 1] & atom37_mask[..., 1].bool()
    if ca_present.any():
        ca_error = torch.max(
            torch.abs(projected_nm[..., 1, :][ca_present] - coords_nm[..., 1, :][ca_present])
        )
        if ca_error > 1e-5:
            raise RuntimeError(f"ideal-geometry projection moved C-alpha by {ca_error.item()} nm")
    return projected_nm, projected_mask
