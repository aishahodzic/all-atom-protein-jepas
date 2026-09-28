import torch

from openfold.np import residue_constants

from Evaluation.structure_rec.geometry_projection import idealize_atom37_geometry


def test_projection_preserves_ca_and_repairs_arg_bonds():
    arg = residue_constants.restype_order["R"]
    coords = torch.randn(1, 1, 37, 3, dtype=torch.float32)
    coords[..., 1, :] = torch.tensor([[[0.2, -0.4, 0.7]]])
    mask = torch.as_tensor(residue_constants.STANDARD_ATOM_MASK[arg]).view(1, 1, 37)
    projected, projected_mask = idealize_atom37_geometry(
        coords, torch.tensor([[arg]]), mask
    )

    torch.testing.assert_close(projected[..., 1, :], coords[..., 1, :], atol=1e-6, rtol=0)
    bonds, _, _ = residue_constants.load_stereo_chemical_props()
    projected_ang = projected[0, 0] * 10.0
    for bond in bonds["ARG"]:
        atom1 = residue_constants.atom_order[bond.atom1_name]
        atom2 = residue_constants.atom_order[bond.atom2_name]
        assert projected_mask[0, 0, atom1]
        assert projected_mask[0, 0, atom2]
        distance = torch.linalg.vector_norm(projected_ang[atom1] - projected_ang[atom2])
        torch.testing.assert_close(
            # OpenFold's rigid-group literature coordinates and its separate
            # stereochemical-props table differ by a few thousandths of an Å.
            distance, torch.tensor(bond.length), atol=1e-2, rtol=0
        )
