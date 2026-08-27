"""CPU smoke tests for JEPA independent rotations and optional SIGReg."""

import torch

from proteinfoundation.jepa.jepa_model import (
    SIGReg,
    covariance_effective_rank,
    independently_rotate_target_batch,
)


def test_independent_rotation_preserves_geometry_and_input() -> None:
    coords_nm = torch.randn(3, 7, 5, 3)
    coords = torch.randn(3, 7, 5, 3)
    batch = {"coords_nm": coords_nm, "coords": coords, "sentinel": object()}
    original = coords_nm.clone()
    rotated = independently_rotate_target_batch(batch)

    assert rotated is not batch
    assert rotated["coords"] is coords
    assert rotated["sentinel"] is batch["sentinel"]
    assert torch.equal(batch["coords_nm"], original)
    assert not torch.allclose(rotated["coords_nm"], original)
    assert torch.allclose(
        rotated["coords_nm"].norm(dim=-1), original.norm(dim=-1), atol=1e-5
    )
    original_d = torch.cdist(original[:, :, 0], original[:, :, 0])
    rotated_d = torch.cdist(
        rotated["coords_nm"][:, :, 0], rotated["coords_nm"][:, :, 0]
    )
    assert torch.allclose(rotated_d, original_d, atol=1e-5)


def test_sigreg_detects_collapse_and_has_gradients() -> None:
    torch.manual_seed(42)
    regularizer = SIGReg(num_proj=64)
    isotropic = torch.randn(1, 4096, 16, requires_grad=True)
    collapsed = torch.zeros(1, 4096, 16)

    torch.manual_seed(7)
    isotropic_loss = regularizer(isotropic)
    torch.manual_seed(7)
    collapsed_loss = regularizer(collapsed)
    assert isotropic_loss < collapsed_loss
    isotropic_loss.backward()
    assert isotropic.grad is not None
    assert torch.isfinite(isotropic.grad).all()
    assert isotropic.grad.abs().sum() > 0

    one_direction = torch.randn(512, 1).repeat(1, 16)
    full_rank = torch.randn(4096, 16)
    assert covariance_effective_rank(one_direction) < 1.01
    assert covariance_effective_rank(full_rank) > 14.0


if __name__ == "__main__":
    test_independent_rotation_preserves_geometry_and_input()
    test_sigreg_detects_collapse_and_has_gradients()
    print("rotation + SIGReg smoke tests passed")
