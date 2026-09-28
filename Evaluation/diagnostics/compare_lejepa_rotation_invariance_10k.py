"""Matched SO(3) invariance check for the two uncropped LeJEPA 10k EMA encoders.

Run on CPU with the pstbench environment. The same held-out proteins and four
fixed random rotations are used for both models. There is no view masking at
evaluation: only the input coordinates change between paired encodings.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import rotation_latent_gauge as gauge


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "uncropped_lejepa_ema_10k": ROOT / "store/canonical_lejepa_v1_100k/evaluation/canonical-lejepa-v1-step10000-ema-staged.ckpt",
    "uncropped_lejepa_so3_ema_10k": ROOT / "store/canonical_lejepa_rotations_100k/evaluation/canonical-lejepa-v1-rotations-step10000-ema-staged.ckpt",
}
PROTEIN_IDS = gauge.TEST_IDS
RESULT = ROOT / "Evaluation/diagnostics/results/lejepa_rotation_invariance_ema_10k.json"


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    # A normalised Gaussian quaternion is uniform on SO(3).
    q = torch.randn(4, generator=generator)
    w, x, y, z = q / q.norm()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float(), b.float()
    return {
        "cosine": F.cosine_similarity(a, b, dim=-1).mean().item(),
        "relative_l2": (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)).item(),
    }


def main() -> None:
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(20260918)
    gauge.ROTATIONS = {"identity": torch.eye(3)}
    gauge.ROTATIONS.update({f"so3_{i}": random_rotation(generator) for i in range(4)})
    report = {
        "description": "Full-context 8D residue-latent rotation invariance; same proteins and rotations for both 10k EMA models",
        "protein_ids": PROTEIN_IDS,
        "rotations": {name: matrix.tolist() for name, matrix in gauge.ROTATIONS.items() if name != "identity"},
        "models": {},
    }
    for name, checkpoint in CHECKPOINTS.items():
        print(f"\n{name}: {checkpoint}", flush=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        encoder = gauge.load_encoder(checkpoint)
        values = gauge.encode_all(encoder, PROTEIN_IDS, batch_size=2)
        del encoder
        paired, shuffled = [], []
        centered_paired, centered_shuffled, residue_signal = [], [], []
        for rotation in list(gauge.ROTATIONS)[1:]:
            for protein in PROTEIN_IDS:
                original = values["identity"][protein]
                rotated = values[rotation][protein]
                if original.shape != rotated.shape:
                    raise ValueError(f"Residue mismatch: {protein}, {rotation}")
                paired.append(pair_metrics(original, rotated))
                # Deliberately wrong residue correspondence: a collapse check.
                permutation = torch.randperm(len(rotated), generator=generator)
                shuffled.append(pair_metrics(original, rotated[permutation]))
                original_centered = original - original.mean(dim=0, keepdim=True)
                rotated_centered = rotated - rotated.mean(dim=0, keepdim=True)
                centered_paired.append(pair_metrics(original_centered, rotated_centered))
                centered_shuffled.append(pair_metrics(original_centered, rotated_centered[permutation]))
                residue_signal.append(
                    (torch.linalg.vector_norm(original_centered) / torch.linalg.vector_norm(original)).item()
                )
        summary = {
            "checkpoint": str(checkpoint),
            "n_proteins": len(PROTEIN_IDS),
            "n_rotations": len(gauge.ROTATIONS) - 1,
            "n_residues": sum(len(values["identity"][protein]) for protein in PROTEIN_IDS),
            "paired_mean_cosine": sum(m["cosine"] for m in paired) / len(paired),
            "paired_mean_relative_l2": sum(m["relative_l2"] for m in paired) / len(paired),
            "shuffled_mean_cosine": sum(m["cosine"] for m in shuffled) / len(shuffled),
            "shuffled_mean_relative_l2": sum(m["relative_l2"] for m in shuffled) / len(shuffled),
            "centered_paired_mean_cosine": sum(m["cosine"] for m in centered_paired) / len(centered_paired),
            "centered_paired_mean_relative_l2": sum(m["relative_l2"] for m in centered_paired) / len(centered_paired),
            "centered_shuffled_mean_cosine": sum(m["cosine"] for m in centered_shuffled) / len(centered_shuffled),
            "centered_shuffled_mean_relative_l2": sum(m["relative_l2"] for m in centered_shuffled) / len(centered_shuffled),
            "mean_residue_signal_to_total_norm": sum(residue_signal) / len(residue_signal),
            "per_pair": paired,
        }
        report["models"][name] = summary
        print(json.dumps({k: v for k, v in summary.items() if k != "per_pair"}, indent=2), flush=True)
        del values
    RESULT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {RESULT}", flush=True)


if __name__ == "__main__":
    main()
