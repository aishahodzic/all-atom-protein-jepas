"""Cheap matched representation audit for canonical residue-LeJEPA A/B at 10k.

This intentionally does not fit t-SNE.  It reuses the established held-out
protein cohort, fixed 15,000-residue sample, cross-protein cosine-neighbour
metrics, and position/length controls used by the canonical LeJEPA checkpoint
audits.

Use ``jobs/run_canonical_lr_diagnostics_jarvis.sh`` for an arbitrary Jarvis
run. That entry point fixes the raw encoder, checkpoint step, and output tag.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr


REPO = pathlib.Path(__file__).resolve().parents[2]
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
LATENT_VIZ = REPO / "Evaluation/latent_viz"
for path in (REPO, LINEAR, KNN, LATENT_VIZ):
    sys.path.insert(0, str(path))

import checkpoints as checkpoint_registry  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)
from analyze_current_four_health import (  # noqa: E402
    aligned_property,
    external_neighbours,
    neighbour_spearman,
    normalized,
    purity,
    same_protein_fraction,
    spectrum_metrics,
)
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import knn_exact_index_stats, local_index_array  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


ANALYSIS_PROFILE = os.environ.get(
    "CANONICAL_RESIDUE_ANALYSIS_PROFILE", "baseline-10k"
)
EXTERNAL_PROFILE = "external-protein-perview-independent-locals"
ENCODER_SOURCE = os.environ.get("CANONICAL_ENCODER_SOURCE", "ema").lower()
if ENCODER_SOURCE not in ("ema", "raw"):
    raise ValueError(f"unknown canonical encoder source: {ENCODER_SOURCE}")
if ANALYSIS_PROFILE not in (
    "baseline-10k",
    "protein-centered-sigreg-2k",
    "protein-centered-sigreg-20k",
    "cropped-fixed96-10k",
    "protein-joint-b32-cropped-fixed96-10k",
    "protein-joint-b96-cropped-fixed96-10k",
    "protein-joint-b96-cropped-fixed96-20k",
    "protein-perview-b96-cropped-fixed96-10k",
    "protein-perview-b96-independent-locals-cropped-fixed96-10k",
    EXTERNAL_PROFILE,
):
    raise ValueError(f"unknown canonical residue analysis profile: {ANALYSIS_PROFILE}")

EXPECTED_STEP = {
    "baseline-10k": 10_000,
    "protein-centered-sigreg-2k": 2_000,
    "protein-centered-sigreg-20k": 20_000,
    "cropped-fixed96-10k": 10_000,
    "protein-joint-b32-cropped-fixed96-10k": 10_000,
    "protein-joint-b96-cropped-fixed96-10k": 10_000,
    "protein-joint-b96-cropped-fixed96-20k": 20_000,
    "protein-perview-b96-cropped-fixed96-10k": 10_000,
    "protein-perview-b96-independent-locals-cropped-fixed96-10k": 10_000,
    EXTERNAL_PROFILE: int(os.environ.get("CANONICAL_EXPECTED_STEP", "10000")),
}[ANALYSIS_PROFILE]
EXPECTED_EMA_UPDATES = {2_000: 62, 10_000: 312, 20_000: 625}.get(
    EXPECTED_STEP
)
EXPECTED_LAST_EMA_STEP = {
    2_000: 1_984,
    10_000: 9_984,
    20_000: 20_000,
}.get(EXPECTED_STEP)
EXPECTED_PROTEINS = 5_224
EXPECTED_RESIDUE_POOL = 748_402
N_POINTS = 15_000
K = 10
EXPECTED_PHYSICAL_SAMPLE_SHA256 = (
    "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"
)

if ANALYSIS_PROFILE == EXTERNAL_PROFILE:
    checkpoint_path = pathlib.Path(
        os.environ["CANONICAL_EXTERNAL_CHECKPOINT"]
    ).expanduser().resolve()
    config_path = pathlib.Path(
        os.environ["CANONICAL_EXTERNAL_CONFIG"]
    ).expanduser().resolve()
    external_name = os.environ.get(
        "CANONICAL_EXTERNAL_NAME", checkpoint_path.parents[1].name
    )
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in external_name
    )
    RUNS = {
        f"{safe_name}-{ENCODER_SOURCE}-step{EXPECTED_STEP}": {
            "mode": "visible_only",
            "sigreg_input": "protein_mean",
            "run_name": external_name,
            "checkpoint": checkpoint_path,
            "config": config_path,
            "staged": checkpoint_path.parents[1]
            / "evaluation"
            / f"{safe_name}-step{EXPECTED_STEP}-{ENCODER_SOURCE}-staged.ckpt",
        }
    }
elif ANALYSIS_PROFILE.startswith("protein-joint-b96-cropped-fixed96-"):
    step_label = str(EXPECTED_STEP)
    RUNS = {
        f"canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-ema-step{step_label}": {
            "mode": "visible_only",
            "sigreg_input": "protein_mean",
            "run_name": (
                "canonical_lejepa_crop_fixed96_protein_joint_sigreg_b96_10k"
            ),
            "stage_name": (
                "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
                f"step{step_label}-ema-staged.ckpt"
            ),
        }
    }
elif ANALYSIS_PROFILE in (
    "protein-perview-b96-cropped-fixed96-10k",
    "protein-perview-b96-independent-locals-cropped-fixed96-10k",
):
    independent_locals = "independent-locals" in ANALYSIS_PROFILE
    RUNS = {
        "canonical-lejepa-protein-cropped-fixed96-perview-sigreg-b96-"
        + ("independent-locals-" if independent_locals else "")
        + "ema-step10000": {
            "mode": "visible_only",
            "sigreg_input": "protein_mean",
            "run_name": (
                "canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_"
                "independent_locals_30k"
                if independent_locals
                else "canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_100k"
            ),
            "stage_name": (
                "canonical-lejepa-protein-cropped-fixed96-perview-sigreg-b96-"
                + ("independent-locals-" if independent_locals else "")
                + "step10000-ema-staged.ckpt"
            ),
        }
    }
elif ANALYSIS_PROFILE == "protein-joint-b32-cropped-fixed96-10k":
    RUNS = {
        "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b32-ema-step10000": {
            "mode": "visible_only",
            "sigreg_input": "protein_mean",
            "run_name": (
                "canonical_lejepa_crop_fixed96_protein_joint_sigreg_b32_100k"
            ),
            "stage_name": (
                "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b32-"
                "step10000-ema-staged.ckpt"
            ),
        }
    }
elif ANALYSIS_PROFILE == "cropped-fixed96-10k":
    RUNS = {
        "canonical-lejepa-residue-cropped-fixed96-local48-ema-step10000": {
            "mode": "visible_only",
            "sigreg_input": "protein_centered_residue",
            "run_name": "canonical_lejepa_crop_fixed_global96_local48_100k",
            "stage_name": (
                "canonical-lejepa-residue-cropped-fixed96-local48-"
                "step10000-ema-staged.ckpt"
            ),
        }
    }
elif ANALYSIS_PROFILE.startswith("protein-centered-sigreg-"):
    step_label = str(EXPECTED_STEP)
    RUNS = {
        f"canonical-lejepa-residue-iterative-protein-centered-sigreg-ema-step{step_label}": {
            "mode": "iterative",
            "sigreg_input": "protein_centered_residue",
            "run_name": (
                "canonical_lejepa_residue_iterative_"
                "protein_centered_sigreg_2k"
            ),
            "stage_name": (
                "canonical-lejepa-residue-iterative-protein-centered-sigreg-"
                f"step{step_label}-ema-staged.ckpt"
            ),
        },
        f"canonical-lejepa-residue-visible-only-protein-centered-sigreg-ema-step{step_label}": {
            "mode": "visible_only",
            "sigreg_input": "protein_centered_residue",
            "run_name": (
                "canonical_lejepa_residue_visible_only_"
                "protein_centered_sigreg_2k"
            ),
            "stage_name": (
                "canonical-lejepa-residue-visible-only-protein-centered-sigreg-"
                f"step{step_label}-ema-staged.ckpt"
            ),
        },
    }
else:
    RUNS = {
        "canonical-lejepa-residue-iterative-ema-step10000": {
            "mode": "iterative",
            "sigreg_input": "protein_mean",
            "run_name": "canonical_lejepa_residue_iterative_10k",
            "stage_name": (
                "canonical-lejepa-residue-iterative-step10000-ema-staged.ckpt"
            ),
        },
        "canonical-lejepa-residue-visible-only-ema-step10000": {
            "mode": "visible_only",
            "sigreg_input": "protein_mean",
            "run_name": "canonical_lejepa_residue_visible_only_10k",
            "stage_name": (
                "canonical-lejepa-residue-visible-only-step10000-ema-staged.ckpt"
            ),
        },
    }

if ENCODER_SOURCE == "raw" and ANALYSIS_PROFILE != EXTERNAL_PROFILE:
    RUNS = {
        checkpoint_id.replace("-ema-step", "-raw-step"): {
            **spec,
            "stage_name": spec["stage_name"].replace(
                "-ema-staged.ckpt", "-raw-staged.ckpt"
            ),
        }
        for checkpoint_id, spec in RUNS.items()
    }

if ANALYSIS_PROFILE != EXTERNAL_PROFILE:
    for spec in RUNS.values():
        run_root = REPO / "store" / spec["run_name"]
        spec["checkpoint"] = (
            run_root / "checkpoints" / f"step-{EXPECTED_STEP:08d}.ckpt"
        )
        spec["config"] = (
            run_root / "checkpoints" / f"exp_config_{spec['run_name']}.json"
        )
        spec["staged"] = run_root / "evaluation" / spec["stage_name"]

RESULTS_DIR = REPO / "Evaluation/diagnostics/results"
RESULT_TAG = {
    "baseline-10k": "ema_10k",
    "protein-centered-sigreg-2k": "protein_centered_sigreg_ema_2k",
    "protein-centered-sigreg-20k": "protein_centered_sigreg_ema_20k",
    "cropped-fixed96-10k": "cropped_fixed96_local48_ema_10k",
    "protein-joint-b32-cropped-fixed96-10k": (
        "protein_joint_b32_cropped_fixed96_local48_ema_10k"
    ),
    "protein-joint-b96-cropped-fixed96-10k": (
        "protein_joint_b96_cropped_fixed96_local48_ema_10k"
    ),
    "protein-joint-b96-cropped-fixed96-20k": (
        "protein_joint_b96_cropped_fixed96_local48_ema_20k"
    ),
    "protein-perview-b96-cropped-fixed96-10k": (
        "protein_perview_b96_cropped_fixed96_local48_ema_10k"
    ),
    "protein-perview-b96-independent-locals-cropped-fixed96-10k": (
        "protein_perview_b96_independent_locals_cropped_fixed96_local48_ema_10k"
    ),
    EXTERNAL_PROFILE: os.environ.get(
        "CANONICAL_EXTERNAL_RESULT_TAG",
        f"external_{ENCODER_SOURCE}_{EXPECTED_STEP}",
    ),
}[ANALYSIS_PROFILE]
if ENCODER_SOURCE == "raw":
    RESULT_TAG = RESULT_TAG.replace("_ema_", "_raw_")
CSV_PATH = RESULTS_DIR / f"canonical_lejepa_residue_ab_{RESULT_TAG}_representation.csv"
MANIFEST_PATH = (
    RESULTS_DIR / f"canonical_lejepa_residue_ab_{RESULT_TAG}_representation.json"
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partial_spearman(
    x: np.ndarray, y: np.ndarray, control: np.ndarray
) -> float:
    """Spearman correlation after residualizing ranked chain length."""
    ranked_control = rankdata(control).astype(np.float64)
    design = np.column_stack((np.ones(len(control)), ranked_control))

    def residual(values: np.ndarray) -> np.ndarray:
        ranked = rankdata(values).astype(np.float64)
        beta, *_ = np.linalg.lstsq(design, ranked, rcond=None)
        return ranked - design @ beta

    return float(np.corrcoef(residual(x), residual(y))[0, 1])


def within_length_bin_neighbours(
    values: np.ndarray,
    protein_index: np.ndarray,
    lengths: np.ndarray,
    *,
    width: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-protein cosine kNN restricted to narrow chain-length bins."""
    values = normalized(values)
    length_bins = ((lengths - 1) // width).astype(np.int64)
    result = np.full((len(values), K), -1, dtype=np.int64)
    usable = np.zeros(len(values), dtype=bool)
    for length_bin in np.unique(length_bins):
        members = np.flatnonzero(length_bins == length_bin)
        if len(members) <= K:
            continue
        query_k = min(len(members), 128)
        _, local_candidates = cKDTree(values[members]).query(
            values[members], k=query_k, workers=-1
        )
        if query_k == 1:
            local_candidates = local_candidates[:, None]
        for local_row, candidates in enumerate(local_candidates):
            row = members[local_row]
            candidates = members[np.atleast_1d(candidates)]
            external = candidates[
                protein_index[candidates] != protein_index[row]
            ]
            if len(external) >= K:
                result[row] = external[:K]
                usable[row] = True
    return result, usable


def validate_training_config(spec: dict[str, object]) -> None:
    config_path = pathlib.Path(spec["config"])
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    external_profile = ANALYSIS_PROFILE == EXTERNAL_PROFILE
    protein_profile = ANALYSIS_PROFILE.startswith("protein-") or external_profile
    protein_joint_profile = ANALYSIS_PROFILE.startswith("protein-joint-")
    expected = {
        "alignment_level": "protein" if protein_profile else "residue",
        "masked_kv_mode": spec["mode"],
        "seed": 42,
        "independent_view_rotations": False,
        "residue_view_checkpointing": not protein_profile,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    nested_expected = {
        "views.global_mask_pct": (config.get("views") or {}).get(
            "global_mask_pct"
        ),
        "views.local_mask_pct": (config.get("views") or {}).get(
            "local_mask_pct"
        ),
        "views.num_local_views": (config.get("views") or {}).get(
            "num_local_views"
        ),
        "schedule.total_steps": (config.get("schedule") or {}).get(
            "total_steps"
        ),
        "schedule.peak_step": (config.get("schedule") or {}).get("peak_step"),
        "ema.enabled": (config.get("ema") or {}).get("enabled"),
        "sigreg.input": (config.get("sigreg") or {}).get("input", "protein_mean"),
    }
    nested_values = {
        "views.global_mask_pct": 0.2,
        "views.local_mask_pct": (
            0.5
            if ANALYSIS_PROFILE
            in (
                "cropped-fixed96-10k",
                "protein-joint-b32-cropped-fixed96-10k",
                "protein-joint-b96-cropped-fixed96-10k",
                "protein-joint-b96-cropped-fixed96-20k",
                "protein-perview-b96-cropped-fixed96-10k",
                "protein-perview-b96-independent-locals-cropped-fixed96-10k",
                EXTERNAL_PROFILE,
            )
            else 0.8
        ),
        "views.num_local_views": 4,
        "schedule.total_steps": 100_000,
        "schedule.peak_step": 12_000,
        "ema.enabled": True,
        "sigreg.input": spec["sigreg_input"],
    }
    if external_profile:
        nested_values.pop("schedule.total_steps")
        nested_values.pop("schedule.peak_step")
    mismatches.update(
        {
            key: (nested_expected[key], value)
            for key, value in nested_values.items()
            if nested_expected[key] != value
        }
    )
    if ANALYSIS_PROFILE in (
        "cropped-fixed96-10k",
        "protein-joint-b32-cropped-fixed96-10k",
        "protein-joint-b96-cropped-fixed96-10k",
        "protein-joint-b96-cropped-fixed96-20k",
        "protein-perview-b96-cropped-fixed96-10k",
        "protein-perview-b96-independent-locals-cropped-fixed96-10k",
        EXTERNAL_PROFILE,
    ):
        crop_expected = {
            "views.cropping": True,
            "views.packed_views": True,
            "views.global_crop_length": 96,
            "views.local_crop_length": 48,
            "views.locals_within_global": True,
            "views.local_mask_pct": 0.5,
            "residue_compact_visible_kv": not protein_profile,
        }
        crop_actual = {
            "views.cropping": (config.get("views") or {}).get("cropping"),
            "views.packed_views": (config.get("views") or {}).get("packed_views"),
            "views.global_crop_length": (config.get("views") or {}).get(
                "global_crop_length"
            ),
            "views.local_crop_length": (config.get("views") or {}).get(
                "local_crop_length"
            ),
            "views.locals_within_global": (config.get("views") or {}).get(
                "locals_within_global"
            ),
            "views.local_mask_pct": (config.get("views") or {}).get(
                "local_mask_pct"
            ),
            "residue_compact_visible_kv": config.get(
                "residue_compact_visible_kv"
            ),
        }
        mismatches.update(
            {
                key: (crop_actual[key], value)
                for key, value in crop_expected.items()
                if crop_actual[key] != value
            }
        )
    if protein_joint_profile:
        protein_joint_expected = {
            "dataset.datamodule.batch_size": (
                48 if "b96" in ANALYSIS_PROFILE else 16
            ),
            "trainer.devices": 2,
            "sigreg.joint_views": True,
        }
        protein_joint_actual = {
            "dataset.datamodule.batch_size": (
                ((config.get("dataset") or {}).get("datamodule") or {}).get(
                    "batch_size"
                )
            ),
            "trainer.devices": (config.get("trainer") or {}).get("devices"),
            "sigreg.joint_views": (config.get("sigreg") or {}).get(
                "joint_views", False
            ),
        }
        mismatches.update(
            {
                key: (protein_joint_actual[key], value)
                for key, value in protein_joint_expected.items()
                if protein_joint_actual[key] != value
            }
        )
    if ANALYSIS_PROFILE in (
        "protein-perview-b96-cropped-fixed96-10k",
        "protein-perview-b96-independent-locals-cropped-fixed96-10k",
        EXTERNAL_PROFILE,
    ):
        protein_perview_expected = {
            "dataset.datamodule.batch_size": 48,
            "trainer.devices": 2,
            "sigreg.joint_views": False,
            "views.shared_crop_anchor": (
                False
                if external_profile
                else "independent-locals" not in ANALYSIS_PROFILE
            ),
        }
        protein_perview_actual = {
            "dataset.datamodule.batch_size": (
                ((config.get("dataset") or {}).get("datamodule") or {}).get(
                    "batch_size"
                )
            ),
            "trainer.devices": (config.get("trainer") or {}).get("devices"),
            "sigreg.joint_views": (config.get("sigreg") or {}).get(
                "joint_views", False
            ),
            "views.shared_crop_anchor": (config.get("views") or {}).get(
                "shared_crop_anchor", True
            ),
        }
        mismatches.update(
            {
                key: (protein_perview_actual[key], value)
                for key, value in protein_perview_expected.items()
                if protein_perview_actual[key] != value
            }
        )
    if mismatches:
        raise RuntimeError(
            f"{spec['run_name']} is not the matched residue-A/B recipe: "
            f"{mismatches}"
        )


def stage_and_validate(
    checkpoint_id: str, spec: dict[str, object]
) -> dict[str, object]:
    validate_training_config(spec)
    training_config = json.loads(
        pathlib.Path(spec["config"]).read_text(encoding="utf-8")
    )
    optimizer_config = training_config.get("optimizer") or {}
    schedule_config = training_config.get("schedule") or {}
    checkpoint_path = pathlib.Path(spec["checkpoint"])
    staged_path = pathlib.Path(spec["staged"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    source_step = checkpoint.get("global_step")
    if source_step != EXPECTED_STEP:
        raise RuntimeError(
            f"{checkpoint_path} has global_step={source_step}, expected "
            f"{EXPECTED_STEP}"
        )
    if ENCODER_SOURCE == "ema":
        ema = checkpoint.get("state_dict_ema")
        if not isinstance(ema, Mapping) or len(ema) != 381:
            ema_count = len(ema) if isinstance(ema, Mapping) else 0
            raise RuntimeError(
                f"{checkpoint_path} has {ema_count} EMA tensors, expected 381"
            )
        if any(not str(key).startswith("encoder.") for key in ema):
            raise RuntimeError(
                f"{checkpoint_path} EMA state contains non-encoder parameters"
            )
        ema_meta = checkpoint.get("state_dict_ema_meta", {})
        expected_ema_meta = {}
        if EXPECTED_EMA_UPDATES is not None:
            expected_ema_meta["num_updates"] = EXPECTED_EMA_UPDATES
        if EXPECTED_LAST_EMA_STEP is not None:
            expected_ema_meta["last_ema_step"] = EXPECTED_LAST_EMA_STEP
        ema_meta_mismatches = {
            key: (ema_meta.get(key), value)
            for key, value in expected_ema_meta.items()
            if ema_meta.get(key) != value
        }
        if ema_meta_mismatches:
            raise RuntimeError(
                f"{checkpoint_path} EMA metadata mismatch: {ema_meta_mismatches}"
            )
    else:
        raw_state = checkpoint.get("state_dict", {})
        raw_encoder = {
            key: value
            for key, value in raw_state.items()
            if str(key).startswith("encoder.")
        }
        if len(raw_encoder) != 381:
            raise RuntimeError(
                f"{checkpoint_path} has {len(raw_encoder)} raw encoder "
                "tensors, expected 381"
            )
    del checkpoint
    gc.collect()

    source_sha256 = sha256_file(checkpoint_path)
    if not staged_path.is_file():
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{staged_path.name}.",
            suffix=".tmp",
            dir=staged_path.parent,
        )
        os.close(descriptor)
        temporary_path = pathlib.Path(temporary_name)
        try:
            stage_structtokenbench_checkpoint(
                checkpoint_path, temporary_path, source=ENCODER_SOURCE
            )
            try:
                # Same-filesystem link is an atomic create-if-absent.  Never
                # overwrite a staged artifact produced by another process.
                os.link(temporary_path, staged_path)
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)

    staged = torch.load(
        staged_path, map_location="cpu", weights_only=False, mmap=True
    )
    provenance = staged.get("canonical_lejepa_evaluation", {})
    expected_provenance = {
        "schema_version": 1,
        "encoder_source": ENCODER_SOURCE,
        "source_checkpoint_sha256": source_sha256,
        "source_global_step": EXPECTED_STEP,
        "latent_dim": 8,
        "projector": None,
    }
    provenance_mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    state = staged.get("state_dict", {})
    if len(state) != 381 or any(
        not str(key).startswith("online_encoder.") for key in state
    ):
        provenance_mismatches["state_dict"] = (
            f"{len(state)} tensors or non-online-encoder keys",
            "381 online_encoder.* tensors",
        )
    if provenance_mismatches:
        raise RuntimeError(
            f"staged {ENCODER_SOURCE} provenance mismatch for {checkpoint_id}: "
            f"{provenance_mismatches}"
        )
    del staged
    gc.collect()

    entry = (str(staged_path), "jepa")
    checkpoint_registry.CHECKPOINTS[checkpoint_id] = entry
    checkpoint_registry.ALL_CHECKPOINTS[checkpoint_id] = entry
    return {
        "checkpoint": str(checkpoint_path),
        "staged": str(staged_path),
        "source_checkpoint_sha256": source_sha256,
        "source_global_step": EXPECTED_STEP,
        "encoder_source": ENCODER_SOURCE,
        "optimizer_peak_lr": optimizer_config.get("lr"),
        "schedule_start_lr": schedule_config.get("start_lr"),
        "schedule_peak_step": schedule_config.get("peak_step"),
        "schedule_end_lr": schedule_config.get("end_lr"),
        "schedule_total_steps": schedule_config.get("total_steps"),
    }


def matched_selection(
    latent_caches: dict[str, dict[str, torch.Tensor]], labels: dict
) -> tuple[CombinedSelection, np.ndarray, str]:
    checkpoint_ids = list(latent_caches)
    reference_keys = set(latent_caches[checkpoint_ids[0]])
    for checkpoint_id in checkpoint_ids[1:]:
        if set(latent_caches[checkpoint_id]) != reference_keys:
            raise RuntimeError(
                f"{checkpoint_id} does not contain exactly the same proteins "
                "as the matched A checkpoint"
            )
    for protein_id in sorted(reference_keys):
        lengths = {
            checkpoint_id: int(cache[protein_id].shape[0])
            for checkpoint_id, cache in latent_caches.items()
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                f"latent residue counts differ for {protein_id}: {lengths}"
            )

    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, labels)
    if len(protein_ids) != EXPECTED_PROTEINS or selection.n != EXPECTED_RESIDUE_POOL:
        raise RuntimeError(
            "matched cohort changed: "
            f"proteins={len(protein_ids)} (expected {EXPECTED_PROTEINS}), "
            f"residue_pool={selection.n} (expected {EXPECTED_RESIDUE_POOL})"
        )
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    physical_sample = np.concatenate(
        [
            np.asarray(
                [f"{protein_id}:{int(index)}" for index in indices],
                dtype=object,
            )
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    sample_sha256 = hashlib.sha256(
        "\n".join(physical_sample.tolist()).encode()
    ).hexdigest()
    if sample_sha256 != EXPECTED_PHYSICAL_SAMPLE_SHA256:
        raise RuntimeError(
            "physical residue sample differs from the frozen canonical sample: "
            f"{sample_sha256}"
        )
    return selection, sampled, sample_sha256


def main() -> None:
    provenance = {
        checkpoint_id: stage_and_validate(checkpoint_id, spec)
        for checkpoint_id, spec in RUNS.items()
    }
    if ANALYSIS_PROFILE == EXTERNAL_PROFILE:
        only_spec = next(iter(RUNS.values()))
        checkpoint_registry.JEPA_ENCODER_CFG = pathlib.Path(only_spec["config"])
    checkpoint_ids = list(RUNS)
    extract(checkpoint_ids=checkpoint_ids, device="cuda")

    labels = load_labels()
    latent_caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in checkpoint_ids
    }
    selection, sampled, sample_sha256 = matched_selection(latent_caches, labels)

    amino_acids = selection.aa[sampled]
    secondary_structure = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate(
        [
            np.full(
                len(indices), len(labels[protein_id]["aa"]), dtype=np.float64
            )
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    normalized_position = (residue_index + 1.0) / lengths
    log_length = np.log(lengths)
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]

    rows: list[dict[str, object]] = []
    for checkpoint_id in checkpoint_ids:
        latent = selection.build_X(latent_caches[checkpoint_id])[sampled].astype(
            np.float32
        )
        neighbours = external_neighbours(latent, protein_index)
        exact_index_stats = knn_exact_index_stats(
            normalized(latent), residue_index
        )
        raw_prediction = residue_index[neighbours].mean(axis=1)
        within_bin_neighbours, within_bin_usable = within_length_bin_neighbours(
            latent, protein_index, lengths
        )
        within_raw_prediction = residue_index[
            within_bin_neighbours[within_bin_usable]
        ].mean(axis=1)
        within_raw_rho = spearmanr(
            residue_index[within_bin_usable], within_raw_prediction
        ).statistic

        all_valid = np.ones(N_POINTS, dtype=bool)
        raw_rho, raw_n = neighbour_spearman(
            residue_index, all_valid, neighbours
        )
        normalized_rho, normalized_n = neighbour_spearman(
            normalized_position, all_valid, neighbours
        )
        length_rho, length_n = neighbour_spearman(
            log_length, all_valid, neighbours
        )
        rsa_rho, rsa_n = neighbour_spearman(rsa, rsa_valid, neighbours)
        spectrum = spectrum_metrics(latent)
        position_quintile = np.minimum(
            (normalized_position * 5).astype(np.int64), 4
        )

        rows.append(
            {
                "checkpoint": checkpoint_id,
                "masked_kv_mode": RUNS[checkpoint_id]["mode"],
                **provenance[checkpoint_id],
                "proteins": EXPECTED_PROTEINS,
                "residue_pool": selection.n,
                "sampled_residues": len(sampled),
                "physical_sample_sha256": sample_sha256,
                "latent_dim": latent.shape[1],
                "residue_effective_rank": spectrum["effective_rank"],
                "residue_embedding_std": float(
                    torch.from_numpy(latent).float().std().item()
                ),
                "latent_dimension_std_min": spectrum[
                    "latent_dimension_std_min"
                ],
                "latent_dimension_std_max": spectrum[
                    "latent_dimension_std_max"
                ],
                "latent_isotropy_ratio": spectrum["latent_isotropy_ratio"],
                "aa_purity_at_10_cross_protein": purity(
                    amino_acids, neighbours
                ),
                "ss_purity_at_10_cross_protein": purity(
                    secondary_structure, neighbours
                ),
                "rsa_knn10_spearman": rsa_rho,
                "rsa_n": rsa_n,
                "log_chain_length_knn10_spearman": length_rho,
                "log_chain_length_n": length_n,
                "raw_index_knn10_spearman": raw_rho,
                "raw_index_n": raw_n,
                "normalized_position_knn10_spearman": normalized_rho,
                "normalized_position_n": normalized_n,
                "raw_index_partial_length_spearman": partial_spearman(
                    residue_index, raw_prediction, log_length
                ),
                "raw_index_within_16aa_length_bin_knn10_spearman": float(
                    within_raw_rho
                ),
                "raw_index_within_16aa_length_bin_n": int(
                    within_bin_usable.sum()
                ),
                "exact_index_at_10_cross_protein": float(
                    (
                        residue_index[neighbours]
                        == residue_index[:, None]
                    ).mean()
                ),
                "exact_index_at_10_all_neighbours": exact_index_stats[
                    "exact_index_purity_at_10"
                ],
                "nearest_neighbour_exact_index_all_neighbours": (
                    exact_index_stats["nearest_neighbor_exact_index"]
                ),
                "normalized_position_quintile_purity_at_10_cross_protein": (
                    purity(position_quintile, neighbours)
                ),
                "same_protein_fraction_at_10": same_protein_fraction(
                    normalized(latent), protein_index
                ),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "description": (
            "Matched cheap representation diagnostics for canonical residue-"
            "LeJEPA iterative versus visible-only K/V at "
            f"{EXPECTED_STEP} steps ({ANALYSIS_PROFILE})"
        ),
        "encoder_source": ENCODER_SOURCE.upper(),
        "no_tsne": True,
        "neighbours": {
            "metric": "cosine",
            "k": K,
            "cross_protein": True,
        },
        "selection": {
            "proteins": EXPECTED_PROTEINS,
            "aa_ss_valid_residue_pool": selection.n,
            "sampled_residues": len(sampled),
            "seed": SEED,
            "same_physical_residues": True,
            "physical_sample_sha256": sample_sha256,
        },
        "position": {
            "raw_index": "one-based local tensor-row ordinal within chain",
            "normalized_position": "one-based i/L",
            "length_control": [
                "partial Spearman after residualizing ranked log(chain length)",
                "cross-protein kNN restricted to 16-residue length bins",
            ],
        },
        "representation_statistics": (
            "computed on the same 15,000 sampled full-context residue latents"
        ),
        "runs": provenance,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {CSV_PATH}")
    print(f"wrote {MANIFEST_PATH}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
