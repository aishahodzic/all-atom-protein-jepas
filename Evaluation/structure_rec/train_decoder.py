"""Decoder reconstruction: train a fresh decoder against a frozen encoder,
measure RMSD (all-atom + CA-excluded) and seq_rec.

Encoder frozen (no grad) throughout; only the decoder's params are in the
optimizer. Latent tap is `mean` (deterministic, no reparam noise) -- matches
every other part of this evaluation suite (probes, collapse diagnostics,
768-dim), not the noisy z_latent the original joint AE training used; since
the encoder is frozen there's no KL term to compute either way.

Decoder architecture per group (see the Ca-conditioning code walkthrough
this was built from -- ca_coors_nm is ALWAYS passed to DecoderTransformer
and used as the coordinate-frame anchor for every variant; feats_seq only
gates whether the trunk additionally attends to it):
  - "ca-cond" group:      ca-cond's own decoder config (Ca as anchor AND attention feature)
  - "latent-only" group:  fully-latent's own decoder config (Ca as anchor ONLY)
Both configs are read directly from the real checkpoints' own saved
hyperparameters, not hand-written.

Training data: PDBLightningDataModule.train_dataloader() (real batching,
cluster-random sampling, includes GlobalRotationTransform -- matches how
the AE was actually trained; the no-rotation eval_transform_compose() from
data_pool.py is for held-out evaluation only, not training).
"""

import argparse
import csv
import pathlib
import sys
from contextlib import nullcontext
from itertools import cycle
from typing import Dict, List, Optional

import torch
import wandb
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path.insert(0, str(LINEAR_PROBING_DIR))

from checkpoints import ALL_CHECKPOINTS, CKPT_ROOT, load_frozen_encoder  # noqa: E402
from data_pool import DATA_DIR, EXCLUDE_IDS, build_protein_pool, eval_transform_compose  # noqa: E402

from proteinfoundation.datasets.pdb_data import PDBDataSelector, PDBDataSplitter, PDBLightningDataModule  # noqa: E402
from proteinfoundation.datasets.transforms import (  # noqa: E402
    CenterStructureTransform,
    ChainBreakPerResidueTransform,
    CoordsToNanometers,
    GlobalRotationTransform,
)
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list  # noqa: E402
from proteinfoundation.utils.coors_utils import nm_to_ang  # noqa: E402
from stereochemistry import stereochemical_loss  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parent
CKPT_DIR = OUT_DIR / "checkpoints"
CURVES_DIR = OUT_DIR / "curves"
RESULTS_CSV = OUT_DIR / "results.csv"
RESULTS_FIELDS = ["run_id", "encoder", "group", "decoder_style", "step",
                   "rmsd_all_atom", "rmsd_no_ca", "seq_rec_rate", "notes"]

CA_SLOT = 1  # OpenFold atom37 ordering, confirmed earlier in this session

DECODER_DONOR = {"ca-cond": "ca-cond", "fully-latent": "fully-latent"}

GROUP_OF = {
    "ca-cond": "ca-cond",
    "fully-latent": "latent-only",
    "jepa-mutation": "latent-only",
    "jepa-rand-mask-50": "latent-only",
    "jepa-rand-mask-70": "latent-only",
    "jepa-rand70-d8-independent-rotations-step100k": "latent-only",
    "le-jepa-mutation": "latent-only",
    "le-jepa-rand-mask-50": "latent-only",
    "le-jepa-rand-mask-70": "latent-only",
    "le-jepa-rand70-relative-position-rotation": "latent-only",
    "canonical-lejepa-v1-rotations-raw-step100k": "latent-only",
    "canonical-lejepa-v1-rotations-ema-step100k": "latent-only",
    "canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k": "latent-only",
}
DECODER_STYLE_OF = {
    "ca-cond": "ca-cond",
    "fully-latent": "fully-latent",
    "jepa-mutation": "fully-latent",
    "jepa-rand-mask-50": "fully-latent",
    "jepa-rand-mask-70": "fully-latent",
    "jepa-rand70-d8-independent-rotations-step100k": "fully-latent",
    "le-jepa-mutation": "fully-latent",
    "le-jepa-rand-mask-50": "fully-latent",
    "le-jepa-rand-mask-70": "fully-latent",
    "le-jepa-rand70-relative-position-rotation": "fully-latent",
    "canonical-lejepa-v1-rotations-raw-step100k": "fully-latent",
    "canonical-lejepa-v1-rotations-ema-step100k": "fully-latent",
    "canonical-lejepa-residue-perview-rot-middle-lr-raw-step10k": "fully-latent",
}


def load_donor_cfg_ae(decoder_style: str):
    """decoder_style: "ca-cond" or "fully-latent" -- the donor checkpoint whose
    saved cfg_ae.nn_ae.decoder config we reuse verbatim for the fresh decoder."""
    rel_path, _ = ALL_CHECKPOINTS[decoder_style]
    ckpt_path = pathlib.Path(rel_path)
    if not ckpt_path.is_absolute():
        ckpt_path = CKPT_ROOT / ckpt_path
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt["hyper_parameters"]["cfg_ae"]


def build_train_dataloader(batch_size: int = 8, num_workers: int = 8):
    """Real batched train-split dataloader -- same dataselector/datasplitter
    params as data_pool.py (identical to what all 8 checkpoints were trained
    with), but with the FULL training transform list (including
    GlobalRotationTransform, unlike the eval-only transform in data_pool.py)."""
    dataselector = PDBDataSelector(
        data_dir=str(DATA_DIR), fraction=1, molecule_type="protein",
        experiment_types=["diffraction", "EM"], min_length=50, max_length=256,
        oligomeric_min=None, oligomeric_max=None, best_resolution=0.0, worst_resolution=5.0,
        has_ligands=[], remove_ligands=[], remove_non_standard_residues=True,
        remove_pdb_unavailable=True, exclude_ids=EXCLUDE_IDS,
    )
    datasplitter = PDBDataSplitter(
        data_dir=str(DATA_DIR), train_val_test=[0.98, 0.019, 0.001],
        split_type="sequence_similarity", split_sequence_similarity=0.5,
        overwrite_sequence_clusters=False,
    )
    train_transforms = [
        CoordsToNanometers(), CenterStructureTransform(),
        GlobalRotationTransform(), ChainBreakPerResidueTransform(),
    ]
    dm = PDBLightningDataModule(
        data_dir=str(DATA_DIR), dataselector=dataselector, datasplitter=datasplitter,
        in_memory=False, format="cif", skip_unprocessed=True, batch_padding=True,
        sampling_mode="cluster-random", transforms=train_transforms,
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )
    dm.setup(stage="fit")
    return dm.train_dataloader()


def infinite(dl):
    while True:
        for batch in dl:
            yield batch


def compute_rmsd(output_dec: Dict, batch: Dict, exclude_ca: bool) -> torch.Tensor:
    """Masked, unaligned all-atom RMSD in Angstrom -- same formula as
    AutoEncoder.compute_struct_rec_loss's rmsd_no_align_a37_ang_justlog,
    just reimplemented here so we can optionally zero out the CA slot (index
    1) from the atom mask. Returns per-batch-element RMSD, shape [b]."""
    coors_nm_pred = output_dec["coors_nm"]
    coors_nm_true = batch["coords_nm"]
    mask = output_dec["residue_mask"]
    atom_mask_true = batch["coord_mask"] * mask[..., None]
    if exclude_ca:
        atom_mask_true = atom_mask_true.clone()
        atom_mask_true[..., CA_SLOT] = False
    err = coors_nm_true - coors_nm_pred
    err = err * mask[..., None, None] * atom_mask_true[..., None]
    err_ang = nm_to_ang(err)
    err_ang = torch.linalg.norm(err_ang, dim=-1) ** 2  # [b, n, 37]
    err_ang = err_ang * mask[..., None] * atom_mask_true
    nat = atom_mask_true.sum((-1, -2))
    return torch.sqrt(torch.sum(err_ang, dim=(-1, -2)) / nat.clamp(min=1))


@torch.no_grad()
def evaluate(
    model: AutoEncoder,
    protein_ids: List[str],
    pool,
    device: str,
    coordinate_mode: str = "ca-anchored",
    geometry_predicted_sequence_weight: Optional[float] = None,
    geometry_tolerance_factor: float = 12.0,
    geometry_clash_overlap: float = 1.5,
) -> Dict[str, float]:
    """Full-context, batch-of-1 eval over the given protein list (either the
    fixed calibration subset or the full test+val pool). Returns mean
    rmsd_all_atom / rmsd_no_ca / seq_rec_rate."""
    model.decoder.eval()
    transform = eval_transform_compose()
    id_to_idx = {pid: i for i, pid in enumerate(pool.protein_ids)}

    sum_rmsd_all, sum_rmsd_noca, sum_seq_rec, n = 0.0, 0.0, 0.0, 0
    geometry_sums = {
        "geometry_violation": 0.0,
        "geometry_true_violation": 0.0,
        "geometry_predicted_violation": 0.0,
    }
    for pid in protein_ids:
        idx = id_to_idx[pid]
        graph = transform(pool.dataset[idx])
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
        batch = batch.to(device)
        mask = batch["mask"]
        ca_coors_nm = batch["coords_nm"][..., CA_SLOT, :] * mask[..., None]

        mean = model.encoder(batch)["mean"]
        input_decoder = {"z_latent": mean, "residue_mask": mask, "mask": mask}
        if coordinate_mode == "ca-anchored":
            input_decoder["ca_coors_nm"] = ca_coors_nm
        output_dec = model.decoder(input_decoder)

        rmsd_all = compute_rmsd(output_dec, batch, exclude_ca=False)
        rmsd_noca = compute_rmsd(output_dec, batch, exclude_ca=True)
        seq_losses = model.compute_seq_rec_loss(output_dec, batch, weight=1.0)

        sum_rmsd_all += rmsd_all.sum().item()
        sum_rmsd_noca += rmsd_noca.sum().item()
        sum_seq_rec += seq_losses["seq_rec_rate_justlog"].sum().item()
        if geometry_predicted_sequence_weight is not None:
            geometry = stereochemical_loss(
                output_dec,
                batch,
                predicted_sequence_weight=geometry_predicted_sequence_weight,
                tolerance_factor=geometry_tolerance_factor,
                clash_overlap_tolerance=geometry_clash_overlap,
            )
            geometry_sums["geometry_violation"] += geometry["total"].item()
            geometry_sums["geometry_true_violation"] += geometry["true_total"].item()
            geometry_sums["geometry_predicted_violation"] += geometry[
                "predicted_total"
            ].item()
        n += 1

    model.decoder.train()
    metrics = dict(
        rmsd_all_atom=sum_rmsd_all / n,
        rmsd_no_ca=sum_rmsd_noca / n,
        seq_rec_rate=sum_seq_rec / n,
    )
    if geometry_predicted_sequence_weight is not None:
        metrics.update({name: value / n for name, value in geometry_sums.items()})
    return metrics


def pick_fixed_eval_subset(pool, size: int, seed: int) -> List[str]:
    """Draws from val only -- test stays untouched until the final eval."""
    import random
    rng = random.Random(seed)
    ids = [pid for pid in pool.protein_ids if pool.origin[pid] == "val"]
    rng.shuffle(ids)
    return ids[:size]


def test_only_ids(pool) -> List[str]:
    return [pid for pid in pool.protein_ids if pool.origin[pid] == "test"]


def train_one(
    run_id: str,
    encoder_id: str,
    decoder_style: str,
    group: str,
    max_steps: int,
    eval_every: int = 2000,
    calibration: bool = False,
    patience_evals: int = 5,
    patience_threshold: float = 0.01,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 42,
    device: Optional[str] = None,
    eval_subset_size: int = 200,
    use_wandb: bool = True,
    resume_from: Optional[str] = None,
    resume_step: int = 0,
    encoder_source: str = "online",
    geometry_weight: float = 0.0,
    geometry_predicted_sequence_weight: float = 0.5,
    geometry_tolerance_factor: float = 12.0,
    geometry_clash_overlap: float = 1.5,
    save_every: int = 10000,
    evaluate_before_training: bool = False,
    coordinate_mode: str = "ca-anchored",
    precision: str = "fp32",
):
    """resume_from/resume_step: continue a calibration decoder's weights
    (e.g. an early-stopped or already-at-budget calibration checkpoint)
    instead of training fresh -- weights only, optimizer state restarts
    clean. If resume_step >= max_steps, the training loop is a no-op and
    this just runs the final full-pool eval on the loaded weights."""
    if geometry_weight < 0:
        raise ValueError("geometry_weight must be non-negative")
    if save_every <= 0:
        raise ValueError("save_every must be positive")
    if coordinate_mode not in {"ca-anchored", "fully-latent"}:
        raise ValueError(f"unknown coordinate_mode={coordinate_mode!r}")
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unknown precision={precision!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if precision == "bf16":
        if not str(device).startswith("cuda"):
            raise ValueError("BF16 decoder training requires a CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("the allocated CUDA device does not support BF16")
    torch.manual_seed(seed)

    def autocast_context():
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    encoder = load_frozen_encoder(
        encoder_id, device=device, encoder_source=encoder_source
    )

    donor_cfg_ae = load_donor_cfg_ae(decoder_style)
    model = AutoEncoder(cfg_ae=donor_cfg_ae, store_dir=str(OUT_DIR / "tmp_ae" / run_id))
    model.encoder = encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model = model.to(device)
    model.encoder.eval()
    model.decoder.train()

    if resume_from:
        model.decoder.load_state_dict(torch.load(resume_from, map_location=device))
        logger.info(f"{run_id}: resumed decoder weights from {resume_from} at step={resume_step}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_dl = build_train_dataloader(batch_size=batch_size)
    eval_pool = build_protein_pool()
    eval_subset = pick_fixed_eval_subset(eval_pool, eval_subset_size, seed=seed)

    curve_path = CURVES_DIR / f"{run_id}_curve.csv"
    curve_fields = ["step", "train_loss", "rmsd_all_atom", "rmsd_no_ca", "seq_rec_rate"]
    if geometry_weight > 0:
        curve_fields.extend(
            [
                "geometry_violation",
                "geometry_true_violation",
                "geometry_predicted_violation",
            ]
        )
    with open(curve_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=curve_fields).writeheader()

    if use_wandb:
        wb_run = wandb.init(
            project="protein_decoder_reconstruction", name=run_id,
            config=dict(encoder=encoder_id, decoder_style=decoder_style, group=group,
                        max_steps=max_steps, lr=lr, batch_size=batch_size, seed=seed,
                        calibration=calibration, encoder_source=encoder_source,
                        coordinate_mode=coordinate_mode,
                        geometry_weight=geometry_weight,
                        geometry_predicted_sequence_weight=geometry_predicted_sequence_weight,
                        geometry_tolerance_factor=geometry_tolerance_factor,
                        geometry_clash_overlap=geometry_clash_overlap,
                        precision=precision),
        )

    best_score, best_step, best_decoder_state, last_loss = (
        float("inf"), resume_step, None, None
    )
    if evaluate_before_training:
        initial_metrics = evaluate(
            model,
            eval_subset,
            eval_pool,
            device,
            coordinate_mode=coordinate_mode,
            geometry_predicted_sequence_weight=(
                geometry_predicted_sequence_weight if geometry_weight > 0 else None
            ),
            geometry_tolerance_factor=geometry_tolerance_factor,
            geometry_clash_overlap=geometry_clash_overlap,
        )
        initial_score = (
            initial_metrics["rmsd_no_ca"]
            if coordinate_mode == "ca-anchored"
            else initial_metrics["rmsd_all_atom"]
        )
        if geometry_weight > 0:
            initial_score += geometry_weight * initial_metrics["geometry_violation"]
        best_score = initial_score
        best_decoder_state = {
            key: value.detach().cpu().clone()
            for key, value in model.decoder.state_dict().items()
        }
        with open(curve_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=curve_fields).writerow(
                dict(step=resume_step, train_loss=float("nan"), **initial_metrics)
            )
        logger.info(
            f"{run_id} initial step={resume_step} "
            f"rmsd_noca={initial_metrics['rmsd_no_ca']:.4f} "
            f"seq_rec={initial_metrics['seq_rec_rate']:.4f} "
            f"score={initial_score:.4f}"
        )
        if use_wandb:
            wb_run.log(dict(step=resume_step, **initial_metrics))

    step = resume_step - 1  # so `step + 1` after the loop is correct even if the loop body never runs
    train_iter = infinite(train_dl)
    for step in range(resume_step, max_steps):
        batch = next(train_iter)
        batch = batch.to(device)
        mask = batch["mask_dict"]["coords"][..., 0, 0]
        batch["mask"] = mask
        ca_coors_nm = batch["coords_nm"][..., CA_SLOT, :] * mask[..., None]

        with autocast_context():
            with torch.no_grad():
                mean = model.encoder(batch)["mean"]
            input_decoder = {"z_latent": mean, "residue_mask": mask, "mask": mask}
            if coordinate_mode == "ca-anchored":
                input_decoder["ca_coors_nm"] = ca_coors_nm
            output_dec = model.decoder(input_decoder)

            losses = {}
            losses.update(model.compute_struct_rec_loss(output_dec, batch, reduce_mode="sum", loss_ty="l2", weight=1.0))
            losses.update(model.compute_seq_rec_loss(output_dec, batch, weight=1.0))
            if geometry_weight > 0:
                geometry = stereochemical_loss(
                    output_dec,
                    batch,
                    predicted_sequence_weight=geometry_predicted_sequence_weight,
                    tolerance_factor=geometry_tolerance_factor,
                    clash_overlap_tolerance=geometry_clash_overlap,
                )
                losses["stereochemistry"] = geometry_weight * geometry["total"]
            loss = sum(v.mean() for k, v in losses.items() if "_justlog" not in k)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

        step_1 = step + 1
        if step_1 % eval_every == 0 or step_1 == max_steps:
            metrics = evaluate(
                model,
                eval_subset,
                eval_pool,
                device,
                coordinate_mode=coordinate_mode,
                geometry_predicted_sequence_weight=(
                    geometry_predicted_sequence_weight if geometry_weight > 0 else None
                ),
                geometry_tolerance_factor=geometry_tolerance_factor,
                geometry_clash_overlap=geometry_clash_overlap,
            )
            geometry_message = (
                f" geometry={metrics['geometry_violation']:.4f}"
                if geometry_weight > 0
                else ""
            )
            logger.info(
                f"{run_id} step={step_1}/{max_steps} train_loss={last_loss:.4f} "
                f"rmsd_all={metrics['rmsd_all_atom']:.4f} rmsd_noca={metrics['rmsd_no_ca']:.4f} "
                f"seq_rec={metrics['seq_rec_rate']:.4f}{geometry_message}"
            )
            with open(curve_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=curve_fields).writerow(dict(
                    step=step_1, train_loss=last_loss, **metrics))
            if use_wandb:
                wb_run.log(dict(step=step_1, train_loss=last_loss, **metrics))

            if step_1 % save_every == 0 or step_1 == max_steps:
                torch.save(model.decoder.state_dict(), CKPT_DIR / f"{run_id}_step{step_1}.pt")

            stopping_score = (
                metrics["rmsd_no_ca"]
                if coordinate_mode == "ca-anchored"
                else metrics["rmsd_all_atom"]
            )
            if geometry_weight > 0:
                stopping_score += geometry_weight * metrics["geometry_violation"]
            if stopping_score < best_score * (1 - patience_threshold):
                best_score, best_step = stopping_score, step_1
                best_decoder_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.decoder.state_dict().items()
                }
            elif calibration and (step_1 - best_step) >= patience_evals * eval_every:
                logger.info(
                    f"{run_id}: auto-stop at step={step_1} (no >{patience_threshold*100:.0f}% "
                    f"improvement in {patience_evals} evals since step={best_step}, best_score={best_score:.4f})"
                )
                break

    trained_through_step = step + 1
    if best_decoder_state is not None:
        model.decoder.load_state_dict(best_decoder_state)
        logger.info(
            f"{run_id}: restored best validation decoder from step={best_step} "
            f"before the held-out test evaluation"
        )

    test_ids = test_only_ids(eval_pool)
    final_metrics = evaluate(
        model,
        test_ids,
        eval_pool,
        device,
        coordinate_mode=coordinate_mode,
        geometry_predicted_sequence_weight=(
            geometry_predicted_sequence_weight if geometry_weight > 0 else None
        ),
        geometry_tolerance_factor=geometry_tolerance_factor,
        geometry_clash_overlap=geometry_clash_overlap,
    )
    logger.info(f"{run_id} FINAL (test only, n={len(test_ids)}): {final_metrics}")
    torch.save(model.decoder.state_dict(), CKPT_DIR / f"{run_id}_final.pt")

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(
            run_id=run_id, encoder=encoder_id, group=group, decoder_style=decoder_style,
            step=best_step, rmsd_all_atom=final_metrics["rmsd_all_atom"],
            rmsd_no_ca=final_metrics["rmsd_no_ca"], seq_rec_rate=final_metrics["seq_rec_rate"],
            notes=(
                f"{'calibration; ' if calibration else ''}"
                f"trained_through_step={trained_through_step}; "
                f"geometry_weight={geometry_weight}; "
                f"geometry_predicted_sequence_weight={geometry_predicted_sequence_weight}"
                f"; coordinate_mode={coordinate_mode}; precision={precision}"
            ),
        ))

    if use_wandb:
        wb_run.log({f"final_{k}": v for k, v in final_metrics.items()})
        wb_run.finish()

    return final_metrics, best_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder-style", default=None, help="Defaults from DECODER_STYLE_OF; required for random-init")
    parser.add_argument("--group", default=None, help="Defaults from GROUP_OF; required for random-init")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-steps", type=int, default=150_000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--patience-evals", type=int, default=5)
    parser.add_argument("--patience-threshold", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume-from", default=None, help="Path to a saved decoder state_dict to continue from")
    parser.add_argument("--resume-step", type=int, default=0, help="Step count the --resume-from checkpoint represents")
    parser.add_argument(
        "--encoder-source", choices=("online", "target"), default="online",
        help="JEPA encoder branch to decode. Use target for decoding predictor outputs.",
    )
    parser.add_argument(
        "--geometry-weight",
        type=float,
        default=0.0,
        help="Weight of OpenFold stereochemical violations; zero preserves legacy training.",
    )
    parser.add_argument(
        "--geometry-predicted-sequence-weight",
        type=float,
        default=0.5,
        help="Fraction of geometry loss evaluated with predicted rather than true residue types.",
    )
    parser.add_argument("--geometry-tolerance-factor", type=float, default=12.0)
    parser.add_argument("--geometry-clash-overlap", type=float, default=1.5)
    parser.add_argument("--eval-subset-size", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Floating-point precision for training forwards; evaluation remains FP32.",
    )
    parser.add_argument("--evaluate-before-training", action="store_true")
    parser.add_argument(
        "--coordinate-mode",
        choices=("ca-anchored", "fully-latent"),
        default="ca-anchored",
        help=(
            "Use native C-alpha coordinates as decoder anchors, or train the "
            "original fully latent decoder without them."
        ),
    )
    args = parser.parse_args()

    decoder_style = args.decoder_style or DECODER_STYLE_OF[args.encoder]
    group = args.group or GROUP_OF[args.encoder]
    run_id = args.run_id or (f"{args.encoder}_calib" if args.calibration else args.encoder)

    train_one(
        run_id=run_id, encoder_id=args.encoder, decoder_style=decoder_style, group=group,
        max_steps=args.max_steps, eval_every=args.eval_every, calibration=args.calibration,
        patience_evals=args.patience_evals, patience_threshold=args.patience_threshold,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, use_wandb=not args.no_wandb,
        resume_from=args.resume_from, resume_step=args.resume_step,
        encoder_source=args.encoder_source,
        geometry_weight=args.geometry_weight,
        geometry_predicted_sequence_weight=args.geometry_predicted_sequence_weight,
        geometry_tolerance_factor=args.geometry_tolerance_factor,
        geometry_clash_overlap=args.geometry_clash_overlap,
        eval_subset_size=args.eval_subset_size,
        save_every=args.save_every,
        evaluate_before_training=args.evaluate_before_training,
        coordinate_mode=args.coordinate_mode,
        precision=args.precision,
    )
