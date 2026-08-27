"""JEPA protein model.

Context encoder receives n-k residues, and outputs n-k residue latents.
Predictor receives n-k residues, and predicts the k target residues.
Target encoder sees n residues, and outputs k target residues.

Three components:
  * online encoder  - the existing VAE encoder architecture, trained by backprop. We
                      use only its ``mean`` output as the deterministic latent (no
                      sampling, no KL).
  * EMA target encoder - an exact copy of the online encoder, updated as an EMA of the online weights. It is never trained by
                      backprop (requires_grad=False) and produces the prediction targets.
  * predictor       - cross-attention network mapping context latents -> target latents.


The loss is a smooth-L1 between the predictor's output and the encoder's target
latents, over target residues only. 
"""

import copy
from typing import Dict, List

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.decomposition import PCA

from proteinfoundation.jepa.masking import (
    apply_mask_to_latents,
    block_residue_mask,
    random_residue_mask,
    spatial_3d_block_mask,
)
from proteinfoundation.jepa.predictor import JEPAPredictor
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.coors_utils import sample_uniform_rotation


def covariance_effective_rank(values: torch.Tensor) -> torch.Tensor:
    """Effective rank of a centered ``[N, D]`` embedding covariance.

    Per-dimension standard deviation misses dimensional collapse when every
    output dimension carries a copy of the same scalar.  Effective rank is 1
    for that failure and approaches D for an isotropic D-dimensional cloud.
    """
    if values.ndim != 2:
        raise ValueError("effective rank expects an [N, D] embedding matrix")
    if values.shape[0] < 2:
        return values.new_zeros((), dtype=torch.float32)
    centered = values.detach().float() - values.detach().float().mean(
        dim=0, keepdim=True
    )
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    total_variance = variance.sum()
    if total_variance <= 0:
        return total_variance
    probabilities = variance / total_variance
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization from LeJEPA.

    Random one-dimensional projections are tested against a standard normal.
    This jointly discourages zero-variance and redundant/low-rank embeddings.
    """

    def __init__(self, knots: int = 17, num_proj: int = 256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [T, N, D]; these experiments use T=1.
        directions = torch.randn(
            embeddings.size(-1), self.num_proj, device=embeddings.device
        )
        directions = directions.div_(directions.norm(p=2, dim=0))
        projected = (embeddings @ directions).unsqueeze(-1) * self.t
        error = (
            (projected.cos().mean(-3) - self.phi).square()
            + projected.sin().mean(-3).square()
        )
        statistic = (error @ self.weights) * embeddings.size(-2)
        return statistic.mean()


def independently_rotate_target_batch(batch: Dict) -> Dict:
    """Give each target protein an independent uniform global rotation.

    The dataloader has already applied a uniform rotation shared by the context
    and target views.  Right-multiplying only the target ``coords_nm`` by a new
    independent Haar-uniform rotation makes the resulting target frame uniform
    and independent of the context frame.  This mirrors
    ``GlobalRotationTransform``, which deliberately rotates ``coords_nm`` only;
    invariant pair features computed from ``coords`` remain unchanged.
    """
    coords_nm = batch["coords_nm"]
    rotations = sample_uniform_rotation(
        shape=(coords_nm.shape[0],),
        dtype=coords_nm.dtype,
        device=coords_nm.device,
    )
    rotated = dict(batch)
    rotated["coords_nm"] = torch.einsum("bnac,bcd->bnad", coords_nm, rotations)
    return rotated


def mask_batch_for_encoder(batch: Dict, context_mask: torch.Tensor) -> Dict:
    """Mask a batch so the online encoder only sees context residues (n-k residues).
     
    Target-residue coordinates are zeroed out and marked invalid.

    Args:
        batch:         dict with "coords" [B, L, 37, 3], "coords_nm" [B, L, 37, 3],
                        and "mask_dict"["coords"] [B, L, 37, 3] bool
        context_mask:  [B, L] bool, True for context residues

    Returns a shallow copy of ``batch`` with "coords", "coords_nm", and "mask_dict"
    replaced; every other key is the same reference as in the input batch. The input
    batch itself is not modified.
    """
    keep = context_mask[..., None, None]  # [B, L, 1, 1], broadcasts over (37 atoms, 3 coords)

    masked_batch = dict(batch)  # shallow copy: only the keys below are overwritten
    masked_batch["coords"] = batch["coords"] * keep  # [B, L, 37, 3], target positions zeroed
    masked_batch["coords_nm"] = batch["coords_nm"] * keep  # [B, L, 37, 3], target positions zeroed

    masked_batch["mask_dict"] = dict(batch["mask_dict"])
    masked_batch["mask_dict"]["coords"] = batch["mask_dict"]["coords"] & keep  # [B, L, 37, 3]

    return masked_batch


def mask_batch_for_encoder_chemmask_aa_only(batch: Dict, context_mask: torch.Tensor) -> Dict:
    """Chemical masking scheme 1 ("AA only, no coords"): EVERY real residue's AA
    identity reaches the context encoder, context and target alike. Only geometry
    is masked, and only on target (non-context) residues.

    Geometry is hidden via ``coord_mask`` (all 37 atom slots set False on target
    residues), NOT by zeroing coordinates -- "coords" and "coords_nm" are left
    untouched everywhere. ``mask_dict["coords"]`` is also left untouched (unlike
    ``mask_batch_for_encoder``): EncoderTransformer.forward re-derives its
    per-residue mask from mask_dict["coords"][..., 0, 0] and does
    ``seqs = seq_f_repr * mask[..., None]`` AFTER all sequence features (including
    x1_aatype) are computed (encoder.py:119-128) -- if mask_dict["coords"] were
    False on target residues (as it is in mask_batch_for_encoder, where it's used
    to exclude them entirely), that blanket step would zero AA identity too. Only
    coord_mask governs geometry visibility here; mask_dict["coords"] keeps every
    real (non-padding) residue "valid" so identity survives that step for all of
    them.

    Args:
        batch:         dict with "coord_mask" [B, L, 37] bool and "mask_dict"["coords"]
                        [B, L, 37, 3] bool; "coords"/"coords_nm"/"residue_type" are
                        read only via the returned shallow copy's shared references
        context_mask:  [B, L] bool, True for the 30% residues keeping full geometry

    Returns a shallow copy of ``batch`` with only "coord_mask" replaced; every other
    key (coords, coords_nm, residue_type, mask_dict, ...) is the same reference as
    in the input batch, untouched. The input batch itself is not modified.
    """
    # Target (non-context) residues lose geometry: coord_mask -> False on every
    # atom slot. Context residues' coord_mask is untouched (real, per-atom values
    # from the raw structure). AA identity is never touched here at all -- it
    # flows through unchanged for every real residue, context and target alike.
    strip_geometry = (~context_mask)[..., None]  # [B, L, 1], True for target residues, broadcasts over 37 atom slots

    masked_batch = dict(batch)  # shallow copy: only "coord_mask" is overwritten
    masked_batch["coord_mask"] = batch["coord_mask"] & (~strip_geometry)

    return masked_batch


def mask_batch_for_encoder_chemmask_partial_coordinates(batch: Dict, mask_pct: float = 0.7) -> Dict:
    """Variant 1 ("per-atom coordinate masking"): every residue keeps
    AA identity; each residue independently has ``mask_pct`` of its OWN valid atom
    slots hidden (only slots already True in coord_mask are eligible -- absent atoms,
    e.g. Glycine's CG, stay False). No context/target residue split: every residue is
    both context (contributes its unmasked atoms) and target (its full-atom latent is
    predicted) -- use ``apply_mask_to_latents(latents, valid_mask, valid_mask)`` for
    the overlapping context=target set, no new function needed there.

    Same mechanism as chemmask_aa_only: coord_mask carries the mask, mask_dict is
    untouched (protects AA identity through EncoderTransformer's blanket step).

    Args:
        batch:     dict with "coord_mask" [B, L, 37] bool
        mask_pct:  fraction of each residue's valid atoms to hide

    Returns a shallow copy of ``batch`` with only "coord_mask" replaced. The input
    batch itself is not modified.
    """
    coord_mask = batch["coord_mask"]  # [B, L, 37]
    n_valid = coord_mask.sum(dim=-1, keepdim=True)  # [B, L, 1]
    n_hide = (n_valid * mask_pct).round().long()  # [B, L, 1]

    # rank each atom slot's random score within its residue, restricted to valid slots
    # (invalid slots pushed above any valid score so they're never among the n_hide lowest)
    scores = torch.rand_like(coord_mask, dtype=torch.float32).masked_fill(~coord_mask, 2.0)
    ranks = scores.argsort(dim=-1).argsort(dim=-1)  # [B, L, 37], 0 = lowest score in this residue
    hide = ranks < n_hide  # [B, L, 37], the mask_pct lowest-scored valid atoms per residue

    masked_batch = dict(batch)
    masked_batch["coord_mask"] = coord_mask & (~hide)
    return masked_batch


def mask_batch_for_encoder_chemmask_partial_coords_and_identity(
    batch: Dict, atom_mask_pct: float = 0.7, identity_mask_pct: float = 0.5
) -> Dict:
    """Variant 2: Variant 1 coordinate masking plus independent AA masking.

    Coordinates are masked identically to ``chemmask_partial_coordinates``. Then,
    for each protein, exactly ``round(identity_mask_pct * n_valid)`` valid residues
    have their residue-type feature hidden by setting
    ``mask_dict['residue_type']`` to False. Coordinate validity is deliberately
    untouched, so identity-hidden residues remain context and target positions.
    """
    masked_batch = mask_batch_for_encoder_chemmask_partial_coordinates(
        batch, mask_pct=atom_mask_pct
    )
    valid_residues = batch["mask_dict"]["coords"][..., 0, 0].bool()  # [B, L]
    n_valid = valid_residues.sum(dim=-1, keepdim=True)
    n_hide = (n_valid * identity_mask_pct).round().long()

    # Draw a separate per-residue permutation from the coordinate-mask draw.
    scores = torch.rand(valid_residues.shape, device=valid_residues.device)
    scores = scores.masked_fill(~valid_residues, 2.0)
    ranks = scores.argsort(dim=-1).argsort(dim=-1)
    hide_identity = valid_residues & (ranks < n_hide)

    masked_batch["mask_dict"] = dict(batch["mask_dict"])
    masked_batch["mask_dict"]["residue_type"] = (
        batch["mask_dict"]["residue_type"].bool() & ~hide_identity
    )
    return masked_batch


def uniform_sampling(exclude_idx: torch.Tensor) -> torch.Tensor:
    """Samples one amino-acid index uniformly from the 20 AAs, excluding `exclude_idx`.

    Args:
        exclude_idx: [B] long, AA index to exclude per element; only 0-19 is excludable

    Returns:
        [B] long, sampled AA index; != exclude_idx whenever exclude_idx was in [0, 19]
    """
    B = exclude_idx.shape[0]
    device = exclude_idx.device
    can_exclude = (exclude_idx >= 0) & (exclude_idx <= 19)  # [B]
    safe_exclude = exclude_idx.clamp(0, 19)  # [B]

    full = torch.arange(20, device=device).expand(B, 20)  # [B, 20]
    remaining = full[full != safe_exclude.unsqueeze(1)].view(B, 19)  # [B, 19]
    pick_19 = torch.randint(0, 19, (B,), device=device)  # [B]
    excluded_choice = remaining[torch.arange(B, device=device), pick_19]  # [B]

    unrestricted_choice = torch.randint(0, 20, (B,), device=device)  # [B]
    return torch.where(can_exclude, excluded_choice, unrestricted_choice)  # [B]


def sample_mutation(valid_mask: torch.Tensor, residue_type: torch.Tensor):
    """Samples one random point mutation per protein.

    Args:
        valid_mask:   [B, L] bool, True on non-padding residues
        residue_type: [B, L] long, current (wildtype) amino-acid index (0-19) per residue

    Returns:
        mutation_positions: [B] long, sampled residue position per protein
        mutated_aa: [B] long, sampled new amino-acid index (0-19) per protein
    """
    B, L = valid_mask.shape
    device = valid_mask.device

    scores = torch.rand(B, L, device=device)  # [B, L]
    scores = scores.masked_fill(~valid_mask, float("-inf"))  # [B, L]
    mutation_positions = scores.argmax(dim=-1)  # [B]

    wildtype_aa = residue_type[torch.arange(B, device=device), mutation_positions]  # [B]
    mutated_aa = uniform_sampling(wildtype_aa)  # [B]

    return mutation_positions, mutated_aa  # [B] long, [B] long


def apply_mutation(batch: Dict, position: torch.Tensor, mutated_aa: torch.Tensor) -> Dict:
    """Returns a copy of `batch` with one residue's identity mutated per protein.

    Args:
        batch:          dict with "residue_type" [B, L] long
        position:       [B] long, mutation position (index into L) per protein
        mutated_aa: [B] long, new amino-acid index (0-19) per protein

    Returns a shallow copy of `batch` with "residue_type" replaced at `[b, position[b]]`
    for each protein b; every other key is the same reference as in the input batch.
    """
    B = batch["residue_type"].shape[0]
    residue_type = batch["residue_type"].clone()  # [B, L]
    residue_type[torch.arange(B, device=residue_type.device), position] = mutated_aa  # [B, L]

    mutated_batch = dict(batch)
    mutated_batch["residue_type"] = residue_type  # [B, L]
    return mutated_batch


class JEPAModel(L.LightningModule):
    def __init__(
        self,
        cfg,
        ema_decay=0.996,
        use_action_conditioning: bool = False,
        action_reg_warmup_steps: int = 25000,
        action_reg_anneal_steps: int = 50000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.ema_decay = ema_decay
        self.latent_dim = self.cfg.nn_ae["latent_z_dim"]  # 8
        self.baseline_run_layernorm = self.cfg.get("baseline_run_layernorm", False)
        self.baseline_run_ema = self.cfg.get("baseline_run_ema", False)
        self.baseline_run_predictor = self.cfg.get("baseline_run_predictor", False)
        self.jepa_sigreg = bool(self.cfg.get("jepa_sigreg", False))
        self.jepa_sigreg_weight = float(self.cfg.get("jepa_sigreg_weight", 0.0))
        self.jepa_independent_target_rotation = bool(
            self.cfg.get("jepa_independent_target_rotation", False)
        )
        if self.jepa_sigreg_weight < 0:
            raise ValueError("jepa_sigreg_weight must be non-negative")
        if self.jepa_sigreg:
            self.sigreg = SIGReg(
                num_proj=int(self.cfg.get("jepa_sigreg_num_proj", 256))
            )
        self._sigreg_equal_weight_samples: List[float] = []
        self._last_ema_update_step = -1

        if self.baseline_run_ema:
            self.ema_schedule_steps = self.cfg.opt.get("max_steps", None)
            if not isinstance(self.ema_schedule_steps, int) or self.ema_schedule_steps < 2:
                raise ValueError(
                    "baseline_run_ema=true requires opt.max_steps to be an integer >= 2 "
                    "so the 0.996-to-1.0 EMA schedule has a defined endpoint."
                )
        else:
            self.ema_schedule_steps = None

        self.use_action_conditioning = use_action_conditioning
        self.action_reg_warmup_steps = action_reg_warmup_steps
        self.action_reg_anneal_steps = action_reg_anneal_steps
        if self.use_action_conditioning:
            self.mutation_embedding = nn.Embedding(20, 128)  # matches predictor hidden_dim

        # Online encoder: trained via backprop, reuses the existing encoder architecture
        self.online_encoder = EncoderTransformer(**self.cfg.nn_ae)

        # EMA target encoder: exact copy of the online encoder, same initial weights,
        # not trained via backprop (excluded from the optimizer)
        self.ema_encoder = copy.deepcopy(self.online_encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False

        # Predictor: context latents + positions -> predicted target latents.
        # max_len must be >= the dataset's max crop length (256).
        predictor_cfg = self.cfg.get("predictor", {}) or {}
        max_len = predictor_cfg.get("max_len", 256)
        use_self_attn = predictor_cfg.get("use_self_attn", False)
        predictor_type = (
            "mask_token_transformer" if self.baseline_run_predictor else "cross_attention"
        )
        self.predictor = JEPAPredictor(
            latent_z_dim=self.latent_dim,
            max_len=max_len,
            use_self_attn=use_self_attn,
            predictor_type=predictor_type,
        )

        # For scaling/* logging, mirroring the autoencoder (proteinfoundation/partial_autoencoder/autoencoder.py)
        self.nsamples_processed = 0
        self.nparams_enc = sum(
            p.numel() for p in self.online_encoder.parameters() if p.requires_grad
        )
        self.nparams_predictor = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )

        self.lr_scheduler = None  # set in configure_optimizers, stepped manually in on_validation_end
        self._pending_lr_scheduler_state = None  # applied in configure_optimizers once the scheduler exists

        # Grace period: don't let the LR scheduler react for the first N steps
        self._lr_warmup_steps = 30000
        self._lr_scheduler_seeded = False  # reseed .best once, right when grace ends

    def on_save_checkpoint(self, checkpoint):
        checkpoint["nsamples_processed"] = self.nsamples_processed
        checkpoint["lr_scheduler_state"] = self.lr_scheduler.state_dict()
        checkpoint["lr_scheduler_seeded"] = self._lr_scheduler_seeded

    def on_load_checkpoint(self, checkpoint):
        try:
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except KeyError:
            self.nsamples_processed = 0
        if "lr_scheduler_state" in checkpoint:
            self._pending_lr_scheduler_state = checkpoint["lr_scheduler_state"]
        elif checkpoint.get("lr_schedulers"):
            # checkpoints saved before the switch to manual stepping
            # kept scheduler state under Lightning's own "lr_schedulers" key instead.
            self._pending_lr_scheduler_state = checkpoint["lr_schedulers"][0]
        # defaults False for checkpoints saved before this key existed, so the reseed
        # below still fires once this (resumed) run crosses _lr_warmup_steps again
        self._lr_scheduler_seeded = checkpoint.get("lr_scheduler_seeded", False)

    def on_validation_end(self):
        # Manually stepped, not registered via configure_optimizers' scheduler dict:
        if self.trainer.sanity_checking:
            return
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return

        if self.global_step < self._lr_warmup_steps:
            return  # grace period: don't step or seed the scheduler yet

        if not self._lr_scheduler_seeded:
            # seed with val_loss AT grace-end, not the min seen during grace ( min
            # can reflect an early not-yet-diverged online/EMA state, not real learning)
            self.lr_scheduler.best = float(val_loss)
            self.lr_scheduler.num_bad_epochs = 0
            self._lr_scheduler_seeded = True

        self.lr_scheduler.step(val_loss)

    def _action_sensitivity_lambda(self) -> float:
        """Anneals the action_reg weight 0.01 -> 0.001 over action_reg_anneal_steps, after
        action_reg_warmup_steps at the fixed 0.01."""
        gs = self.global_step
        if gs < self.action_reg_warmup_steps:
            return 0.01
        if gs < self.action_reg_warmup_steps + self.action_reg_anneal_steps:
            progress = (gs - self.action_reg_warmup_steps) / self.action_reg_anneal_steps
            return max(0.001, 0.01 - (0.01 - 0.001) * progress)
        return 0.001

    def _encode_mean(self, encoder, batch: Dict) -> torch.Tensor:
        """Encoder that only returns the deterministic mean latent (no var)."""
        return encoder(batch)["mean"]  # [B, L, latent_z_dim]

    def compute_loss(self, batch: Dict, stage: str = "train") -> Dict[str, torch.Tensor]:
        """Pure JEPA latent-prediction loss for one batch."""
        # this mask is for padding vs. real residues [1 0 1 1 0 ...]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [B, L] boolean
        batch["mask"] = mask

        # Split residues into context / target BEFORE encoding.
        # Partial-coordinate modes are overlapping: every valid
        # residue is both context (contributes its partial-atom latent) and target
        # (its full-atom latent is predicted) -- block_mask/masked_pct don't apply.
        # `+block_mask=true` at the CLI switches to block masking.
        # `+masked_pct=0.7` overrides random_residue_mask's default 0.5 (ignored for block masking).
        masking_mode = self.cfg.get("masking_mode", None)
        if masking_mode in {
            "chemmask_partial_coordinates",
            "chemmask_partial_coords_and_identity",
        }:
            context_mask = target_mask = mask.bool()
        elif masking_mode == "spatial_3d_block":
            context_mask, target_mask = spatial_3d_block_mask(batch)
        elif self.cfg.get("block_mask", False):
            context_mask, target_mask = block_residue_mask(batch)
        else:
            masked_pct = self.cfg.get("masked_pct", 0.5)
            context_mask, target_mask = random_residue_mask(batch, masked_pct=masked_pct)

        # Online encoder sees only the n-k context residues.
        # `+masking_mode=chemmask_aa_only` switches context residues from "real
        # coords, zeroed for target only" to "AA identity kept, all geometry hidden
        # via coord_mask". Partial-coordinate modes hide
        # `+atom_mask_pct=0.7` of EVERY residue's own atoms, identity always kept.
        # Default (flag absent) is byte-identical to before.
        if masking_mode is None or masking_mode == "spatial_3d_block":
            masked_batch = mask_batch_for_encoder(batch, context_mask)
        elif masking_mode == "chemmask_aa_only":
            masked_batch = mask_batch_for_encoder_chemmask_aa_only(batch, context_mask)
        elif masking_mode == "chemmask_partial_coordinates":
            atom_mask_pct = self.cfg.get("atom_mask_pct", 0.7)
            masked_batch = mask_batch_for_encoder_chemmask_partial_coordinates(batch, mask_pct=atom_mask_pct)
        elif masking_mode == "chemmask_partial_coords_and_identity":
            masked_batch = mask_batch_for_encoder_chemmask_partial_coords_and_identity(
                batch,
                atom_mask_pct=self.cfg.get("atom_mask_pct", 0.7),
                identity_mask_pct=self.cfg.get("identity_mask_pct", 0.5),
            )
        else:
            raise ValueError(f"Unknown masking_mode: {masking_mode!r}")
        online_mean = self._encode_mean(self.online_encoder, masked_batch)  # [B, L, latent_z_dim], only context positions carry signal

        # Representation collape monitoring
        # Std of the online latents over real context residues, per latent dim,
        # averaged over dims. If the representation is collapsing, this drops to 0 
        latent_std = online_mean[context_mask.bool()].std(dim=0).mean()

        # Full latent stats (max/min/mean/median/std + histogram)
        self.log_tensor_statistics(
            bs=online_mean.shape[0],
            v=online_mean,
            log_prefix=f"{stage}_stats_latent_mean_LS",
            mask=context_mask.bool(),
            histogram_every_n=5000,
        )

        # PCA scatter of the online latents
        self.log_pca(
            v=online_mean,
            mask=context_mask.bool(),
            log_prefix=f"{stage}_stats_latent_z_PCA",
            every_n=5000,
        )

        # Pack the (context-only) online latents into dense context/target tensors
        masked_latents = apply_mask_to_latents(online_mean, context_mask, target_mask)

        action_z = None
        mutation_positions = None
        ema_batch = batch
        if self.use_action_conditioning:
            mutation_positions, mutated_aa = sample_mutation(
                mask.bool(), batch["residue_type"]
            )  # [B], [B]
            ema_batch = apply_mutation(batch, mutation_positions, mutated_aa)

            action_z = self.mutation_embedding(mutated_aa) + self.predictor.pos_emb(
                mutation_positions
            )  # [B, hidden_dim]
            action_z = action_z.unsqueeze(1)  # [B, 1, hidden_dim], broadcasts over target queries

        # The context keeps the dataloader's global rotation.  The target gets
        # an additional independent uniform rotation and the predictor is not
        # told either frame.  Default false preserves all historical runs.
        if self.jepa_independent_target_rotation:
            ema_batch = independently_rotate_target_batch(ema_batch)

        # Context/target overlap under partial-coordinate modes (every residue is
        # both) -- self_exclude blocks query i from attending to its own context
        # entry, forcing it to reconstruct from neighbours instead of copying itself.
        self_exclude = (
            masking_mode in {
                "chemmask_partial_coordinates",
                "chemmask_partial_coords_and_identity",
            }
            and self.cfg.get("partial_coordinate_self_exclude", True)
        )

        if masking_mode == "chemmask_partial_coords_and_identity":
            identity_hidden = (
                mask.bool() & ~masked_batch["mask_dict"]["residue_type"].bool()
            )
            hidden_per_protein = identity_hidden.sum(dim=-1).float()
            valid_per_protein = mask.sum(dim=-1).clamp_min(1).float()
            hidden_fraction = hidden_per_protein / valid_per_protein
            self.log(f"{stage}_identity_mask/fraction_mean", hidden_fraction.mean(), batch_size=mask.shape[0])
            self.log(f"{stage}_identity_mask/fraction_min", hidden_fraction.min(), batch_size=mask.shape[0])
            self.log(f"{stage}_identity_mask/fraction_max", hidden_fraction.max(), batch_size=mask.shape[0])
            self.log(f"{stage}_identity_mask/n_hidden", hidden_per_protein.mean(), batch_size=mask.shape[0])
            self.log(
                f"{stage}_identity_mask/n_shown",
                (valid_per_protein - hidden_per_protein).mean(),
                batch_size=mask.shape[0],
            )

        # Predict target latents from context latents (+ positions [+ action_z])
        predicted_latent = self.predictor(
            masked_latents["context_latents"],                       # [B, L_ctx, latent_z_dim]
            masked_latents["context_indices"],                       # [B, L_ctx]
            masked_latents["target_indices"],                        # [B, L_tgt]
            key_padding_mask=masked_latents["context_padding"], # [B, L_ctx]
            target_padding_mask=masked_latents["target_padding"], # [B, L_tgt]
            action_z=action_z,
            action_position=mutation_positions,
            self_exclude=self_exclude,
        )  # [B, L_tgt, latent_z_dim]

        # EMA target encoder mean latents.
        with torch.no_grad():
            # Encoder sees all n residues
            ema_mean = self._encode_mean(self.ema_encoder, ema_batch)  # [B, L, latent_z_dim]
        # split the encoded latents on context / target masks
        ema_masked_latents = apply_mask_to_latents(ema_mean, context_mask, target_mask)
        # Select only k target latents from n encoded
        ema_target = ema_masked_latents["target_latents"]  # regression target; [B, L_tgt, latent_z_dim]
        # target latents is already padded (diff. proteins have different target counts)

        # Smooth-L1 over real target residues only (drop padding)
        target_padding = masked_latents["target_padding"]  # [B, L_tgt], True = padding
        real_target_residues = torch.logical_not(target_padding)

        predicted_for_loss = predicted_latent[real_target_residues]
        ema_target_for_loss = ema_target[real_target_residues]
        if self.baseline_run_layernorm:
            # I-JEPA normalizes teacher features per token, then asks the predictor
            # to match them in the original latent scale.
            ema_target_for_loss = F.layer_norm(
                ema_target_for_loss, (ema_target_for_loss.shape[-1],)
            )
        else:
            # Existing experiment behavior, kept unchanged unless explicitly opted in.
            predicted_for_loss = F.normalize(predicted_for_loss, dim=-1)
            ema_target_for_loss = F.normalize(ema_target_for_loss, dim=-1)

        # Smooth-L1 latent prediction loss.
        jepa_loss = F.smooth_l1_loss(predicted_for_loss, ema_target_for_loss)
        loss = jepa_loss

        sigreg_loss = None
        weighted_sigreg_loss = None
        context_effective_rank = None
        target_effective_rank = None
        predictor_effective_rank = None
        suggested_sigreg_equal_weight = None
        if self.jepa_sigreg:
            # Only trainable paths are regularized.  The EMA target is measured
            # but not included in SIGReg: it is stop-gradient, so including its
            # statistic would add a large constant without changing gradients.
            context_flat = online_mean[context_mask.bool()]
            predictor_flat = predicted_latent[real_target_residues]
            target_flat = ema_target[real_target_residues]
            sigreg_loss = self.sigreg(context_flat.unsqueeze(0)) + self.sigreg(
                predictor_flat.unsqueeze(0)
            )
            weighted_sigreg_loss = self.jepa_sigreg_weight * sigreg_loss
            loss = loss + weighted_sigreg_loss

            with torch.no_grad():
                context_effective_rank = covariance_effective_rank(context_flat)
                target_effective_rank = covariance_effective_rank(target_flat)
                predictor_effective_rank = covariance_effective_rank(predictor_flat)
                # Weight making the two scalar loss contributions equal on this
                # batch.  Calibration uses a robust median over many batches and
                # then takes a conservative fraction of this value.
                suggested_sigreg_equal_weight = (
                    jepa_loss.detach() / sigreg_loss.detach().clamp_min(1e-12)
                )

        action_sensitivity_reg = None
        z_sensitivity = None
        if self.use_action_conditioning:
            z2_aa = uniform_sampling(mutated_aa)  # [B], != z1's mutated AA
            action_z2 = self.mutation_embedding(z2_aa) + self.predictor.pos_emb(
                mutation_positions
            )  # [B, hidden_dim]
            action_z2 = action_z2.unsqueeze(1)  # [B, 1, hidden_dim]

            predicted_latent_z2 = self.predictor(
                masked_latents["context_latents"],
                masked_latents["context_indices"],
                masked_latents["target_indices"],
                key_padding_mask=masked_latents["context_padding"],
                target_padding_mask=masked_latents["target_padding"],
                action_z=action_z2,
                action_position=mutation_positions,
                self_exclude=self_exclude,
            )  # [B, L_tgt, latent_z_dim]

            # L2-normalize before the distance so neither side can inflate its raw
            # output scale to game z_sensitivity (caps numerator/denominator to [0, 2] each).
            predicted_norm = F.normalize(predicted_latent[real_target_residues], dim=-1)
            predicted_norm_z2 = F.normalize(predicted_latent_z2[real_target_residues], dim=-1)
            numerator = (predicted_norm - predicted_norm_z2).norm(dim=-1).mean()
            action_norm_z1 = F.normalize(action_z, dim=-1)
            action_norm_z2 = F.normalize(action_z2, dim=-1)
            denominator = (action_norm_z1 - action_norm_z2).norm(dim=-1).mean() + 1e-8
            z_sensitivity = numerator / denominator
            action_sensitivity_reg = -z_sensitivity

            lambda_action_sensitivity = self._action_sensitivity_lambda()
            loss = loss + lambda_action_sensitivity * action_sensitivity_reg

            debug_log_kwargs = dict(
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=predicted_latent.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(f"{stage}_debug/action_sensitivity_lambda", lambda_action_sensitivity, **debug_log_kwargs)
            self.log(f"{stage}_debug/action_reg_numerator", numerator, **debug_log_kwargs)
            self.log(f"{stage}_debug/action_reg_denominator", denominator, **debug_log_kwargs)
            self.log(f"{stage}_debug/action_z_norm", action_z.norm(dim=-1).mean(), **debug_log_kwargs)
            self.log(f"{stage}_debug/action_z2_norm", action_z2.norm(dim=-1).mean(), **debug_log_kwargs)

        # Predictor-output vs. target-encoder stats and their cosine similarity
        self.log_tensor_statistics(
            bs=predicted_latent.shape[0],
            v=predicted_latent,
            log_prefix=f"{stage}_stats_latent_predicted_LS",
            mask=real_target_residues,
            histogram_every_n=5000,
        )
        self.log_tensor_statistics(
            bs=ema_target.shape[0],
            v=ema_target,
            log_prefix=f"{stage}_stats_latent_target_LS",
            mask=real_target_residues,
            histogram_every_n=5000,
        )
        cos_sim = F.cosine_similarity(
            predicted_latent[real_target_residues], ema_target[real_target_residues], dim=-1
        ).mean()
        self.log(
            f"{stage}_predicted_target_cosine_sim",
            cos_sim,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=predicted_latent.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )

        # Collapse-monitoring stds for the three components (online encoder, target encoder, predictor)
        online_latent_std = online_mean[context_mask.bool()].std()
        target_latent_std = ema_mean[mask.bool()].std()
        pred_latent_std = predicted_latent[real_target_residues].std()

        return {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "sigreg_loss": sigreg_loss,
            "weighted_sigreg_loss": weighted_sigreg_loss,
            "context_effective_rank": context_effective_rank,
            "target_effective_rank": target_effective_rank,
            "predictor_effective_rank": predictor_effective_rank,
            "suggested_sigreg_equal_weight": suggested_sigreg_equal_weight,
            "latent_std": latent_std,
            "online_latent_std": online_latent_std,
            "target_latent_std": target_latent_std,
            "pred_latent_std": pred_latent_std,
            "action_sensitivity_reg": action_sensitivity_reg,
            "z_sensitivity": z_sensitivity,
        }

# compute loss -> log -> return loss
    def training_step(self, batch: Dict, batch_idx: int):
        bs = batch["mask_dict"]["coords"].shape[0]
        out = self.compute_loss(batch, stage="train")
        # Lightning logs this as train_loss_step and train_loss_epoch
        self.log(
            "train_loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "train_latent_std",
            out["latent_std"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if out["sigreg_loss"] is not None:
            sigreg_log_values = {
                "train_jepa_loss": out["jepa_loss"],
                "train_sigreg_loss_raw": out["sigreg_loss"],
                "train_sigreg_loss_weighted": out["weighted_sigreg_loss"],
                "train_effective_rank/context": out["context_effective_rank"],
                "train_effective_rank/target": out["target_effective_rank"],
                "train_effective_rank/predictor": out["predictor_effective_rank"],
                "train_sigreg_equal_loss_weight": out["suggested_sigreg_equal_weight"],
            }
            for name, value in sigreg_log_values.items():
                self.log(
                    name,
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            weighted_ratio = out["weighted_sigreg_loss"] / out[
                "jepa_loss"
            ].detach().clamp_min(1e-12)
            self.log(
                "train_sigreg_weighted_to_jepa_ratio",
                weighted_ratio,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            # A weight-zero calibration run leaves the training objective and
            # its gradients unchanged while collecting the coefficient that
            # would make raw SIGReg and JEPA scalar losses equal.  (Sampling the
            # diagnostic projections still consumes RNG, so this is not a
            # bitwise reproduction of a baseline trajectory.)  Print robust
            # summaries so calibration does not depend on W&B availability.
            equal_weight = float(out["suggested_sigreg_equal_weight"].detach().cpu())
            if torch.isfinite(torch.tensor(equal_weight)) and equal_weight > 0:
                self._sigreg_equal_weight_samples.append(equal_weight)
            n_calibration = len(self._sigreg_equal_weight_samples)
            if n_calibration in {50, 100, 200, 500} and self.global_rank == 0:
                values = torch.tensor(self._sigreg_equal_weight_samples)
                q25, median, q75 = torch.quantile(
                    values, torch.tensor([0.25, 0.5, 0.75])
                ).tolist()
                print(
                    "SIGREG_CALIBRATION "
                    f"n={n_calibration} equal_loss_weight_median={median:.8g} "
                    f"q25={q25:.8g} q75={q75:.8g} "
                    f"recommended_10pct={0.1 * median:.8g} "
                    f"recommended_25pct={0.25 * median:.8g}",
                    flush=True,
                )

        self.log(
            "online_latent_std",
            out["online_latent_std"],
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "target_latent_std",
            out["target_latent_std"],
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "pred_latent_std",
            out["pred_latent_std"],
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        if out["action_sensitivity_reg"] is not None:
            self.log(
                "train_jepa_loss",
                out["jepa_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                "train_action_sensitivity_reg",
                out["action_sensitivity_reg"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                "train_z_sensitivity",
                out["z_sensitivity"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        self.update_n_log_nsamples_processed(bs)
        self.log_nparams()
        return out["loss"]

    def validation_step(self, batch: Dict, batch_idx: int):
        bs = batch["mask_dict"]["coords"].shape[0]
        out = self.compute_loss(batch, stage="val")
        self.log(
            "val_loss",
            out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val_latent_std",
            out["latent_std"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        if out["sigreg_loss"] is not None:
            sigreg_log_values = {
                "val_jepa_loss": out["jepa_loss"],
                "val_sigreg_loss_raw": out["sigreg_loss"],
                "val_sigreg_loss_weighted": out["weighted_sigreg_loss"],
                "val_effective_rank/context": out["context_effective_rank"],
                "val_effective_rank/target": out["target_effective_rank"],
                "val_effective_rank/predictor": out["predictor_effective_rank"],
            }
            for name, value in sigreg_log_values.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
        if out["action_sensitivity_reg"] is not None:
            self.log(
                "val_jepa_loss",
                out["jepa_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                "val_action_sensitivity_reg",
                out["action_sensitivity_reg"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                "val_z_sensitivity",
                out["z_sensitivity"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        return out["loss"]

    def log_histogram(self, id_log: str, v: torch.Tensor):
        """Logs histogram, v must be a flat tensor. Ported from the autoencoder."""
        assert v.ndim == 1, f"Tensor v has shape {v.shape}, cannot log histogram"
        try:
            self.logger.experiment.log({id_log: wandb.Histogram(v.cpu().detach().numpy())})
        except Exception:
            pass

    def log_tensor_statistics(
        self,
        bs: int,
        v: torch.Tensor,
        log_prefix: str,
        mask: torch.Tensor,
        do_stats: bool = True,
        do_hist: bool = True,
        suffix: str = "",
        histogram_every_n: int = 5000,
        stats_to_log: List[str] = ["max", "min", "mean", "median", "std"],
    ) -> None:
        """Logs tensor statistics: mean, median, max, min, std. Ported from the
        autoencoder's log_tensor_statistics."""
        vals = v.clone()[mask]  # [n_real_residues, latent_z_dim]
        vals = torch.flatten(vals)  # 1D vector
        if do_stats:
            funs = {
                "max": torch.max,
                "min": torch.min,
                "mean": torch.mean,
                "median": torch.median,
                "std": torch.std,
            }
            for k in stats_to_log:
                self.log(
                    f"{log_prefix}/{k}{suffix}",
                    funs[k](vals),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        if self.global_step % histogram_every_n == 0 and do_hist:
            self.log_histogram(id_log=f"{log_prefix}/histogram{suffix}", v=vals)

    def log_nparams(self):
        """Logs param counts for encoder/predictor, ported from the autoencoder's
        log_nparams (there it's encoder/decoder; JEPA has no decoder, so encoder/predictor)."""
        self.log(
            "scaling/nparams_enc",
            self.nparams_enc * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "scaling/nparams_predictor",
            self.nparams_predictor * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_nsamples_processed(self, b: int):
        """Ported from the autoencoder's update_n_log_nsamples_processed."""
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def log_pca(
        self,
        v: torch.Tensor,
        mask: torch.Tensor,
        log_prefix: str,
        every_n: int = 5000,
    ) -> None:
        """Logs PCA scatter plots of a latent tensor to wandb, ported from the
        autoencoder's log_pca (proteinfoundation/partial_autoencoder/autoencoder.py).

        v:    [B, L, latent_z_dim]  latents (e.g. online encoder mean)
        mask: [B, L] bool           True on real (non-padding) residues
        """

        def _log_scatter(vals_x, vals_y, xlabel, ylabel, log_id):
            fig, ax = plt.subplots()
            ax.scatter(vals_x, vals_y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{xlabel} - {ylabel} - Step: {self.trainer.global_step}")
            try:
                self.logger.experiment.log({log_id: wandb.Image(fig)})
            except Exception:
                pass
            plt.close("all")

        if self.global_step % every_n != 0:
            return

        n_components = min(4, self.latent_dim)
        vals = v.clone()[mask].cpu().detach().float().numpy()  # [n_real_residues, latent_z_dim]
        if vals.shape[0] < n_components:
            return  # not enough points to fit PCA this batch
        vals_pca = PCA(n_components=n_components).fit_transform(vals)  # [n_real_residues, n_components]

        for i in range(n_components):
            for j in range(n_components):
                if j <= i:
                    continue
                _log_scatter(
                    vals_x=vals_pca[:, i],
                    vals_y=vals_pca[:, j],
                    xlabel=f"PC {i}",
                    ylabel=f"PC {j}",
                    log_id=f"{log_prefix}/{i}_{j}",
                )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update of the target encoder after each training step."""
        if self.baseline_run_ema:
            # With gradient accumulation this hook runs on every batch, while
            # global_step advances only after an optimizer step. Update once per
            # optimizer step and interpolate from 0.996 at step 1 to 1.0 at max_steps.
            if self.global_step == self._last_ema_update_step:
                return
            self._last_ema_update_step = self.global_step
            progress = min(
                max(self.global_step - 1, 0) / (self.ema_schedule_steps - 1), 1.0
            )
            ema_decay = self.ema_decay + progress * (1.0 - self.ema_decay)
        else:
            ema_decay = self.ema_decay
        with torch.no_grad():
            for online_p, ema_p in zip(
                self.online_encoder.parameters(), self.ema_encoder.parameters()
            ):
                ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * online_p.data

    def configure_optimizers(self):
        # Only parameters with requires_grad=True are optimized, which excludes the
        # EMA target encoder (online encoder + predictor are trained).
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.cfg.opt.lr,
            amsgrad=True,
            weight_decay=1e-2,
        )
        # Cuts LR in half after 4 val checks with no improvement
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, threshold=1e-2, threshold_mode="rel",
            min_lr=1e-6,
        )
        if self._pending_lr_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(self._pending_lr_scheduler_state)
            self._pending_lr_scheduler_state = None
        return optimizer
