"""LeJEPA protein model, with optional action (mutation) conditioning.

Context encoder receives n-k residues, and outputs n-k residue latents.
Predictor receives n-k residues, and predicts the k target residues.
Target pass uses the same encoder, sees n residues, and outputs k target latents.
No separate target network, no EMA, no stop-gradient.

Two components:
  * online encoder  - the existing VAE encoder architecture, trained by backprop. We
                      use only its ``mean`` output as the deterministic latent (no
                      sampling, no KL). Used for both the context pass and the
                      target pass. 
  * predictor       - cross-attention network mapping context latents -> target latents.

The loss is (1-lamb)*jepa_loss + lamb*sigreg_loss:
  * jepa_loss   - plain MSE between the predictor's output and the encoder's
                  target latents, over target residues only.
  * sigreg_loss - SIGReg (Sketched Isotropic Gaussian Regularization), pushing
                  the distribution of the encoder's output embeddings (context +
                  target combined) toward an isotropic Gaussian. This is what
                  prevents collapse, replacing the EMA/stop-gradient heuristics
                  I-JEPA-style architectures rely on.
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
    mask_partial_atom_coordinates,
    mask_partial_sidechain_coordinates,
    random_residue_mask,
    sidechain_atom_mask,
    sidechain_chemical_context_target_masks,
    spatial_3d_block_mask,
)
from proteinfoundation.jepa.predictor import JEPAPredictor
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.coors_utils import sample_uniform_rotation


def independently_rotate_target_batch(batch: Dict) -> Dict:
    """Return a shallow batch copy with independently rotated target coordinates."""
    coords_nm = batch["coords_nm"]
    rotations = sample_uniform_rotation(
        shape=(coords_nm.shape[0],),
        dtype=coords_nm.dtype,
        device=coords_nm.device,
    )
    rotated = dict(batch)
    rotated["coords_nm"] = torch.einsum("bnac,bcd->bnad", coords_nm, rotations)
    return rotated


def covariance_effective_rank(values: torch.Tensor) -> torch.Tensor:
    """Effective rank of the centered embedding covariance, in ``[0, D]``.

    A healthy representation uses several latent directions; rank approaching 1
    indicates dimensional collapse, while 0 means the batch has no variance.
    """
    if values.ndim != 2:
        raise ValueError("effective rank expects an [N, D] embedding matrix")
    if values.shape[0] < 2:
        return values.new_zeros((), dtype=torch.float32)

    centered = values.detach().float() - values.detach().float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    total_variance = variance.sum()
    if total_variance <= 0:
        return total_variance
    probabilities = variance / total_variance
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


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

#Note: we're only changing the residue_type in the batch, not the coordinates. The coordinates are still the wildtype coordinates. 
#We want the model to learn the latent representation of the mutated residue, while being aware of the wildtype coordinates.
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


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization: projects embeddings onto random
    1D directions and tests (via the Epps-Pulley characteristic-function test on a
    fixed quadrature grid) whether each projection looks standard-normal. By the
    Cramer-Wold theorem, matching this along every 1D projection implies the full
    embedding distribution is isotropic Gaussian. num_proj=32 (not the reference's
    256): latent_dim=8 here, so more random directions than the ambient dimension
    add no additional coverage.
    """

    def __init__(self, knots=17, num_proj=32):
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

    def forward(self, proj):
        # proj: (T, B, D), called with T=1 -- no natural "views" axis here
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class JEPAModel(L.LightningModule):
    def __init__(self, cfg, lamb: float = 0.02, use_action_conditioning: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.lamb = lamb
        self.latent_dim = self.cfg.nn_ae["latent_z_dim"]  # 8
        self.baseline_run_layernorm = self.cfg.get("baseline_run_layernorm", False)
        self.baseline_run_ema = self.cfg.get("baseline_run_ema", False)
        self.baseline_run_predictor = self.cfg.get("baseline_run_predictor", False)
        self.lejepa_relative_positions = self.cfg.get(
            "lejepa_relative_positions", False
        )
        self.jepa_independent_target_rotation = bool(
            self.cfg.get("jepa_independent_target_rotation", False)
        )
        self.partial_coordinate_self_exclude = self.cfg.get(
            "partial_coordinate_self_exclude", False
        )
        self.partial_coordinate_diagnostics = self.cfg.get(
            "partial_coordinate_diagnostics", True
        )
        self.ema_decay = 0.996
        self._last_ema_update_step = -1

        if self.baseline_run_ema:
            self.ema_schedule_steps = self.cfg.opt.get("max_steps", None)
            if not isinstance(self.ema_schedule_steps, int) or self.ema_schedule_steps < 2:
                raise ValueError("baseline_run_ema=true requires opt.max_steps >= 2")
        else:
            self.ema_schedule_steps = None

        self.use_action_conditioning = use_action_conditioning
        if self.use_action_conditioning:
            self.mutation_embedding = nn.Embedding(20, 128)  # matches predictor hidden_dim
        if self.lejepa_relative_positions and self.baseline_run_predictor:
            raise ValueError(
                "lejepa_relative_positions=true is defined for the old "
                "cross-attention predictor; set baseline_run_predictor=false"
            )
        if self.lejepa_relative_positions and self.use_action_conditioning:
            raise ValueError(
                "lejepa_relative_positions is not yet defined for mutation action "
                "conditioning, which currently uses predictor.pos_emb"
            )

        # Used for both context and target passes -- no separate target network.
        self.online_encoder = EncoderTransformer(**self.cfg.nn_ae)

        if self.baseline_run_ema:
            self.ema_encoder = copy.deepcopy(self.online_encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad = False

        self.sigreg = SIGReg(num_proj=32)

        # Predictor: context latents + positions -> predicted target latents.
        # max_len must be >= the dataset's max crop length (256).
        predictor_cfg = self.cfg.get("predictor", {}) or {}
        max_len = predictor_cfg.get("max_len", 256)
        relative_position_max_distance = predictor_cfg.get(
            "relative_position_max_distance", 128
        )
        self.predictor = JEPAPredictor(
            latent_z_dim=self.latent_dim,
            max_len=max_len,
            predictor_type=("mask_token_transformer" if self.baseline_run_predictor else "cross_attention"),
            use_relative_positions=self.lejepa_relative_positions,
            relative_position_max_distance=relative_position_max_distance,
        )

        # used for below (log_isotropy, log_mutation_sensitivity). 
        self._diag_step = {"train": 0, "val": 0}

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

    def on_save_checkpoint(self, checkpoint):
        checkpoint["nsamples_processed"] = self.nsamples_processed
        checkpoint["diag_step"] = self._diag_step
        checkpoint["lr_scheduler_state"] = self.lr_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        try:
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except KeyError:
            self.nsamples_processed = 0
        self._diag_step = checkpoint.get("diag_step", {"train": 0, "val": 0})
        if "lr_scheduler_state" in checkpoint:
            self._pending_lr_scheduler_state = checkpoint["lr_scheduler_state"]
        elif checkpoint.get("lr_schedulers"):
            # checkpoints saved before the switch to manual stepping
            # kept scheduler state under Lightning's own "lr_schedulers" key instead.
            self._pending_lr_scheduler_state = checkpoint["lr_schedulers"][0]

    def on_validation_end(self):
        # Manually stepped, not registered via configure_optimizers' scheduler dict:
        if self.trainer.sanity_checking:
            return
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        self.lr_scheduler.step(val_loss)

    def _encode_mean(self, encoder, batch: Dict) -> torch.Tensor:
        """Encoder that only returns the deterministic mean latent (no var)."""
        return encoder(batch)["mean"]  # [B, L, latent_z_dim]

    def compute_loss(self, batch: Dict, stage: str = "train") -> Dict[str, torch.Tensor]:
        """Pure JEPA latent-prediction loss for one batch."""
        self._diag_step[stage] = self._diag_step.get(stage, 0) + 1
        diag_step = self._diag_step[stage]

        # this mask is for padding vs. real residues [1 0 1 1 0 ...]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [B, L] boolean
        batch["mask"] = mask

        # Split residues into context / target BEFORE encoding.
        # `+block_mask=true` at the CLI switches to block masking.
        # `+masked_pct=0.7` overrides random_residue_mask's default 0.5 (ignored for block masking).
        masking_mode = self.cfg.get("masking_mode", None)
        partial_observation_modes = {
            "chemmask_partial_coordinates",
            "chemmask_sidechain_partial",
            "chemmask_no_aa_sidechain_partial",
        }
        partial_observation_mode = masking_mode in partial_observation_modes
        if masking_mode == "chemmask_partial_coordinates":
            # All valid residues contribute a partially observed context latent and
            # are simultaneously targets for the full-structure encoder latent.
            context_mask = mask.bool()
            target_mask = mask.bool()
        elif masking_mode == "chemmask_sidechain_partial":
            context_mask, target_mask = sidechain_chemical_context_target_masks(
                batch, mask_aa_identity=False
            )
        elif masking_mode == "chemmask_no_aa_sidechain_partial":
            context_mask, target_mask = sidechain_chemical_context_target_masks(
                batch, mask_aa_identity=True
            )
        elif masking_mode == "spatial_3d_block":
            context_mask, target_mask = spatial_3d_block_mask(batch)
        elif masking_mode is not None:
            raise ValueError(f"Unknown masking_mode: {masking_mode!r}")
        elif self.cfg.get("block_mask", False):
            context_mask, target_mask = block_residue_mask(batch)
        else:
            masked_pct = self.cfg.get("masked_pct", 0.5)
            context_mask, target_mask = random_residue_mask(batch, masked_pct=masked_pct)

        if partial_observation_mode:
            if masking_mode == "chemmask_partial_coordinates":
                masked_batch = mask_partial_atom_coordinates(
                    batch, mask_pct=self.cfg.get("atom_mask_pct", 0.5)
                )
                eligible_atoms = batch["coord_mask"].bool()
                aa_identity_hidden = False
            else:
                aa_identity_hidden = masking_mode == "chemmask_no_aa_sidechain_partial"
                masked_batch = mask_partial_sidechain_coordinates(
                    batch,
                    mask_pct=self.cfg.get("sidechain_mask_pct", 0.5),
                    mask_aa_identity=aa_identity_hidden,
                )
                eligible_atoms = sidechain_atom_mask(batch["coord_mask"])
            original_atoms = batch["coord_mask"].bool()
            visible_atoms = masked_batch["coord_mask"].bool()
            hidden_atoms = original_atoms & ~visible_atoms
            valid_atom_count = original_atoms.sum().clamp_min(1)
            eligible_atom_count = eligible_atoms.sum().clamp_min(1)
            valid_residue_count = mask.bool().sum().clamp_min(1)
            self.log(
                f"{stage}_atom_mask/hidden_fraction",
                hidden_atoms.sum().float() / valid_atom_count,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=mask.shape[0],
                sync_dist=True,
            )
            self.log(
                f"{stage}_atom_mask/hidden_fraction_of_eligible",
                hidden_atoms.sum().float() / eligible_atom_count,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=mask.shape[0],
                sync_dist=True,
            )
            self.log(
                f"{stage}_atom_mask/visible_atoms_per_residue",
                visible_atoms.sum().float() / valid_residue_count,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=mask.shape[0],
                sync_dist=True,
            )
            self.log(
                f"{stage}_atom_mask/aa_identity_hidden_fraction",
                torch.tensor(
                    float(aa_identity_hidden), device=mask.device, dtype=torch.float32
                ),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=mask.shape[0],
                sync_dist=True,
            )
            self.log(
                f"{stage}_atom_mask/hidden_ca_fraction",
                hidden_atoms[..., 1].sum().float() / valid_residue_count,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=mask.shape[0],
                sync_dist=True,
            )
        else:
            # Standard JEPA: the online encoder sees only n-k context residues.
            masked_batch = mask_batch_for_encoder(batch, context_mask)
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
        target_batch = batch
        if self.use_action_conditioning:
            mutation_positions, mutated_aa = sample_mutation(
                mask.bool(), batch["residue_type"]
            )  # [B], [B]
            target_batch = apply_mutation(batch, mutation_positions, mutated_aa)

            action_z = self.mutation_embedding(mutated_aa) + self.predictor.pos_emb(
                mutation_positions
            )  # [B, hidden_dim]
            action_z = action_z.unsqueeze(1)  # [B, 1, hidden_dim], broadcasts over target queries

        # The context remains in the dataloader frame; only the full target view
        # receives an additional independent uniform rotation.
        if self.jepa_independent_target_rotation:
            target_batch = independently_rotate_target_batch(target_batch)

        # Predict target latents from context latents (+ positions)
        predicted_latent = self.predictor(
            masked_latents["context_latents"],                       # [B, L_ctx, latent_z_dim]
            masked_latents["context_indices"],                       # [B, L_ctx]
            masked_latents["target_indices"],                        # [B, L_tgt]
            key_padding_mask=masked_latents["context_padding"], # [B, L_ctx]
            target_padding_mask=masked_latents["target_padding"], # [B, L_tgt]
            action_z=action_z,
            action_position=mutation_positions,
            context_access=(
                "exclude_self"
                if partial_observation_mode and self.partial_coordinate_self_exclude
                else "all"
            ),
        )  # [B, L_tgt, latent_z_dim]

        # Target pass: preserve the original online-encoder path by default;
        # EMA is opt-in and changes only the target encoder.
        if self.baseline_run_ema:
            with torch.no_grad():
                target_mean = self._encode_mean(self.ema_encoder, target_batch)
        else:
            target_mean = self._encode_mean(self.online_encoder, target_batch)
        # split the encoded latents on context / target masks
        target_masked_latents = apply_mask_to_latents(target_mean, context_mask, target_mask)
        # Select only k target latents from n encoded
        target_latent = target_masked_latents["target_latents"]  # regression target; [B, L_tgt, latent_z_dim]
        # target latents is already padded (diff. proteins have different target counts)

        # Real target residues only (drop padding)
        target_padding = masked_latents["target_padding"]  # [B, L_tgt], True = padding
        real_target_residues = torch.logical_not(target_padding)

        predicted_latent_flat = predicted_latent[real_target_residues]
        target_latent_flat = target_latent[real_target_residues]

        if partial_observation_mode:
            # Gather the context latent at each target residue. This remains
            # aligned when glycine is context-only in side-chain-only masking.
            context_latent_flat = online_mean[target_mask]
            if context_latent_flat.shape != target_latent_flat.shape:
                raise RuntimeError("partial-coordinate context/target latents are not aligned")
            with torch.no_grad():
                context_target_cosine = F.cosine_similarity(
                    context_latent_flat.float(), target_latent_flat.float(), dim=-1
                )
            for name, value in {
                "context_target_cosine_mean": context_target_cosine.mean(),
                "context_target_cosine_max": context_target_cosine.max(),
                "context_target_cosine_fraction_gt_0_95": (
                    context_target_cosine > 0.95
                ).float().mean(),
            }.items():
                self.log(
                    f"{stage}_partial_context/{name}",
                    value,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        if self.use_action_conditioning:
            self.log_mutation_sensitivity(
                masked_latents,
                predicted_latent_flat,
                real_target_residues,
                log_prefix=f"{stage}_mutation_sensitivity",
                diag_step=diag_step,
                every_n=1000,
            )

        with torch.no_grad():
            target_mean_baseline = target_latent_flat.mean(dim=0, keepdim=True)
            baseline_loss = (target_mean_baseline - target_latent_flat).square().mean()

        # Plain MSE, no normalization -- matches the LeJEPA reference scripts exactly.
        if self.baseline_run_layernorm:
            target_latent_flat = F.layer_norm(target_latent_flat, (target_latent_flat.shape[-1],))
        jepa_loss = (predicted_latent_flat - target_latent_flat).square().mean()
        pred_vs_baseline_gap = baseline_loss - jepa_loss  # >0 means predictor beats trivial mean-prediction

        # Counterfactual validation probes for overlapping partial-coordinate
        # context/targets. These do not change training. They quantify whether the
        # predictor uses its own partial residue, its neighbours, or merely copies
        # the context encoder latent. For the joint mask-token predictor, exclusion
        # is direct (later layers can still carry information indirectly).
        if (
            stage == "val"
            and partial_observation_mode
            and self.partial_coordinate_diagnostics
        ):
            predictor_args = (
                masked_latents["context_latents"],
                masked_latents["context_indices"],
                masked_latents["target_indices"],
            )
            predictor_kwargs = {
                "key_padding_mask": masked_latents["context_padding"],
                "target_padding_mask": masked_latents["target_padding"],
                "action_z": action_z,
                "action_position": mutation_positions,
            }
            with torch.no_grad():
                if self.partial_coordinate_self_exclude:
                    pred_full = self.predictor(
                        *predictor_args, **predictor_kwargs, context_access="all"
                    )
                    pred_no_self = predicted_latent
                else:
                    pred_full = predicted_latent
                    pred_no_self = self.predictor(
                        *predictor_args,
                        **predictor_kwargs,
                        context_access="exclude_self",
                    )
                pred_self_only = self.predictor(
                    *predictor_args,
                    **predictor_kwargs,
                    context_access="self_only",
                )

                full_loss = (
                    pred_full[real_target_residues] - target_latent_flat
                ).square().mean()
                no_self_loss = (
                    pred_no_self[real_target_residues] - target_latent_flat
                ).square().mean()
                self_only_loss = (
                    pred_self_only[real_target_residues] - target_latent_flat
                ).square().mean()

                copy_latent = online_mean[target_mask]
                copy_loss = (copy_latent - target_latent_flat).square().mean()

            diagnostic_values = {
                "loss_full": full_loss,
                "loss_direct_no_self": no_self_loss,
                "loss_direct_self_only": self_only_loss,
                "loss_context_copy": copy_loss,
                "self_information_gain": no_self_loss - full_loss,
                "neighbour_information_gain": self_only_loss - full_loss,
                "predictor_gain_over_copy": copy_loss - full_loss,
            }
            for name, value in diagnostic_values.items():
                self.log(
                    f"val_partial_context/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=mask.shape[0],
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        # SIGReg on context, target, and predictor output separately, summed.
        context_flat = online_mean[context_mask.bool()]   # n-k residues
        target_flat = target_latent[real_target_residues]  # k residues, no overlap

        # Effective-rank monitoring complements scalar latent std: std can stay
        # nonzero even if all embeddings occupy only one direction.
        with torch.no_grad():
            effective_ranks = {
                "context": covariance_effective_rank(context_flat),
                "target": covariance_effective_rank(target_flat),
                "predictor": covariance_effective_rank(predicted_latent_flat),
            }
        for name, value in effective_ranks.items():
            self.log(
                f"{stage}_effective_rank/{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )

        context_sigreg = self.sigreg(context_flat.unsqueeze(0))
        target_sigreg = self.sigreg(target_flat.unsqueeze(0))
        predictor_sigreg = self.sigreg(predicted_latent_flat.unsqueeze(0))
        sigreg_loss = context_sigreg + target_sigreg + predictor_sigreg

        self.log_isotropy(context_flat, log_prefix=f"{stage}_isotropy_context", diag_step=diag_step, every_n=5000)
        self.log_isotropy(target_flat, log_prefix=f"{stage}_isotropy_target", diag_step=diag_step, every_n=5000)
        self.log_isotropy(predicted_latent_flat, log_prefix=f"{stage}_isotropy_predictor", diag_step=diag_step, every_n=5000)

        loss = (1 - self.lamb) * jepa_loss + self.lamb * sigreg_loss

        # Predictor-output vs. target-encoder stats and their cosine similarity
        self.log_tensor_statistics(
            bs=predicted_latent.shape[0],
            v=predicted_latent,
            log_prefix=f"{stage}_stats_latent_predicted_LS",
            mask=real_target_residues,
            histogram_every_n=5000,
        )
        self.log_tensor_statistics(
            bs=target_latent.shape[0],
            v=target_latent,
            log_prefix=f"{stage}_stats_latent_target_LS",
            mask=real_target_residues,
            histogram_every_n=5000,
        )
        cos_sim = F.cosine_similarity(predicted_latent_flat, target_latent_flat, dim=-1).mean()
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

        # Collapse-monitoring stds for the two components (online encoder, predictor)
        online_latent_std = online_mean[context_mask.bool()].std()
        target_latent_std = target_mean[mask.bool()].std()
        pred_latent_std = predicted_latent[real_target_residues].std()

        return {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "sigreg_loss": sigreg_loss,
            "baseline_loss": baseline_loss,
            "pred_vs_baseline_gap": pred_vs_baseline_gap,
            "latent_std": latent_std,
            "online_latent_std": online_latent_std,
            "target_latent_std": target_latent_std,
            "pred_latent_std": pred_latent_std,
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
            "train_sigreg_loss",
            out["sigreg_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "train_baseline_loss",
            out["baseline_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "train_pred_vs_baseline_gap",
            out["pred_vs_baseline_gap"],
            on_step=True,
            on_epoch=True,
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
            "val_sigreg_loss",
            out["sigreg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val_baseline_loss",
            out["baseline_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val_pred_vs_baseline_gap",
            out["pred_vs_baseline_gap"],
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

    def log_mutation_sensitivity(
        self,
        masked_latents: Dict[str, torch.Tensor],
        predicted_latent_flat: torch.Tensor,
        real_target_residues: torch.Tensor,
        log_prefix: str,
        diag_step: int,
        every_n: int = 1000,
    ) -> None:
        """Logs how much the predictor's output changes when action_z is dropped, i.e.
        whether the model is using the mutation signal at all (near-zero = ignoring it).

        """
        if diag_step % every_n != 0:
            return
        with torch.no_grad():
            predicted_no_action = self.predictor(
                masked_latents["context_latents"],
                masked_latents["context_indices"],
                masked_latents["target_indices"],
                key_padding_mask=masked_latents["context_padding"],
                target_padding_mask=masked_latents["target_padding"],
            )  # [B, L_tgt, latent_z_dim], action_z=None -> no-mutation baseline
            diff = (
                predicted_latent_flat - predicted_no_action[real_target_residues]
            ).norm(dim=-1).mean()
        try:
            self.logger.experiment.log({f"{log_prefix}/pred_diff_norm": diff.item()})
        except Exception:
            pass

    def log_isotropy(
        self, v: torch.Tensor, log_prefix: str, diag_step: int, every_n: int = 5000
    ) -> None:
        """Logs per-dimension std [D] and isotropy_ratio (max/min dim std) of v: [N, D].

        """
        if diag_step % every_n != 0:
            return
        dim_std = v.detach().std(dim=0)  # [D]
        ratio = dim_std.max() / dim_std.min().clamp_min(1e-8)
        payload = {f"{log_prefix}/dim_{i}_std": s for i, s in enumerate(dim_std.tolist())}
        payload[f"{log_prefix}/isotropy_ratio"] = ratio.item()
        try:
            self.logger.experiment.log(payload)
        except Exception:
            pass

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
        if not self.baseline_run_ema:
            return
        if self.global_step == self._last_ema_update_step:
            return
        self._last_ema_update_step = self.global_step
        progress = min(max(self.global_step - 1, 0) / (self.ema_schedule_steps - 1), 1.0)
        decay = self.ema_decay + progress * (1.0 - self.ema_decay)
        with torch.no_grad():
            for online_p, ema_p in zip(self.online_encoder.parameters(), self.ema_encoder.parameters()):
                ema_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)

    def configure_optimizers(self):
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
