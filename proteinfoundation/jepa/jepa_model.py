"""JEPA protein model.

This is a pure JEPA (Joint-Embedding Predictive Architecture) objective: the model
never reconstructs the input. Instead it learns by predicting, in latent space, the
representations of a set of masked (target) residues from the representations of the
remaining (context) residues.

Three components:
  * online encoder  - the existing VAE encoder architecture, trained by backprop. We
                      use only its ``mean`` output as the deterministic latent (no
                      sampling, no KL).
  * EMA target encoder - an exact copy of the online encoder, updated as an exponential
                      moving average of the online weights. It is never trained by
                      backprop (requires_grad=False) and produces the prediction targets.
  * predictor       - cross-attention network mapping context latents -> target latents.

The loss is a smooth-L1 between the predictor's output and the EMA encoder's target
latents, over target residues only. Using an EMA target encoder (rather than the online
encoder) provides stable, non-collapsing targets, which is what makes JEPA work without
a reconstruction term.
"""

import copy
from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F

from proteinfoundation.jepa.masking import apply_mask_to_latents, random_residue_mask
from proteinfoundation.jepa.predictor import JEPAPredictor
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


class JEPAModel(L.LightningModule):
    def __init__(self, cfg, ema_decay=0.999):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.ema_decay = ema_decay
        self.latent_dim = self.cfg.nn_ae["latent_z_dim"]  # 8

        # Online encoder: trained via backprop, reuses the existing encoder architecture
        self.online_encoder = EncoderTransformer(**self.cfg.nn_ae)

        # EMA target encoder: exact copy of the online encoder, same initial weights,
        # not trained via backprop (excluded from the optimizer)
        self.ema_encoder = copy.deepcopy(self.online_encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False

        # Predictor: context latents + positions -> predicted target latents.
        # max_len must be >= the dataset's max crop length (positional table size).
        predictor_cfg = self.cfg.get("predictor", {}) or {}
        max_len = predictor_cfg.get("max_len", 256)
        self.predictor = JEPAPredictor(latent_z_dim=self.latent_dim, max_len=max_len)

    def _encode_mean(self, encoder, batch: Dict) -> torch.Tensor:
        """Run an encoder and return only the deterministic mean latent (no variance /sampling)."""
        return encoder(batch)["mean"]  # [B, L, latent_z_dim]

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Pure JEPA latent-prediction loss for one batch."""
        # this mask is for padding vs. real residues [1 0 1 1 0 ...]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [B, L] boolean
        batch["mask"] = mask  

        # Online encoder mean latents for all residues
        online_mean = self._encode_mean(self.online_encoder, batch)  # [B, L, latent_z_dim]

        # Split residues into context / target and pack each set into dense tensors
        context_mask, target_mask = random_residue_mask(batch)  # [B, L], [B, L]
        masked_latents = apply_mask_to_latents(online_mean, context_mask, target_mask)

        # Predict target latents from context latents (+ positions)
        predicted_latent = self.predictor(
            masked_latents["context_latents"],                       # [B, L_ctx, latent_z_dim]
            masked_latents["context_indices"],                       # [B, L_ctx]
            masked_latents["target_indices"],                        # [B, L_tgt]
            key_padding_mask=masked_latents["context_padding_mask"], # [B, L_ctx]
        )  # [B, L_tgt, latent_z_dim]

        # EMA target encoder mean latents (no grad), gathered at the same target residues.
        # Reusing apply_mask_to_latents with the same masks gives target latents in the
        # exact order/padding as `predicted_latent`, so the two align position-by-position.
        with torch.no_grad():
            ema_mean = self._encode_mean(self.ema_encoder, batch)  # [B, L, latent_z_dim]
        ema_masked_latents = apply_mask_to_latents(ema_mean, context_mask, target_mask)
        ema_target = ema_masked_latents["target_latents"]  # [B, L_tgt, latent_z_dim]

        # Smooth-L1 over real target residues only (drop padding)
        target_padding_mask = masked_latents["target_padding_mask"]  # [B, L_tgt], True = padding
        loss = F.smooth_l1_loss(
            predicted_latent[~target_padding_mask],  # [n_real_targets, latent_z_dim]
            ema_target[~target_padding_mask],        # [n_real_targets, latent_z_dim]
        )
        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        bs = batch["mask_dict"]["coords"].shape[0]
        loss = self.compute_loss(batch)  # scalar
        # Lightning logs this as train_loss_step and train_loss_epoch
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        bs = batch["mask_dict"]["coords"].shape[0]
        loss = self.compute_loss(batch)  # scalar
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update of the target encoder after each training step."""
        with torch.no_grad():
            for online_p, ema_p in zip(
                self.online_encoder.parameters(), self.ema_encoder.parameters()
            ):
                ema_p.data = self.ema_decay * ema_p.data + (1 - self.ema_decay) * online_p.data

    def configure_optimizers(self):
        # Only parameters with requires_grad=True are optimized, which excludes the
        # EMA target encoder (online encoder + predictor are trained).
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.cfg.opt.lr,
            amsgrad=True,
            weight_decay=1e-2,
        )
        return optimizer
