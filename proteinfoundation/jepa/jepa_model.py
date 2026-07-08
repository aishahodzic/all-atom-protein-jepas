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
from typing import Dict, List

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from sklearn.decomposition import PCA

from proteinfoundation.jepa.masking import apply_mask_to_latents, random_residue_mask
from proteinfoundation.jepa.predictor import JEPAPredictor
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


class JEPAModel(L.LightningModule):
    def __init__(self, cfg, ema_decay=0.996):
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

        # For scaling/* logging, mirroring the autoencoder (proteinfoundation/partial_autoencoder/autoencoder.py)
        self.nsamples_processed = 0
        self.nparams_enc = sum(
            p.numel() for p in self.online_encoder.parameters() if p.requires_grad
        )
        self.nparams_predictor = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        try:
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except KeyError:
            self.nsamples_processed = 0

    def _encode_mean(self, encoder, batch: Dict) -> torch.Tensor:
        """Run an encoder and return only the deterministic mean latent (no variance /sampling)."""
        return encoder(batch)["mean"]  # [B, L, latent_z_dim]

    def compute_loss(self, batch: Dict, stage: str = "train") -> Dict[str, torch.Tensor]:
        """Pure JEPA latent-prediction loss for one batch."""
        # this mask is for padding vs. real residues [1 0 1 1 0 ...]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [B, L] boolean
        batch["mask"] = mask

        # Online encoder mean latents for all residues
        online_mean = self._encode_mean(self.online_encoder, batch)  # [B, L, latent_z_dim]

        # Std of the online latents over real (non-padding) residues, per latent dim,
        # averaged over dims. A collapsing encoder (all latents converging to the same
        # vector) shows up as this dropping toward 0, which the loss alone won't catch
        # since predicting a collapsed target is "easy".
        latent_std = online_mean[mask.bool()].std(dim=0).mean()

        # Full latent stats (max/min/mean/median/std + histogram), ported from the
        # autoencoder's log_tensor_statistics.
        self.log_tensor_statistics(
            bs=online_mean.shape[0],
            v=online_mean,
            log_prefix=f"{stage}_stats_latent_mean_LS",
            mask=mask.bool(),
            histogram_every_n=5000,
        )

        # PCA scatter of the online latents, logged to wandb every `pca_every_n` steps.
        # Ported from the autoencoder's train_stats_latent_z_PCA logging: lets you see
        # the *shape* of the latent space (e.g. spread across several axes vs. squashed
        # onto one), which latent_std alone doesn't show.
        self.log_pca(
            v=online_mean,
            mask=mask.bool(),
            log_prefix=f"{stage}_stats_latent_z_PCA",
            every_n=5000,
        )

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
        real_target_mask = ~target_padding_mask
        loss = F.smooth_l1_loss(
            predicted_latent[real_target_mask],  # [n_real_targets, latent_z_dim]
            ema_target[real_target_mask],        # [n_real_targets, latent_z_dim]
        )

        # Predictor-output vs. target-encoder stats and their cosine similarity. AE has
        # nothing analogous (no predictor to compare against a target), but for JEPA this
        # is the closest thing to a "reconstruction quality" chart: predicted_latent should
        # track ema_target in scale/spread (log_tensor_statistics on both) and direction
        # (cosine similarity -> 1 as the predictor gets better, independent of magnitude).
        self.log_tensor_statistics(
            bs=predicted_latent.shape[0],
            v=predicted_latent,
            log_prefix=f"{stage}_stats_latent_predicted_LS",
            mask=real_target_mask,
            histogram_every_n=5000,
        )
        self.log_tensor_statistics(
            bs=ema_target.shape[0],
            v=ema_target,
            log_prefix=f"{stage}_stats_latent_target_LS",
            mask=real_target_mask,
            histogram_every_n=5000,
        )
        cos_sim = F.cosine_similarity(
            predicted_latent[real_target_mask], ema_target[real_target_mask], dim=-1
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

        return {"loss": loss, "latent_std": latent_std}

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
