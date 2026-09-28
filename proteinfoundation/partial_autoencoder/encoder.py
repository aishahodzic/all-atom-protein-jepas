from typing import Dict

import torch

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition


class EncoderTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(EncoderTransformer, self).__init__()
        self.nlayers = kwargs["encoder"]["nlayers"]
        self.token_dim = kwargs["encoder"]["token_dim"]
        self.pair_repr_dim = kwargs["encoder"]["pair_repr_dim"]
        self.update_pair_repr = kwargs["encoder"]["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["encoder"]["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["encoder"]["use_tri_mult"]
        self.pair_update_mode = str(
            kwargs["encoder"].get("pair_update_mode", "additive")
        )
        self.pair_update_hidden_dim = int(
            kwargs["encoder"].get("pair_update_hidden_dim", 64)
        )
        self.tri_mult_c = int(kwargs["encoder"].get("tri_mult_c", 196))
        self.normalize_latent = kwargs["encoder"]["normalize_latent"]
        self.pair_context_to_sequence = bool(
            kwargs["encoder"].get("pair_context_to_sequence", False)
        )

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_seq"],
            dim_feats_out=kwargs["encoder"]["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs["encoder"],
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_cond_seq"],
            dim_feats_out=kwargs["encoder"]["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs["encoder"],
        )

        self.transition_c_1 = Transition(
            kwargs["encoder"]["dim_cond"], expansion_factor=2
        )
        self.transition_c_2 = Transition(
            kwargs["encoder"]["dim_cond"], expansion_factor=2
        )

        # To get pair representation
        self.pair_rep_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_pair_repr"],
            dim_feats_out=kwargs["encoder"]["pair_repr_dim"],
            use_ln_out=False,
            mode="pair",
            **kwargs["encoder"],
        )

        # Pair-biased attention only lets pair features modulate attention
        # weights; it cannot introduce pair information into token values.  In
        # the pair-only-position ablation this optional bridge gives residue i
        # a masked mean of its pair features (i, j) over visible key residues j.
        # It is disabled by default so historical encoder state dicts and
        # forward behavior remain exactly unchanged.
        if self.pair_context_to_sequence:
            self.pair_context_norm = torch.nn.LayerNorm(self.pair_repr_dim)
            self.pair_context_projection = torch.nn.Linear(
                self.pair_repr_dim, self.token_dim, bias=False
            )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["encoder"]["token_dim"],
                    dim_pair=kwargs["encoder"]["pair_repr_dim"],
                    nheads=kwargs["encoder"]["nheads"],
                    dim_cond=kwargs["encoder"]["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=kwargs["encoder"]["use_qkln"],
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["encoder"]["token_dim"],
                            pair_dim=kwargs["encoder"]["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            tri_mult_c=self.tri_mult_c,
                            sequence_to_pair_mode=self.pair_update_mode,
                            sequence_to_pair_hidden_dim=self.pair_update_hidden_dim,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.latent_decoder_mean_n_log_scale = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(
                self.token_dim, int(2 * kwargs["latent_z_dim"]), bias=False
            ),
        )

        self.ln_z = torch.nn.Identity()

    def add_pair_context_to_sequence(
        self,
        seqs: torch.Tensor,
        pair_rep: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inject a padding-safe per-query mean of visible pair features."""
        if not self.pair_context_to_sequence:
            return seqs
        pair_mask = query_mask[:, :, None] & key_mask[:, None, :]
        pair_mask_float = pair_mask[..., None].to(pair_rep.dtype)
        denominator = pair_mask_float.sum(dim=2).clamp_min(1.0)
        pair_context = (pair_rep * pair_mask_float).sum(dim=2) / denominator
        pair_context = self.pair_context_projection(
            self.pair_context_norm(pair_context)
        )
        return (seqs + pair_context) * query_mask[..., None].to(seqs.dtype)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        compact_kv: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            batch: batch from dataset

        Returns:
            Dictionary
            ...
        """
        historical_mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]
        if query_mask is None:
            query_mask = historical_mask
        if key_mask is None:
            key_mask = query_mask
        if (
            query_mask.shape != historical_mask.shape
            or key_mask.shape != historical_mask.shape
        ):
            raise ValueError(
                "query_mask and key_mask must match the encoder residue layout "
                f"{tuple(historical_mask.shape)}"
            )
        if query_mask.dtype != torch.bool or key_mask.dtype != torch.bool:
            raise TypeError("query_mask and key_mask must be boolean")
        if self.update_pair_repr and not torch.equal(query_mask, key_mask):
            raise ValueError(
                "asymmetric query/key masks are unsupported when update_pair_repr=true"
            )
        if compact_kv and torch.equal(query_mask, key_mask):
            raise ValueError(
                "compact_kv is reserved for asymmetric visible-only K/V attention"
            )
        mask = query_mask

        # Conditioning variables
        c = self.cond_factory(batch)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(batch)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_rep_factory(batch)  # [b, n, n, pair_dim]
        seqs = self.add_pair_context_to_sequence(
            seqs, pair_rep, query_mask, key_mask
        )

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs,
                pair_rep,
                c,
                mask,
                key_mask=key_mask,
                compact_kv=compact_kv,
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, pair_dim]

        # Get final coordinates
        flat_out = (
            self.latent_decoder_mean_n_log_scale(seqs) * mask[..., None]
        )  # [b, n, 2 * latent_dim]
        flat_out = flat_out * mask[..., None]
        mean, log_scale = torch.chunk(
            flat_out, chunks=2, dim=-1
        )  # [b, n, latent_dim] each

        z = mean + torch.randn_like(log_scale) * torch.exp(
            log_scale
        )  # [b, n, latent_dim]

        z_pre_ln = z
        z = self.ln_z(z) * mask[..., None]  # [b, n, latent_dim]
        output = {
            "mean": mean,
            "log_scale": log_scale,
            "z_latent": z,
            "z_latent_pre_ln": z_pre_ln,
        }
        return output
