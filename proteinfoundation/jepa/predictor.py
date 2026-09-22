import torch
import torch.nn as nn
from loguru import logger

class CrossAttentionBlock(nn.Module):
    """One cross-attention layer: queries (targets) attend to keys/values (context),
    followed by a feedforward network, both with pre-LayerNorm and residual."""

    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, q, kv, key_padding_mask=None, attn_mask=None):
        attn_out, _ = self.attn(
            self.norm_q(q), kv, kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )                                                # [B, L_tgt, hidden_dim]
        q = q + attn_out                                 # residual_q + attention
        q = q + self.ff(self.norm_ff(q))                 # residual_q + feedforward
        return q  # [B, L_tgt, hidden_dim]


class SelfAttentionBlock(nn.Module):
    """One self-attention layer among target queries only (no context involved),
    so target positions can coordinate with each other before the cross-attention
    block pulls from context. Same pre-LayerNorm + residual + FFN structure as
    CrossAttentionBlock, for architectural consistency -- q attends to q."""

    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, q, key_padding_mask=None):
        normed = self.norm_q(q)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )                                                # [B, L_tgt, hidden_dim]
        q = q + attn_out                                 # residual_q + attention
        q = q + self.ff(self.norm_ff(q))                 # residual_q + feedforward
        return q  # [B, L_tgt, hidden_dim]


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block used by the mask-token JEPA predictor."""

    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        normed = self.norm_attn(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = x + attn_out
        return x + self.ff(self.norm_ff(x))


class JEPAPredictor(nn.Module):
    """JEPA predictor with either cross-attention or I-JEPA-style mask tokens."""

    def __init__(
        self,
        latent_z_dim=8,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        ff_dim=512,
        max_len=256,
        use_self_attn=False,
        predictor_type="cross_attention",
        use_relative_positions=False,
        relative_position_max_distance=128,
    ):
        super().__init__()
        self.latent_z_dim = latent_z_dim
        self.max_len = max_len
        self.use_self_attn = use_self_attn
        self.predictor_type = predictor_type
        self.use_relative_positions = use_relative_positions
        self.relative_position_max_distance = relative_position_max_distance
        if predictor_type not in {"cross_attention", "mask_token_transformer"}:
            raise ValueError(f"Unknown predictor_type: {predictor_type!r}")
        if use_relative_positions and predictor_type != "cross_attention":
            raise ValueError(
                "use_relative_positions currently supports only the old "
                "cross_attention predictor"
            )
        if use_relative_positions and use_self_attn:
            raise ValueError(
                "use_relative_positions is not compatible with target self-attention: "
                "generic target queries require a target-target relative attention bias"
            )
        if relative_position_max_distance < 1:
            raise ValueError("relative_position_max_distance must be >= 1")

        # project context latents up to hidden dim
        self.context_proj = nn.Linear(latent_z_dim, hidden_dim)

        if use_relative_positions:
            self.pos_emb = None
            self.target_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.relative_position_bias = nn.Embedding(
                2 * relative_position_max_distance + 1, num_heads
            )
            nn.init.normal_(self.target_query, std=0.02)
            nn.init.normal_(self.relative_position_bias.weight, std=0.02)
        else:
            # Historical behavior: one shared learned absolute positional
            # embedding table for context and target positions.
            self.pos_emb = nn.Embedding(max_len, hidden_dim)
            self.target_query = None
            self.relative_position_bias = None

        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)]
        ) if predictor_type == "cross_attention" else None
        self.self_attn_layers = (
            nn.ModuleList(
                [SelfAttentionBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)]
            )
            if use_self_attn and predictor_type == "cross_attention"
            else None
        )

        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, hidden_dim))
            if predictor_type == "mask_token_transformer"
            else None
        )
        self.transformer_layers = (
            nn.ModuleList(
                [TransformerBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)]
            )
            if predictor_type == "mask_token_transformer"
            else None
        )
        self.predictor_norm = (
            nn.LayerNorm(hidden_dim)
            if predictor_type == "mask_token_transformer"
            else None
        )

        # project predicted queries back down to latent space
        self.out_proj = nn.Linear(hidden_dim, latent_z_dim)

        self._self_exclude_logged = False  # one-time startup sanity log, not per-step

    def _relative_cross_attention_bias(
        self,
        context_indices,
        target_indices,
        key_padding_mask,
        blocked_context,
        dtype,
    ):
        """Per-head additive bias for every target i / context j pair.

        Distances are clipped before lookup, so the table is independent of
        maximum protein length. Context padding and optional direct-access
        restrictions are folded into the same floating-point attention mask.
        """
        relative = target_indices[:, :, None] - context_indices[:, None, :]
        relative = relative.clamp(
            -self.relative_position_max_distance,
            self.relative_position_max_distance,
        )
        buckets = relative + self.relative_position_max_distance
        bias = self.relative_position_bias(buckets)
        bias = bias.permute(0, 3, 1, 2).reshape(
            -1, target_indices.shape[1], context_indices.shape[1]
        )
        bias = bias.to(dtype=dtype)

        blocked = None
        if key_padding_mask is not None:
            blocked = key_padding_mask[:, None, :].expand(
                -1, target_indices.shape[1], -1
            )
        if blocked_context is not None:
            blocked = (
                blocked_context
                if blocked is None
                else torch.logical_or(blocked, blocked_context)
            )
        if blocked is not None:
            num_heads = self.relative_position_bias.embedding_dim
            blocked = blocked.repeat_interleave(num_heads, dim=0)
            bias = bias.masked_fill(blocked, float("-inf"))
        return bias

    def forward(self, context_latents, context_indices, target_indices, key_padding_mask=None,
                target_padding_mask=None, action_z=None, action_position=None,
                self_exclude=False, context_access=None):
        if self.use_relative_positions:
            assert int(target_indices.min()) >= 0, "target indices must be non-negative"
            assert int(context_indices.min()) >= 0, "context indices must be non-negative"
        else:
            assert int(target_indices.max()) < self.max_len, "target index exceeds max_len positional table"
            assert int(context_indices.max()) < self.max_len, "context index exceeds max_len positional table"
        if context_access is None:
            context_access = "exclude_self" if self_exclude else "all"
        elif self_exclude:
            raise ValueError("pass either self_exclude or context_access, not both")
        if context_access not in {"all", "exclude_self", "self_only"}:
            raise ValueError(f"Unknown context_access mode: {context_access!r}")

        same_residue = target_indices[:, :, None] == context_indices[:, None, :]
        if key_padding_mask is not None:
            same_residue = same_residue & ~key_padding_mask[:, None, :]

        if self.use_relative_positions:
            q = self.target_query.expand(
                target_indices.shape[0], target_indices.shape[1], -1
            )
        else:
            # Historical absolute-position target queries.
            q = self.pos_emb(target_indices)                # [B, L_tgt, hidden_dim]
        if action_z is not None:
            # mutation of target residue
            at_action_pos_tgt = (target_indices == action_position.unsqueeze(1))  # [B, L_tgt]
            q = q + action_z * at_action_pos_tgt.unsqueeze(-1).to(action_z.dtype)  # [B, L_tgt, hidden_dim]

        # keys/values: projected context latents plus their positional embeddings
        kv = self.context_proj(context_latents)             # [B, L_ctx, hidden_dim]
        if not self.use_relative_positions:
            kv = kv + self.pos_emb(context_indices)         # [B, L_ctx, hidden_dim]
        if action_z is not None:
            # mutation of context residue
            at_action_pos_ctx = (context_indices == action_position.unsqueeze(1))  # [B, L_ctx]
            kv = kv + action_z * at_action_pos_ctx.unsqueeze(-1).to(action_z.dtype)

        if self.predictor_type == "mask_token_transformer":
            target_tokens = self.mask_token.expand(q.shape[0], q.shape[1], -1) + q
            all_tokens = torch.cat((kv, target_tokens), dim=1)
            all_padding = None
            if key_padding_mask is not None or target_padding_mask is not None:
                if key_padding_mask is None:
                    key_padding_mask = torch.zeros(
                        kv.shape[:2], dtype=torch.bool, device=kv.device
                    )
                if target_padding_mask is None:
                    target_padding_mask = torch.zeros(
                        q.shape[:2], dtype=torch.bool, device=q.device
                    )
                all_padding = torch.cat((key_padding_mask, target_padding_mask), dim=1)

            joint_attn_mask = None
            if context_access != "all":
                blocked_context = (
                    same_residue if context_access == "exclude_self" else ~same_residue
                )
                if context_access == "self_only" and key_padding_mask is not None:
                    blocked_context = blocked_context & ~key_padding_mask[:, None, :]
                total_len = all_tokens.shape[1]
                joint_attn_mask = torch.zeros(
                    all_tokens.shape[0], total_len, total_len,
                    dtype=torch.bool,
                    device=all_tokens.device,
                )
                joint_attn_mask[:, kv.shape[1] :, : kv.shape[1]] = blocked_context
                num_heads = self.transformer_layers[0].attn.num_heads
                joint_attn_mask = joint_attn_mask.repeat_interleave(num_heads, dim=0)
            for layer in self.transformer_layers:
                all_tokens = layer(
                    all_tokens,
                    key_padding_mask=all_padding,
                    attn_mask=joint_attn_mask,
                )
            return self.out_proj(self.predictor_norm(all_tokens[:, kv.shape[1] :]))

        cross_attn_mask = None
        blocked_context = None
        if context_access != "all":
            blocked_context = (
                same_residue if context_access == "exclude_self" else ~same_residue
            )
            if context_access == "self_only" and key_padding_mask is not None:
                blocked_context = blocked_context & ~key_padding_mask[:, None, :]

            if not self.use_relative_positions:
                num_heads = self.layers[0].attn.num_heads
                cross_attn_mask = blocked_context.repeat_interleave(num_heads, dim=0)

            if not self._self_exclude_logged:
                logger.info(
                    f"JEPAPredictor: context_access={context_access!r}; direct "
                    f"target-query/context-token attention is restricted."
                )
                self._self_exclude_logged = True

        cross_attn_key_padding = key_padding_mask
        if self.use_relative_positions:
            cross_attn_mask = self._relative_cross_attention_bias(
                context_indices,
                target_indices,
                key_padding_mask,
                blocked_context,
                dtype=q.dtype,
            )
            cross_attn_key_padding = None

        if self.self_attn_layers is not None:
            for self_attn_layer in self.self_attn_layers:
                q = self_attn_layer(q, key_padding_mask=target_padding_mask)
        for layer in self.layers:
            q = layer(
                q,
                kv,
                key_padding_mask=cross_attn_key_padding,
                attn_mask=cross_attn_mask,
            )  # [B, L_tgt, hidden_dim]

        # project back to latent space
        pred = self.out_proj(q)                             # [B, L_tgt, latent_z_dim]
        return pred
