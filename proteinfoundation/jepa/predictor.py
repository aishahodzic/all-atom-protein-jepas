import torch.nn as nn

#block: q → norm(q) → MHA → q' = add_residual(q) → norm_q' → FF → p" = add_residual(q') -> output p" 
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

    def forward(self, q, kv, key_padding_mask=None):
        # q:  [B, L_tgt, hidden_dim]   (target queries)
        # kv: [B, L_ctx, hidden_dim]   (context keys/values)
        # key_padding_mask: [B, L_ctx] bool, True at padded context positions to ignore
        attn_out, _ = self.attn(
            self.norm_q(q), kv, kv, key_padding_mask=key_padding_mask
        )                                                # [B, L_tgt, hidden_dim]
        q = q + attn_out                                 # residual_q + attention
        q = q + self.ff(self.norm_ff(q))                 # residual_q + feedforward
        return q  # [B, L_tgt, hidden_dim]


class JEPAPredictor(nn.Module):
    """Cross-attention predictor mapping context latents -> target latents."""

    def __init__(
        self,
        latent_z_dim=8,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        ff_dim=512,
        max_len=256,
    ):
        super().__init__()
        self.latent_z_dim = latent_z_dim
        self.max_len = max_len

        # project context latents up to hidden dim
        self.context_proj = nn.Linear(latent_z_dim, hidden_dim)

        # one shared learned positional encoding for both context and target positions
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

        # project predicted queries back down to latent space
        self.out_proj = nn.Linear(hidden_dim, latent_z_dim)

#forwad: kv latent + kv pos. embedding -> cross-attention (q vs kv) -> feedforward -> output latent
    def forward(self, context_latents, context_indices, target_indices, key_padding_mask=None):
        # context_latents: [B, L_ctx, latent_z_dim]  (encoder mean of visible residues)
        # context_indices: [B, L_ctx]  (long)  residue indices of context residues
        # target_indices:  [B, L_tgt]  (long)  residue indices of target residues
        # key_padding_mask: [B, L_ctx] bool, True at padded context positions to ignore
        assert int(target_indices.max()) < self.max_len, "target index exceeds max_len positional table"
        assert int(context_indices.max()) < self.max_len, "context index exceeds max_len positional table"

        # queries are just positional embeddings of the target residues
        q = self.pos_emb(target_indices)                    # [B, L_tgt, hidden_dim]

        # keys/values: projected context latents plus their positional embeddings
        kv = self.context_proj(context_latents)             # [B, L_ctx, hidden_dim]
        kv = kv + self.pos_emb(context_indices)             # [B, L_ctx, hidden_dim]

        # cross-attention stack (no self-attention)
        for layer in self.layers:
            q = layer(q, kv, key_padding_mask=key_padding_mask)  # [B, L_tgt, hidden_dim]

        # project back to latent space
        pred = self.out_proj(q)                             # [B, L_tgt, latent_z_dim]
        return pred
