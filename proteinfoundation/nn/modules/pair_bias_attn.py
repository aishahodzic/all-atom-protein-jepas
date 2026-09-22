# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, einsum, nn

from proteinfoundation.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        **kawrgs,  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
        *,
        key_mask: Optional[Tensor] = None,
        compact_kv: bool = False,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies, or the query mask when
            ``compact_kv`` is enabled
        :param key_mask: key/value rows retained by compact attention
        :param compact_kv: physically gather the active key/value rows before
            constructing attention logits
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        if compact_kv:
            return self._forward_compact_kv(
                node_feats=node_feats,
                pair_feats=pair_feats,
                query_mask=mask,
                key_mask=key_mask,
            )
        if key_mask is not None:
            raise ValueError("key_mask is only accepted when compact_kv=true")
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    @staticmethod
    def _compact_key_layout(key_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return ordered source indices and padding mask for ragged K/V rows."""
        if key_mask.ndim != 2 or key_mask.dtype != torch.bool:
            raise ValueError("compact key_mask must be boolean [B, L]")
        key_counts = key_mask.sum(dim=1)
        if (key_counts == 0).any():
            rows = torch.where(key_counts == 0)[0].tolist()
            raise ValueError(f"compact attention has no key/value rows: rows={rows}")
        batch_size, length = key_mask.shape
        compact_length = int(key_counts.max())
        source_rows = torch.arange(length, device=key_mask.device).expand(
            batch_size, -1
        )
        source_rows = source_rows.masked_fill(~key_mask, length)
        source_rows = source_rows.sort(dim=1).values[:, :compact_length]
        compact_mask = (
            torch.arange(compact_length, device=key_mask.device)[None]
            < key_counts[:, None]
        )
        source_rows = source_rows.masked_fill(~compact_mask, 0)
        return source_rows, compact_mask

    @staticmethod
    def _gather_key_rows(values: Tensor, source_rows: Tensor) -> Tensor:
        if values.ndim != 3 or source_rows.ndim != 2:
            raise ValueError("compact K/V gather expects [B,L,D] and [B,K]")
        return values.gather(
            1, source_rows[..., None].expand(-1, -1, values.shape[-1])
        )

    @staticmethod
    def _gather_pair_key_columns(values: Tensor, source_rows: Tensor) -> Tensor:
        if values.ndim != 4 or source_rows.ndim != 2:
            raise ValueError("compact pair gather expects [B,L,L,D] and [B,K]")
        return values.gather(
            2,
            source_rows[:, None, :, None].expand(
                -1, values.shape[1], -1, values.shape[-1]
            ),
        )

    def _forward_compact_kv(
        self,
        *,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        query_mask: Optional[Tensor],
        key_mask: Optional[Tensor],
    ) -> Tensor:
        """Rectangular Q-by-visible-K/V attention with unchanged parameters."""
        if query_mask is None or key_mask is None:
            raise ValueError("compact attention requires query_mask and key_mask")
        if (
            query_mask.ndim != 2
            or key_mask.ndim != 2
            or query_mask.dtype != torch.bool
            or key_mask.dtype != torch.bool
            or query_mask.shape != key_mask.shape
            or query_mask.shape != node_feats.shape[:2]
        ):
            raise ValueError(
                "compact query_mask and key_mask must be boolean [B, L] "
                "matching node_feats"
            )
        if (key_mask & ~query_mask).any():
            raise ValueError("compact key/value rows must also be real query rows")
        if exists(pair_feats) and pair_feats.shape[:3] != (
            node_feats.shape[0],
            node_feats.shape[1],
            node_feats.shape[1],
        ):
            raise ValueError("compact pair_feats must have shape [B, L, L, D]")

        source_rows, compact_key_mask = self._compact_key_layout(key_mask)
        node_feats, h = self.node_norm(node_feats), self.heads
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self._gather_key_rows(self.k_layer_norm(k), source_rows)
        v = self._gather_key_rows(v, source_rows)
        g = self.to_g(node_feats)
        if exists(pair_feats):
            pair_feats = self._gather_pair_key_columns(pair_feats, source_rows)
            pair_feats = self.pair_norm(pair_feats)
            b = rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
        else:
            b = 0
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h),
            (q, k, v, g),
        )
        attention_mask = query_mask[:, :, None] & compact_key_mask[:, None, :]
        attn_feats = self._attn(q, k, v, b, attention_mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(
        self, x, pair_rep, cond, mask, key_mask=None, compact_kv: bool = False
    ):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary query/output mask, shape [b, n]
            key_mask: Optional binary key/value mask, shape [b, n]. If omitted,
                the query mask is reused, preserving the historical symmetric
                attention behavior.
            compact_kv: Physically gather key/value rows before attention.

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        if key_mask is None:
            key_mask = mask
        x = self.adaln(x, cond, mask)
        if compact_kv:
            x = self.mha(
                node_feats=x,
                pair_feats=pair_rep,
                mask=mask,
                key_mask=key_mask,
                compact_kv=True,
            )
        else:
            pair_mask = mask[:, :, None] * key_mask[:, None, :]  # [b, n, n]
            x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]
