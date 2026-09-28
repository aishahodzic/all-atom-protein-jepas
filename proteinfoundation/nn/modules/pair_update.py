import torch
from torch.utils.checkpoint import checkpoint

from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
        sequence_to_pair_mode="additive",
        sequence_to_pair_hidden_dim=64,
    ):
        super().__init__()

        self.use_tri_mult = use_tri_mult
        self.sequence_to_pair_mode = str(sequence_to_pair_mode)
        if self.sequence_to_pair_mode not in {"additive", "gated_multiplicative"}:
            raise ValueError(
                "sequence_to_pair_mode must be 'additive' or "
                f"'gated_multiplicative', got {self.sequence_to_pair_mode!r}"
            )
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        if self.sequence_to_pair_mode == "additive":
            # Historical behavior. Keep the parameterization unchanged unless
            # the explicitly gated evolving-pair experiment is selected.
            self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)
        else:
            hidden_dim = int(sequence_to_pair_hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("sequence_to_pair_hidden_dim must be positive")
            # A low-rank, single-sequence analogue of an outer-product-mean
            # update. Unlike a(x_i) + b(x_j), the elementwise product can form
            # pair-specific content. Gates and a zero-initialized output keep
            # the residual branch inactive at initialization.
            self.linear_x = torch.nn.Linear(
                token_dim, int(4 * hidden_dim), bias=False
            )
            self.sequence_to_pair_norm = torch.nn.LayerNorm(hidden_dim)
            self.sequence_to_pair_out = torch.nn.Linear(
                hidden_dim, pair_dim, bias=False
            )
            torch.nn.init.zeros_(self.sequence_to_pair_out.weight)

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicationOutgoing(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
            self.tri_mult_in = TriangleMultiplicationIncoming(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
        self.transition_out = PairTransition(
            c_z=pair_dim, n=expansion_factor_transition
        )

    def _sequence_to_pair_update(self, x):
        """Return a directed, gated multiplicative update for every (i, j)."""
        a, a_gate, b, b_gate = self.linear_x(self.layer_norm_in(x)).chunk(
            4, dim=-1
        )
        a = a * torch.sigmoid(a_gate)
        b = b * torch.sigmoid(b_gate)
        pair_update = a[:, :, None, :] * b[:, None, :, :]
        return self.sequence_to_pair_out(
            self.sequence_to_pair_norm(pair_update)
        )

    def _apply_mask(self, pair_rep, pair_mask):
        """
        pair_rep has shape [b, n, n, pair_dim]
        pair_mask has shape [b, n, n]
        """
        return pair_rep * pair_mask[..., None]

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [b, n, token_dim]
            pair_rep: Input pair representation, shape [b, n, n, pair_dim]
            mask: binary mask, shape [b, n]

        Returns:
            Updated pair representation, shape [b, n, n, pair_dim].
        """
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        x = x * mask[..., None]  # [b, n, token_dim]
        if self.sequence_to_pair_mode == "additive":
            x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(
                2, dim=-1
            )  # [b, n, pair_dim] each
            pair_rep = (
                pair_rep + x_proj_1[:, None, :, :] + x_proj_2[:, :, None, :]
            )  # [b, n, n, pair_dim]
        else:
            pair_rep = pair_rep + checkpoint(
                self._sequence_to_pair_update,
                x,
                use_reentrant=False,
            )
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        if self.use_tri_mult:
            pair_rep = pair_rep + checkpoint(
                self.tri_mult_out,
                *(pair_rep, pair_mask * 1.0),
                use_reentrant=False,
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
            pair_rep = pair_rep + checkpoint(
                self.tri_mult_in,
                *(pair_rep, pair_mask * 1.0),
                use_reentrant=False,
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        pair_rep = pair_rep + checkpoint(
            self.transition_out,
            *(pair_rep, pair_mask * 1.0),
            use_reentrant=False,
        )
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        return pair_rep
