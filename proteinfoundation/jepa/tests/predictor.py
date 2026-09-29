"""Tiny standalone test for JEPAPredictor.

Run with:  python -m proteinfoundation.jepa.tests.predictor
"""

import torch

from proteinfoundation.jepa.predictor import JEPAPredictor


torch.manual_seed(0)
B, L_ctx, L_tgt, z = 2, 10, 6, 8
model = JEPAPredictor(latent_z_dim=z).eval()

context_latents = torch.randn(B, L_ctx, z)                  # [B, L_ctx, z]
context_indices = torch.randint(0, 256, (B, L_ctx))         # [B, L_ctx]
target_indices = torch.randint(0, 256, (B, L_tgt))          # [B, L_tgt]

# Scenario 1: output has the right shape
with torch.no_grad():
    out_ref = model(context_latents, context_indices, target_indices)  # [B, L_tgt, z]
assert out_ref.shape == (B, L_tgt, z)
print(f"scenario 1 (shape): out = {tuple(out_ref.shape)} == {(B, L_tgt, z)}")

# Scenario 2: padded context positions masked by key_padding_mask must be ignored.
# Append garbage context columns and mask them out -> predictions must not change.
pad = 4
ctx_padded = torch.cat([context_latents, torch.randn(B, pad, z)], dim=1)            # [B, L_ctx+pad, z]
idx_padded = torch.cat([context_indices, torch.zeros(B, pad, dtype=torch.long)], 1)  # [B, L_ctx+pad]
key_padding_mask = torch.zeros(B, L_ctx + pad, dtype=torch.bool)                     # [B, L_ctx+pad]
key_padding_mask[:, L_ctx:] = True  # True = ignore the appended pad columns

with torch.no_grad():
    out_masked = model(ctx_padded, idx_padded, target_indices, key_padding_mask=key_padding_mask)
diff_masked = (out_ref - out_masked).abs().max().item()
assert diff_masked < 1e-5, f"masked padding leaked into output (diff={diff_masked})"
print(f"scenario 2 (mask ignores padding): max abs diff = {diff_masked:.2e} (< 1e-5)")

# Scenario 3: negative control - same garbage WITHOUT masking SHOULD change the output.
with torch.no_grad():
    out_unmasked = model(ctx_padded, idx_padded, target_indices)
diff_unmasked = (out_ref - out_unmasked).abs().max().item()
assert diff_unmasked > 1e-3, f"unmasked padding had no effect (diff={diff_unmasked})"
print(f"scenario 3 (no mask changes output): max abs diff = {diff_unmasked:.2e} (> 1e-3)")

# Scenario 4: gradients flow (trainable end to end)
out = model(context_latents, context_indices, target_indices)
out.sum().backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
print("scenario 4 (backward): gradients flow to all parameters")


# Scenario 5: direct self-context policies do what their names claim. A single
# layer is used for the joint predictor so this tests its direct attention edge;
# deeper joint layers can intentionally propagate information indirectly.
for predictor_type in ("cross_attention", "mask_token_transformer"):
    policy_model = JEPAPredictor(
        latent_z_dim=z,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ff_dim=64,
        predictor_type=predictor_type,
    ).eval()
    overlap_indices = torch.arange(4).unsqueeze(0)
    overlap_context = torch.randn(1, 4, z)
    changed_self = overlap_context.clone()
    changed_self[:, 1] += 100.0
    changed_neighbours = overlap_context.clone()
    changed_neighbours[:, [0, 2, 3]] += 100.0

    with torch.no_grad():
        excluded_ref = policy_model(
            overlap_context,
            overlap_indices,
            overlap_indices,
            context_access="exclude_self",
        )
        excluded_changed = policy_model(
            changed_self,
            overlap_indices,
            overlap_indices,
            context_access="exclude_self",
        )
        self_only_ref = policy_model(
            overlap_context,
            overlap_indices,
            overlap_indices,
            context_access="self_only",
        )
        self_only_changed = policy_model(
            changed_neighbours,
            overlap_indices,
            overlap_indices,
            context_access="self_only",
        )

    assert torch.allclose(excluded_ref[:, 1], excluded_changed[:, 1], atol=1e-5)
    assert torch.allclose(self_only_ref[:, 1], self_only_changed[:, 1], atol=1e-5)
    print(f"scenario 5 ({predictor_type}): direct context policies verified")


# Scenario 6: both relative-position predictors have no absolute embedding
# lookup, are translation invariant, distinguish targets through i-j relations,
# and ignore padded context tokens.
for predictor_type in ("cross_attention", "mask_token_transformer"):
    relative_model = JEPAPredictor(
        latent_z_dim=z,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        ff_dim=64,
        predictor_type=predictor_type,
        use_relative_positions=True,
        relative_position_max_distance=64,
    ).eval()
    assert relative_model.pos_emb is None

    relative_context = torch.randn(1, 5, z)
    relative_context_indices = torch.tensor([[0, 5, 11, 29, 40]])
    relative_target_indices = torch.tensor([[8, 20, 35]])
    with torch.no_grad():
        relative_ref = relative_model(
            relative_context, relative_context_indices, relative_target_indices
        )
        relative_shifted = relative_model(
            relative_context,
            relative_context_indices + 37,
            relative_target_indices + 37,
        )

    assert torch.allclose(relative_ref, relative_shifted, atol=1e-6), (
        "relative predictor changed under a shared absolute-index translation"
    )
    target_difference = (relative_ref[:, 0] - relative_ref[:, 1]).abs().max().item()
    assert target_difference > 1e-6, "relative biases did not distinguish target queries"

    relative_padded_context = torch.cat(
        [relative_context, torch.randn(1, 3, z)], dim=1
    )
    relative_padded_indices = torch.cat(
        [relative_context_indices, torch.zeros(1, 3, dtype=torch.long)], dim=1
    )
    relative_padding = torch.tensor([[False] * 5 + [True] * 3])
    with torch.no_grad():
        relative_padded = relative_model(
            relative_padded_context,
            relative_padded_indices,
            relative_target_indices,
            key_padding_mask=relative_padding,
        )
    assert torch.allclose(relative_ref, relative_padded, atol=1e-5), (
        "padded context affected relative-position predictions"
    )

    relative_model.zero_grad(set_to_none=True)
    relative_train_out = relative_model(
        relative_context, relative_context_indices, relative_target_indices
    )
    relative_train_out.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in relative_model.parameters()
        if parameter.requires_grad
    )
    print(
        f"scenario 6 ({predictor_type}, relative): translation invariant, "
        "target-specific, padding-safe, and differentiable"
    )

print("all checks passed")
