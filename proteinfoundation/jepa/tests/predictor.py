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

print("all checks passed")
