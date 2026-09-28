# JEPA decoder 300k PyMOL structures

The reconstruction files in the historical protein folders were refreshed
from the exact step-300,000 decoder checkpoint:

`Evaluation/structure_rec/checkpoints/jepa-rand-mask-70-target-200k-to-300k_step300000.pt`

Checkpoint SHA256:
`b709a9720cb763906fae4b5c844f3ca735649873d02dc3cf4ab1ed8e4911b4cf`

For each historical protein:

- `<protein>_reconstructed.pdb` is the predicted-sequence direct decoder output.
- `<protein>_reconstructed_idealized.pdb` is the corresponding deterministic
  postprocessed output.
- `<protein>_native.pdb` is the held-out reference used by the frozen
  reconstruction evaluation.

The refreshed historical set is `1kfm_A`, `1nkw_M`, `3d3r_A`, `5a13_G`,
`5fli_A`, `5l0o_A`, `5xr7_A`, `6o14_A`, `7cnw_A`, and `8cbv_A`.
The folders `4etr_A_decoder300k`, `5fli_A_decoder300k`, and
`5v5k_B_decoder300k` contain explicit direct-versus-postprocessed comparison
scenes. Their colors are gray for native, cyan for direct reconstruction, and
orange for postprocessed reconstruction.
