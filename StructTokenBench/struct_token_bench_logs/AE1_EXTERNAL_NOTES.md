# AE1-external rows in summary_results.csv (`branch`/experiment = `ae1-external`)

NVIDIA's published La-Proteina AE (`checkpoints_laproteina/AE1_ucond_512.ckpt`), run through the
same 12-task protocol as the other 8 checkpoints. Not one of "my 8" -- an external reference
point. Caveats, not a matched ablation:

- **Training corpus**: AFDB (`afdb_rep_v4`), not PDB. Zero overlap with any split my own 8
  checkpoints or their eval data come from.
- **Training length**: 640k cumulative steps (500k pretrain + 140k fine-tune, weights-only
  resume -- see decoder-reconstruction task notes), vs my 100k.
- **Input features**: trained with `x1_bb_angles` + `x1_sidechain_angles` (backbone/sidechain
  torsion angles) in addition to my feature set. Confirmed via a one-protein load test that
  `WrappedLaProteinaTokenizer` reads this from AE1's own saved checkpoint config automatically
  (no hardcoding to my reduced feature set, no fix needed) -- so this is a genuine difference
  in what the model was trained with, not a pipeline bug.
- **NOT capacity-confounded**: `latent_z_dim=8`, identical to my 8 checkpoints -- verified
  fresh from the checkpoint's own saved config, `AutoEncoder.load_from_checkpoint(...).latent_dim`,
  and `model.d_model=8` in the StructTokenBench probe head. No dimensionality mismatch.
