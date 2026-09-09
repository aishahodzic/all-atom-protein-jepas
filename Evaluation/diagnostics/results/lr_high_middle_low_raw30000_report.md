# High, Middle and Low LR: raw encoders at 30,000 steps

Same frozen protocol: 5,224 held-out proteins, 748,402-residue pool, 15,000 identical sampled residues. Raw encoder, no EMA. Sample SHA256: `6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8`. All checkpoints report actual global_step=30000.

| Metric | High | Middle | Low |
|---|---:|---:|---:|
| residue_effective_rank | 7.927478 | 7.903403 | 7.839918 |
| residue_embedding_std | 0.970976 | 0.966231 | 1.002558 |
| latent_dimension_std_min | 0.928030 | 0.889585 | 0.928417 |
| latent_dimension_std_max | 1.045899 | 1.060332 | 1.046484 |
| latent_isotropy_ratio | 1.127009 | 1.191940 | 1.127171 |
| aa_purity_at_10_cross_protein | 0.077573 | 0.075973 | 0.079980 |
| ss_purity_at_10_cross_protein | 0.516000 | 0.504493 | 0.515393 |
| rsa_knn10_spearman | 0.149632 | 0.158824 | 0.171107 |
| log_chain_length_knn10_spearman | 0.412651 | 0.418888 | 0.341654 |
| raw_index_knn10_spearman | 0.275511 | 0.265640 | 0.212660 |
| normalized_position_knn10_spearman | 0.231438 | 0.220393 | 0.176085 |
| raw_index_partial_length_spearman | 0.198102 | 0.185937 | 0.145681 |
| raw_index_within_16aa_length_bin_knn10_spearman | 0.508823 | 0.480882 | 0.462676 |
| exact_index_at_10_cross_protein | 0.014900 | 0.015067 | 0.014780 |
| nearest_neighbour_exact_index_all_neighbours | 0.056867 | 0.058533 | 0.062000 |
| normalized_position_quintile_purity_at_10_cross_protein | 0.267853 | 0.263673 | 0.249927 |
| same_protein_fraction_at_10 | 0.054780 | 0.067260 | 0.043127 |

## Provenance

- high: `/home/la-proteina-canonical-lejepa/store/canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e3_cosine_30k_rows96_warmup1200/checkpoints/step-00030000.ckpt`
  SHA256: `9d14a637379611738a531896739b30fef6a488ceed4a0ea76d0858c4f043f9c9`
  LR 0.00025 → 0.001 at 1200 → 1e-05 at 30000.
- middle: `/home/msc-thesis/store/canonical_lejepa_crop_fixed96_protein_perview_b96_indlocal_start1e4_peak4e4_end1e5_warmup1200_30k/checkpoints/step-00030000.ckpt`
  SHA256: `0c49a0bf0f97f3d4a6a06d6cc18ec51f912e4b7c834fca695c642de95cc3f085`
  LR 0.0001 → 0.0004 at 1200 → 1e-05 at 30000.
- low: `/home/la-proteina-canonical-lejepa/store/canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e4_cosine_30k_rows96_warmup1200/checkpoints/step-00030000.ckpt`
  SHA256: `ae612a64575eb5539f6b86e3a4279b57a1c165d424eb9a21834836a7a85396b4`
  LR 2.5e-05 → 0.0001 at 1200 → 1e-05 at 30000.

## Training health

Sampled W&B history, with recent-window medians from steps 28,000–30,000. Sampling cannot rule out every brief event. Separate view losses are not available in the existing aggregate logs.

| Metric | High | Middle | Low |
|---|---:|---:|---:|
| train/effective_rank | 7.243256 | 7.235701 | 7.233284 |
| train/embedding_std | 0.905697 | 0.923138 | 0.928512 |
| train/gradient_norm | 0.102616 | 0.144290 | 0.270916 |
| train/inv_loss | 0.067948 | 0.053035 | 0.052166 |
| train/loss | 0.104059 | 0.084858 | 0.083657 |
| train/sigreg_loss | 1.757812 | 1.523438 | 1.554688 |

## Earlier checkpoints

The following table adds the corresponding High/Low 10k and 16,667 point estimates. No temporal bootstrap or three-way bootstrap was run in this cheap diagnostic pass; differences are descriptive.

| Metric | High 10k | High 16,667 | High 30k | Low 10k | Low 16,667 | Low 30k |
|---|---:|---:|---:|---:|---:|---:|
| residue_effective_rank | 7.790703 | 7.882698 | 7.927478 | 7.902035 | 7.852815 | 7.839918 |
| residue_embedding_std | 0.997543 | 0.966386 | 0.970976 | 0.986585 | 0.973019 | 1.002558 |
| latent_dimension_std_min | 0.948756 | 0.920031 | 0.928030 | 0.939464 | 0.921665 | 0.928417 |
| latent_dimension_std_max | 1.061251 | 1.027791 | 1.045899 | 1.072395 | 1.034720 | 1.046484 |
| latent_isotropy_ratio | 1.118571 | 1.117127 | 1.127009 | 1.141496 | 1.122664 | 1.127171 |
| aa_purity_at_10_cross_protein | 0.075933 | 0.076647 | 0.077573 | 0.087760 | 0.082647 | 0.079980 |
| ss_purity_at_10_cross_protein | 0.523440 | 0.523287 | 0.516000 | 0.524347 | 0.520913 | 0.515393 |
| rsa_knn10_spearman | 0.166418 | 0.150302 | 0.149632 | 0.187318 | 0.178665 | 0.171107 |
| log_chain_length_knn10_spearman | 0.312545 | 0.374758 | 0.412651 | 0.330085 | 0.343107 | 0.341654 |
| raw_index_knn10_spearman | 0.234552 | 0.259736 | 0.275511 | 0.198123 | 0.212277 | 0.212660 |
| normalized_position_knn10_spearman | 0.212787 | 0.238259 | 0.231438 | 0.169735 | 0.186884 | 0.176085 |
| raw_index_partial_length_spearman | 0.181191 | 0.195641 | 0.198102 | 0.137322 | 0.150174 | 0.145681 |
| raw_index_within_16aa_length_bin_knn10_spearman | 0.486851 | 0.507062 | 0.508823 | 0.471137 | 0.473226 | 0.462676 |
| exact_index_at_10_cross_protein | 0.014933 | 0.015013 | 0.014900 | 0.015447 | 0.015487 | 0.014780 |
| nearest_neighbour_exact_index_all_neighbours | 0.057000 | 0.058933 | 0.056867 | 0.064867 | 0.063067 | 0.062000 |
| normalized_position_quintile_purity_at_10_cross_protein | 0.263093 | 0.269273 | 0.267853 | 0.254087 | 0.255247 | 0.249927 |
| same_protein_fraction_at_10 | 0.034360 | 0.044593 | 0.054780 | 0.034213 | 0.040740 | 0.043127 |

## Interpretation

At 30k, Low LR has the highest AA purity and RSA correlation, alongside lower chain-length correlation than High or Middle. High and Low have nearly equal SS purity; Middle is lower. All three residue effective ranks remain near 8. Low is the strongest point-estimate candidate under the stated biology/shortcut criteria, but this three-way 30k comparison has not been bootstrapped. Lower training loss alone is not used to select an LR.

Compared with Low at 10k, Low at 30k has lower AA/SS/RSA point estimates; additional training did not improve those particular held-out scores. High at 30k has higher AA but lower SS/RSA and higher chain-length correlation than its 10k checkpoint. These are descriptive temporal comparisons, not statistically tested changes.
