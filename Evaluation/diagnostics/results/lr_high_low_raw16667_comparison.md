# High vs Low LR: raw encoders at step 16,667

The available evidence favors Low LR: its AA and RSA advantages survive paired protein bootstrapping, while broad chain-length and position shortcuts are lower. SS is inconclusive. Low still has higher nearest-neighbour exact-index accuracy, so the shortcut tradeoff is not uniformly favorable. Neither representation is collapsed.

## Frozen protocol

5,224 held-out proteins; 748,402-residue pool; the same ordered 15,000 physical residues for both models. Raw encoder, no EMA. Sample SHA256: `6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8`. Geometry/bootstrap and t-SNE reused the extracted caches. Extraction used GPU 0 sequentially with a 10% memory cap; subsequent comparisons and t-SNE used CPU.

## Representation metrics

| Metric | High 10k | Low 10k | High 16,667 | Low 16,667 |
|---|---:|---:|---:|---:|
| residue_effective_rank | 7.79070 | 7.90203 | 7.88270 | 7.85282 |
| residue_embedding_std | 0.99754 | 0.98658 | 0.96639 | 0.97302 |
| latent_dimension_std_min | 0.94876 | 0.93946 | 0.92003 | 0.92166 |
| latent_dimension_std_max | 1.06125 | 1.07240 | 1.02779 | 1.03472 |
| latent_isotropy_ratio | 1.11857 | 1.14150 | 1.11713 | 1.12266 |
| aa_purity_at_10_cross_protein | 0.07593 | 0.08776 | 0.07665 | 0.08265 |
| ss_purity_at_10_cross_protein | 0.52344 | 0.52435 | 0.52329 | 0.52091 |
| rsa_knn10_spearman | 0.16642 | 0.18732 | 0.15030 | 0.17867 |
| log_chain_length_knn10_spearman | 0.31254 | 0.33009 | 0.37476 | 0.34311 |
| raw_index_knn10_spearman | 0.23455 | 0.19812 | 0.25974 | 0.21228 |
| normalized_position_knn10_spearman | 0.21279 | 0.16974 | 0.23826 | 0.18688 |
| raw_index_partial_length_spearman | 0.18119 | 0.13732 | 0.19564 | 0.15017 |
| raw_index_within_16aa_length_bin_knn10_spearman | 0.48685 | 0.47114 | 0.50706 | 0.47323 |
| exact_index_at_10_cross_protein | 0.01493 | 0.01545 | 0.01501 | 0.01549 |
| nearest_neighbour_exact_index_all_neighbours | 0.05700 | 0.06487 | 0.05893 | 0.06307 |
| normalized_position_quintile_purity_at_10_cross_protein | 0.26309 | 0.25409 | 0.26927 | 0.25525 |
| same_protein_fraction_at_10 | 0.03436 | 0.03421 | 0.04459 | 0.04074 |

## Paired protein bootstrap

5,000 replicates, seed 42; delta = Low − High. Query proteins resampled together, fixed neighbour graphs, Spearman ranks and partial residualization recomputed in each replicate. Intervals are conditional on the fixed 15,000-residue neighbour pool and unadjusted for multiple comparisons.

| Metric | High | Low | Delta | Bootstrap mean | 95% interval | Fraction > 0 | Excludes zero |
|---|---:|---:|---:|---:|---|---:|---|
| aa_purity_at_10_cross_protein | 0.07665 | 0.08265 | +0.00600 | +0.00600 | [+0.00406, +0.00789] | 1.0000 | True |
| ss_purity_at_10_cross_protein | 0.52329 | 0.52091 | -0.00237 | -0.00238 | [-0.00660, +0.00187] | 0.1288 | False |
| exact_index_at_10_cross_protein | 0.01501 | 0.01549 | +0.00047 | +0.00047 | [-0.00013, +0.00107] | 0.9378 | False |
| nearest_neighbour_exact_index_all_neighbours | 0.05893 | 0.06307 | +0.00413 | +0.00413 | [+0.00141, +0.00692] | 0.9986 | True |
| normalized_position_quintile_purity_at_10_cross_protein | 0.26927 | 0.25525 | -0.01403 | -0.01399 | [-0.01771, -0.01030] | 0.0000 | True |
| same_protein_fraction_at_10 | 0.04459 | 0.04074 | -0.00385 | -0.00386 | [-0.00555, -0.00218] | 0.0000 | True |
| rsa_knn10_spearman | 0.15030 | 0.17867 | +0.02836 | +0.02817 | [+0.00761, +0.04884] | 0.9974 | True |
| log_chain_length_knn10_spearman | 0.37476 | 0.34311 | -0.03165 | -0.03170 | [-0.04984, -0.01383] | 0.0002 | True |
| raw_index_knn10_spearman | 0.25974 | 0.21228 | -0.04746 | -0.04733 | [-0.06629, -0.02807] | 0.0000 | True |
| normalized_position_knn10_spearman | 0.23826 | 0.18688 | -0.05138 | -0.05121 | [-0.07144, -0.03090] | 0.0000 | True |
| raw_index_within_16aa_length_bin_knn10_spearman | 0.50706 | 0.47323 | -0.03384 | -0.03399 | [-0.04521, -0.02285] | 0.0000 | True |
| raw_index_partial_length_spearman | 0.19564 | 0.15017 | -0.04547 | -0.04536 | [-0.06501, -0.02548] | 0.0000 | True |

## Geometry

Linear CKA: **0.62964**. Orthogonal Procrustes similarity/aligned cosine: **0.75957**; normalized residual: **0.69344**. These are meaningfully different representations, not merely the same geometry with a rotated coordinate system.

| Distribution | Mean | Median | Std | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_cosine | 0.31065 | 0.35136 | 0.29170 | -0.23755 | 0.14105 | 0.52062 | 0.71380 |
| cross_protein_knn_overlap_at10 | 0.11401 | 0.10000 | 0.13638 | 0.00000 | 0.00000 | 0.20000 | 0.40000 |
| cross_protein_knn_jaccard_at10 | 0.06676 | 0.05263 | 0.08790 | 0.00000 | 0.00000 | 0.11111 | 0.25000 |

All-neighbour metrics preserve the frozen protocol: discard the first kNN result. Exact ties can retain the query itself; counts are reported. Cross-protein graphs strictly exclude all same-protein rows.

## Training health through 16,667

Values below are medians over steps 15,667–16,667. No nonfinite values were found in the seven scanned training metrics through step 16,667. These are observed logged metrics, not a proof of absence of every instability. Individual-view MSE/SIGReg are not separately logged.

| Metric | High | Low |
|---|---:|---:|
| train/inv_loss | 0.084325 | 0.068248 |
| train/sigreg_loss | 2.062500 | 1.750000 |
| train/loss | 0.126067 | 0.103055 |
| train/effective_rank | 7.199692 | 7.239766 |
| train/gradient_norm | 0.099356 | 0.267209 |
| train/embedding_std | 0.885810 | 0.909451 |
| train/global_local_cosine | 0.910468 | 0.927730 |

## Checkpoint provenance

- High: `canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e3_cosine_30k_rows96_warmup1200`; step 16667; checkpoint optimizer LR `0.00044751613053888874`.
  Source: `/home/la-proteina-canonical-lejepa/store/canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e3_cosine_30k_rows96_warmup1200/checkpoints/step-00016667.ckpt`
  SHA256: `1e6b7ecf64124072285af838c011f1ecd3bd6ddb0fb7167ace5fd86cff80a081`
  Schedule: `{"end_lr": 1e-05, "interval": "step", "peak_step": 1200, "start_lr": 0.00025, "total_steps": 30000, "type": "LinearWarmupCosineAnnealing"}`, peak LR `0.001`.
- Low: `canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e4_cosine_30k_rows96_warmup1200`; step 16667; checkpoint optimizer LR `4.9774193685360605e-05`.
  Source: `/home/la-proteina-canonical-lejepa/store/canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e4_cosine_30k_rows96_warmup1200/checkpoints/step-00016667.ckpt`
  SHA256: `ed5a28dd235f926ab76c9fefef95fe5dfcb7299c50ce70e15e8757514cb36c73`
  Schedule: `{"end_lr": 1e-05, "interval": "step", "peak_step": 1200, "start_lr": 2.5e-05, "total_steps": 30000, "type": "LinearWarmupCosineAnnealing"}`, peak LR `0.0001`.

## t-SNE figures

AA colors remain mixed in both maps: Low’s numerical advantage is not a dramatic visual separation. Both maps show clear secondary-structure organization. Position and chain length form local colored patches in both; neither is free of these patterns. Independent maps have arbitrary orientation and position. The joint Procrustes map is exploratory and not a performance metric.

- [Biology](../../latent_viz/figures/lr_high_low_raw16667_tsne_biology.png)
- [Position and length](../../latent_viz/figures/lr_high_low_raw16667_tsne_shortcuts.png)
- [Joint Procrustes](../../latent_viz/figures/lr_high_low_raw16667_tsne_joint_procrustes.png)

Coordinate caches and the JSON manifest are in `Evaluation/latent_viz/results/`. All plots use the same frozen 15,000 residues, cosine distance, perplexity 30, seed 42 and 1,000 iterations; each projection is fitted once and reused for recoloring.

## Interpretation

Low is the better-supported LR choice under the requested criteria at this checkpoint: AA and RSA improve relative to High without higher broad chain-length/position scores. The SS interval includes zero, and Low’s exact-index nearest-neighbour accuracy is higher. Both effective ranks remain near 8 with healthy isotropy. These diagnostics support a provisional choice, not a claim of better downstream performance.
