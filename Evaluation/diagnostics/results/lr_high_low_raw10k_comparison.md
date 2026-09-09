# High vs Low LR: raw-10k geometry and paired protein bootstrap

CPU-only analysis of existing latent caches; no encoder reruns or added GPU processes. Every physical residue identity agrees across models, and all twelve original point estimates reproduce to 1e-9.

Frozen sample SHA256: 6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8
Frozen pool: 5224 proteins / 748402 residues. Sample: 15000 residues from 4738 represented query proteins.

## Geometry

| Distribution | Mean | Median | SD | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct row cosine | 0.363087 | 0.414545 | 0.298611 | -0.204272 | 0.195291 | 0.581421 | 0.757463 |
| Cross-protein overlap@10 | 0.087907 | 0.000000 | 0.121251 | 0.000000 | 0.000000 | 0.100000 | 0.300000 |
| Cross-protein Jaccard@10 | 0.050786 | 0.000000 | 0.076432 | 0.000000 | 0.000000 | 0.052632 | 0.176471 |

Linear CKA: 0.662594
Procrustes similarity: 0.778767
Aligned vectorized cosine: 0.778767
Normalized Frobenius residual: 0.665182

The models are not effectively geometrically identical. Orthogonal alignment explains part of the coordinate difference, but substantial global and local differences remain. The average shared neighbourhood contains only 0.879 of 10 neighbours, and median overlap is zero. CKA is invariant to orthogonal transformations and uniform scaling, not arbitrary invertible dimension mixing or dimension-wise rescaling.

## Paired protein bootstrap

5000 replicates, seed 42; delta = Low minus High. Draw 4738 represented proteins with replacement and concatenate all their sampled residue rows. Use identical draws for both models. Rank correlations and ranked residualization are recomputed inside every replicate. Neighbour identities and predictions are held fixed.

**Intervals are conditional on the fixed 15000-residue neighbour pool.** These are unadjusted 95% percentile intervals across twelve comparisons. The fraction above zero is an empirical bootstrap fraction, not a p-value or posterior probability.

| Metric | High | Low | Observed delta | Bootstrap mean delta | 95% interval | Fraction > 0 | Excludes zero |
|---|---:|---:|---:|---:|---|---:|---|
| aa_purity_at_10_cross_protein | 0.075933 | 0.087760 | +0.011827 | +0.011824 | [+0.009826, +0.013830] | 1.0000 | True |
| ss_purity_at_10_cross_protein | 0.523440 | 0.524347 | +0.000907 | +0.000879 | [-0.003438, +0.005255] | 0.6500 | False |
| exact_index_at_10_cross_protein | 0.014933 | 0.015447 | +0.000513 | +0.000509 | [-0.000054, +0.001093] | 0.9632 | False |
| nearest_neighbour_exact_index_all_neighbours | 0.057000 | 0.064867 | +0.007867 | +0.007852 | [+0.005157, +0.010589] | 1.0000 | True |
| normalized_position_quintile_purity_at_10_cross_protein | 0.263093 | 0.254087 | -0.009007 | -0.008996 | [-0.012440, -0.005469] | 0.0000 | True |
| same_protein_fraction_at_10 | 0.034360 | 0.034213 | -0.000147 | -0.000152 | [-0.001646, +0.001363] | 0.4190 | False |
| rsa_knn10_spearman | 0.166418 | 0.187318 | +0.020900 | +0.020766 | [+0.000944, +0.041093] | 0.9814 | True |
| log_chain_length_knn10_spearman | 0.312545 | 0.330085 | +0.017541 | +0.017367 | [-0.002035, +0.036340] | 0.9608 | False |
| raw_index_knn10_spearman | 0.234552 | 0.198123 | -0.036429 | -0.036787 | [-0.058087, -0.017394] | 0.0002 | True |
| normalized_position_knn10_spearman | 0.212787 | 0.169735 | -0.043052 | -0.043292 | [-0.063385, -0.022816] | 0.0000 | True |
| raw_index_within_16aa_length_bin_knn10_spearman | 0.486851 | 0.471137 | -0.015714 | -0.015780 | [-0.027295, -0.004256] | 0.0042 | True |
| raw_index_partial_length_spearman | 0.181191 | 0.137322 | -0.043869 | -0.044228 | [-0.065055, -0.024140] | 0.0000 | True |

## Interpretation

- Low’s AA advantage clearly survives the requested bootstrap. Its RSA advantage also has an interval above zero, but only narrowly; treat that as weaker evidence, particularly with multiple comparisons. SS remains inconclusive.
- Low has lower raw-index, normalized-position, length-controlled raw-index, within-length-bin raw-index, and position-quintile signals with intervals excluding zero.
- Low has HIGHER nearest-neighbour exact-index accuracy with an interval excluding zero. The exact-index purity, chain-length correlation, and same-protein fraction differences remain inconclusive.
- Evidence supports Low as the preferred raw-10k candidate for AA signal and reduced broad position leakage, with an explicit adverse exact-index-nearest-neighbour tradeoff. It is not a uniformly better representation or proof of downstream superiority. A matched later-checkpoint comparison can assess persistence.

## Frozen-protocol tie handling

To reproduce the original diagnostic values, all-neighbour graphs discard the first nearest-neighbour result. Exact-distance ties leave self among the ten neighbours for one query per model (1/15000). This is preserved and disclosed, not silently corrected. All cross-protein neighbours strictly exclude the entire source protein. The nearest-neighbour exact-index and same-protein metrics use the frozen all-protein graph; the exact-index purity uses the cross-protein graph.

Saved auxiliary query_arrays.npz contains physical row identities, neighbour graphs, targets, per-query predictions and contributions. bootstrap_deltas.npz contains all 5000 paired replicate deltas. Source and cache hashes are recorded in the JSON outputs.
