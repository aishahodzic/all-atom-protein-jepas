# Raw-10k High/Low t-SNE figures

All figures were generated on CPU from existing raw latent caches. No encoder extraction was run. The frozen sample contains 15,000 identical ordered physical residues; sample SHA256 is 6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8.

The individual High and Low fits each ran once for 1,000 iterations using project_tsne(), then were reused for all six colorings. The third fit is an exploratory 30,000-point joint map after centering both models and orthogonally aligning Low to High. Coordinate caches contain inputs' SHA256, physical-sample SHA256, model names, raw source, global step 10000, and all t-SNE parameters.

Visual inspection: AA labels remain heavily mixed in both maps; Low's measured AA advantage is not an obvious dramatic separation into cleaner AA islands. Both show clearer secondary-structure regions. Some position/length-colored islands have local coherence, but neither shows an obvious single global monotonic gradient. The joint map includes both mixed and model-enriched regions. These observations do not establish relative representation quality; use the original-space metrics and paired protein bootstrap for that comparison.

Independent map positions, orientations, and scales are arbitrary. Legends and property color limits are shared across High/Low rows. RSA uses [0,1], with invalid residues drawn light grey; this particular frozen sample has zero invalid RSA residues.

Recoloring is reproducible by rerunning Evaluation/latent_viz/plot_lr_raw10k_tsne.py: matching coordinate caches are reused after verifying input and parameter hashes. No refit is needed.
