JEPA rand-70 target-decoder reconstruction for held-out protein 9e02_F.

On macOS, put every file from this bundle in one directory. Then either:

1. Start PyMOL from a terminal in that directory:
   pymol 9e02_F_view.pml

2. In an already-open PyMOL window:
   cd /absolute/path/to/the/unpacked/directory
   @9e02_F_view.pml

Objects loaded by the script:
- native: experimental structure, gray
- reconstruction_true_sequence: decoded coordinates with true residue names, cyan
- reconstruction_predicted_sequence: decoded coordinates and decoded residue names,
  orange and initially disabled; run `enable reconstruction_predicted_sequence`
  to display it.

The decoder receives the native C-alpha trace as an explicit anchor. Exact C-alpha
agreement is therefore expected; side-chain/non-C-alpha reconstruction is the
meaningful comparison.
