JEPA rand-70 target-decoder reconstruction for held-out protein 5a13_G.

This is the longer mixed-secondary-structure example (245 residues). Keep all
files in one directory. In an already-open PyMOL window, run:

  cd /absolute/path/to/this/directory
  @5a13_G_view.pml

The script displays:
- native: experimental structure, translucent purple cartoon
- reconstruction_true_sequence: decoder coordinates with true residue names,
  secondary-structure-colored cartoon and element-colored side-chain sticks
- reconstruction_predicted_sequence: decoded coordinates and decoded residue
  names, orange and initially disabled

Useful toggles:
  disable native
  enable reconstruction_predicted_sequence
  enable target_surface
  enable hbonds

For a high-resolution transparent render:
  ray 2400, 1800
  png 5a13_G_rand70_reconstruction.png, dpi=300

The decoder receives the native C-alpha trace as an explicit anchor. Exact
C-alpha agreement is expected; side-chain/non-C-alpha reconstruction is the
meaningful comparison.
