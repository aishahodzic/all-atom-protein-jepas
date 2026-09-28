# Exact step-300000 JEPA decoder; predicted-sequence direct and postprocessed outputs.
reinitialize
load "./4etr_A_native.pdb", native
load "./4etr_A_reconstructed_raw.pdb", reconstruction
load "./4etr_A_reconstructed_postprocessed.pdb", reconstruction_postprocessed

dss native
dss reconstruction
dss reconstruction_postprocessed
bg_color white
set cartoon_side_chain_helper, 1
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1
set cartoon_discrete_colors, 0
set ray_trace_mode, 1
set antialias, 2
set ray_shadows, 1
set ray_opaque_background, 0
set ambient, 0.1
set ray_trace_gain, 0.05
set ray_blend_colors, 1
set ray_trace_color, black
set stick_radius, 0.15

set_color reconstruction_cyan, [0.00, 1.00, 1.00]
set_color postprocessed_orange, [0.95, 0.55, 0.20]

hide everything
show cartoon, native or reconstruction or reconstruction_postprocessed
show sticks, native or reconstruction or reconstruction_postprocessed
color gray70, native
set cartoon_color, gray70, native
set cartoon_highlight_color, gray70, native
color reconstruction_cyan, reconstruction
set cartoon_color, reconstruction_cyan, reconstruction
set cartoon_highlight_color, reconstruction_cyan, reconstruction
color postprocessed_orange, reconstruction_postprocessed
set cartoon_color, postprocessed_orange, reconstruction_postprocessed
set cartoon_highlight_color, postprocessed_orange, reconstruction_postprocessed

align reconstruction and name CA, native and name CA
align reconstruction_postprocessed and name CA, native and name CA
set cartoon_transparency, 0.00, native or reconstruction or reconstruction_postprocessed
set stick_transparency, 0.00, native or reconstruction or reconstruction_postprocessed

disable reconstruction_postprocessed
orient native or reconstruction
zoom native or reconstruction
scene direct, store

disable reconstruction
enable reconstruction_postprocessed
orient native or reconstruction_postprocessed
zoom native or reconstruction_postprocessed
scene postprocessed, store

scene direct, recall
