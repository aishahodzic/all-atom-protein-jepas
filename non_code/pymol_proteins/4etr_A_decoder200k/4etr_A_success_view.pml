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
set cartoon_highlight_color, grey90
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

set_color helix_color, [0.81, 0.59, 0.79]
set_color beta_color, [0.88, 0.70, 0.51]
set_color loop_color, [0.84, 0.85, 0.86]
set_color c_color, [0.45, 0.45, 0.45]
set_color o_color, [1.0, 0.7, 0.7]
set_color n_color, [0.7, 0.8, 1.0]
set_color s_color, [1.0, 0.95, 0.7]
set_color cl_color, [0.75, 0.90, 0.75]
set_color f_color, [0.7, 1.0, 0.9]
set_color hbond_color, [0.86, 0.08, 0.24]
set_color target_cartoon_color, [0.61, 0.59, 0.81]
set_color reconstruction_blue, [0.00, 1.00, 1.00]
set_color postprocessed_orange, [0.95, 0.55, 0.20]

hide everything
show cartoon, native or reconstruction or reconstruction_postprocessed
show sticks, native or reconstruction or reconstruction_postprocessed
color gray70, native
set cartoon_color, gray70, native
set cartoon_highlight_color, gray70, native
color reconstruction_blue, reconstruction
set cartoon_color, reconstruction_blue, reconstruction
set cartoon_highlight_color, reconstruction_blue, reconstruction
color postprocessed_orange, reconstruction_postprocessed
set cartoon_color, postprocessed_orange, reconstruction_postprocessed
set cartoon_highlight_color, postprocessed_orange, reconstruction_postprocessed

align reconstruction and name CA, native and name CA
align reconstruction_postprocessed and name CA, native and name CA
set cartoon_transparency, 0.65, native
set cartoon_transparency, 0.00, reconstruction or reconstruction_postprocessed
set stick_transparency, 0.55, native

create target_surface, native
show surface, target_surface
color grey90, target_surface
set transparency, 0.75, target_surface
disable target_surface

distance hbonds_direct, reconstruction and donors, reconstruction and acceptors, 3.2, 2
distance hbonds_postprocessed, reconstruction_postprocessed and donors, reconstruction_postprocessed and acceptors, 3.2, 2
color hbond_color, hbonds_direct or hbonds_postprocessed
set dash_width, 7.0, hbonds_direct or hbonds_postprocessed
set dash_gap, 0.4, hbonds_direct or hbonds_postprocessed
hide labels, hbonds_direct or hbonds_postprocessed
disable hbonds_direct
disable hbonds_postprocessed

orient native or reconstruction
zoom native or reconstruction

disable reconstruction_postprocessed
enable hbonds_direct
scene direct, store
disable reconstruction
disable hbonds_direct
enable reconstruction_postprocessed
enable hbonds_postprocessed
scene postprocessed, store
scene direct, recall
