reinitialize
load "./5xr7_A_native.pdb", native
load "./5xr7_A_reconstructed.pdb", reconstruction
dss native
dss reconstruction
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
hide everything
show cartoon, native or reconstruction
show sticks, reconstruction and not backbone
color c_color, reconstruction and elem C and not name CA
color o_color, reconstruction and elem O
color n_color, reconstruction and elem N
color s_color, reconstruction and elem S
color f_color, reconstruction and elem F
color cl_color, reconstruction and elem Cl
set cartoon_color, target_cartoon_color, native
set cartoon_color, helix_color, reconstruction and ss H
set cartoon_color, beta_color, reconstruction and ss S
set cartoon_color, loop_color, reconstruction and not (ss H+S)
align reconstruction and name CA, native and name CA
set cartoon_transparency, 0.55, native
set cartoon_transparency, 0.00, reconstruction
create target_surface, native
show surface, target_surface
color grey90, target_surface
set transparency, 0.75, target_surface
disable target_surface
distance hbonds, reconstruction and donors, reconstruction and acceptors, 3.2, 2
color hbond_color, hbonds
set dash_width, 7.0, hbonds
set dash_gap, 0.4, hbonds
hide labels, hbonds
disable hbonds
orient native or reconstruction
zoom native or reconstruction
