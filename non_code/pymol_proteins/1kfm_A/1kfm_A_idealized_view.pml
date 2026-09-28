reinitialize
load "./1kfm_A_native.pdb", native
load "./1kfm_A_reconstructed.pdb", reconstruction_raw
load "./1kfm_A_reconstructed_idealized.pdb", reconstruction_projected
dss native
dss reconstruction_raw
dss reconstruction_projected
bg_color white
hide everything
show cartoon, native or reconstruction_projected
show sticks, reconstruction_projected and not backbone
color gray70, native
color tv_orange, reconstruction_projected
color red, reconstruction_projected and elem O
color marine, reconstruction_projected and elem N
color yellow, reconstruction_projected and elem S
disable reconstruction_raw
align reconstruction_projected and name CA, native and name CA
set cartoon_transparency, 0.55, native
set cartoon_transparency, 0.00, reconstruction_projected
set stick_radius, 0.15
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1
set ray_trace_mode, 1
set antialias, 2
set ray_opaque_background, 0
orient native or reconstruction_projected
zoom native or reconstruction_projected
