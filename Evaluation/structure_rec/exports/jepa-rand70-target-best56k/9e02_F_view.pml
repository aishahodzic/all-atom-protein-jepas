reinitialize
load "./9e02_F_native.pdb", native
load "./9e02_F_reconstructed_true_sequence.pdb", reconstruction_true_sequence
load "./9e02_F_reconstructed_predicted_sequence.pdb", reconstruction_predicted_sequence
hide everything
show cartoon, native or reconstruction_true_sequence
color gray70, native
color cyan, reconstruction_true_sequence
color tv_orange, reconstruction_predicted_sequence
disable reconstruction_predicted_sequence
align reconstruction_true_sequence and name CA, native and name CA
set cartoon_transparency, 0.35, native
set cartoon_transparency, 0.10, reconstruction_true_sequence
bg_color white
orient native or reconstruction_true_sequence
zoom native or reconstruction_true_sequence
