reinitialize
load "./1kfm_A_reconstructed_idealized.pdb", reconstruction_projected

# Remove any bonds PyMOL inferred inside residue 49, then install only the
# chemically valid ILE topology.  This makes the topology check independent
# of PyMOL's distance-based bond guessing.
unbond (reconstruction_projected and chain A and resi 49), (reconstruction_projected and chain A and resi 49)
bond (reconstruction_projected and chain A and resi 49 and name N), (reconstruction_projected and chain A and resi 49 and name CA)
bond (reconstruction_projected and chain A and resi 49 and name CA), (reconstruction_projected and chain A and resi 49 and name C)
bond (reconstruction_projected and chain A and resi 49 and name C), (reconstruction_projected and chain A and resi 49 and name O)
bond (reconstruction_projected and chain A and resi 49 and name CA), (reconstruction_projected and chain A and resi 49 and name CB)
bond (reconstruction_projected and chain A and resi 49 and name CB), (reconstruction_projected and chain A and resi 49 and name CG1)
bond (reconstruction_projected and chain A and resi 49 and name CB), (reconstruction_projected and chain A and resi 49 and name CG2)
bond (reconstruction_projected and chain A and resi 49 and name CG1), (reconstruction_projected and chain A and resi 49 and name CD1)

hide everything
show sticks, reconstruction_projected and chain A and resi 49
show spheres, reconstruction_projected and chain A and resi 49
set sphere_scale, 0.25
set stick_radius, 0.12
label reconstruction_projected and chain A and resi 49, name
color tv_orange, reconstruction_projected and chain A and resi 49 and elem C
color marine, reconstruction_projected and chain A and resi 49 and elem N
color red, reconstruction_projected and chain A and resi 49 and elem O
bg_color white
orient reconstruction_projected and chain A and resi 49
zoom reconstruction_projected and chain A and resi 49, 5
