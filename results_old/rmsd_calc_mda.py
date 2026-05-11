import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd

u_ref = mda.Universe("") # Path to pdb file
u_mob = mda.Universe("") # Path to pdb file

# Align on target (chain B)
align.alignto(u_mob, u_ref,
              select="segid B and backbone")

# Compute target RMSD
target_rmsd = rmsd(
    u_mob.select_atoms("segid B and backbone").positions,
    u_ref.select_atoms("segid B and backbone").positions
)

# Compute binder RMSD
binder_rmsd = rmsd(
    u_mob.select_atoms("segid A and backbone").positions,
    u_ref.select_atoms("segid A and backbone").positions
)

print("Target RMSD:", target_rmsd)
print("Binder RMSD:", binder_rmsd)

print("Segments:", u_ref.segments)
print("Segment IDs:", [seg.segid for seg in u_ref.segments])
print("Unique segids:", set(u_ref.atoms.segids))

print("Does chainIDs exist?", hasattr(u_ref.atoms, "chainIDs"))
if hasattr(u_ref.atoms, "chainIDs"):
    print("Unique chainIDs:", set(u_ref.atoms.chainIDs))
