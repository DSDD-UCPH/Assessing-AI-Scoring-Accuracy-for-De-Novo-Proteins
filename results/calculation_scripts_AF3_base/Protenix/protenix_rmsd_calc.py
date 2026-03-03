import os
import csv
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import glob


def get_ca_atoms_by_resid(structure, chain_id):
    """Return CA atoms dict {resid -> atom} for a given chain."""
    model = structure[0]
    chain = model[chain_id]

    ca_dict = {}
    for residue in chain:
        if residue.id[0] != " ":  # skip heteroatoms
            continue
        resid = residue.id[1]
        if "CA" in residue:
            ca_dict[resid] = residue["CA"]
    return ca_dict


def get_atom_lists_for_chain(struct1, struct2, chain_id):
    """Return lists of CA atoms with matching residue numbers."""
    ca1 = get_ca_atoms_by_resid(struct1, chain_id)
    ca2 = get_ca_atoms_by_resid(struct2, chain_id)

    common = sorted(set(ca1.keys()) & set(ca2.keys()))
    if not common:
        raise ValueError(f"No overlapping residues for chain {chain_id}")

    atoms1 = [ca1[r] for r in common]
    atoms2 = [ca2[r] for r in common]
    return atoms1, atoms2


def compute_rmsd_no_align(atoms1, atoms2):
    """Compute RMSD without performing another alignment."""
    diffs = np.array([a.coord - b.coord for a, b in zip(atoms1, atoms2)])
    return np.sqrt((diffs * diffs).sum() / len(diffs))


def rmsd_of_binder_after_alignment(struct_ref, struct_mob, align_chain, binder_chain="A"):
    """
    1. Align mobile structure onto reference using align_chain.
    2. Compute RMSD of binder_chain CA atoms (no second alignment).
    """

    # Global alignment using align_chain
    ref_align, mob_align = get_atom_lists_for_chain(struct_ref, struct_mob, align_chain)

    sup = Superimposer()
    sup.set_atoms(ref_align, mob_align)
    sup.apply(struct_mob.get_atoms())  # transform entire mobile structure

    # Get binder atoms and compute RMSD WITHOUT re-aligning
    ref_bind, mob_bind = get_atom_lists_for_chain(struct_ref, struct_mob, binder_chain)
    rmsd = compute_rmsd_no_align(ref_bind, mob_bind)

    return rmsd

def strip_egfr_suffix(folder_name):
    """Remove _EGFR suffix from folder name if present."""
    if folder_name.lower().endswith("_egfr"):
        return folder_name[:-5]  # Remove last 5 characters
    return folder_name

# -----------------------------------------------------------------------
#                      DIRECTORY SETUP (your original logic)
# -----------------------------------------------------------------------

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
protenix_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Protenix/Protenix_run_1"

parser = MMCIFParser(QUIET=True)

# Case-insensitive lookup for Protenix folders
protenix_folders = {
    strip_egfr_suffix(folder).lower(): folder
    for folder in os.listdir(protenix_root)
    if os.path.isdir(os.path.join(protenix_root, folder))
}

results = []

# -----------------------------------------------------------------------
# Processing all complexes
# -----------------------------------------------------------------------

for af3_folder in os.listdir(af3_root):

    af3_folder_path = os.path.join(af3_root, af3_folder)
    if not os.path.isdir(af3_folder_path):
        continue

    key = af3_folder.lower()

    if key not in protenix_folders:
        print(f"No Protenix folder match for {af3_folder}")
        continue

    protenix_folder = protenix_folders[key]
    protenix_folder_path = os.path.join(protenix_root, protenix_folder)

    # Get CIF filenames
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]

    pattern = os.path.join(
        protenix_folder_path,
        "seed_*",
        "predictions",
        "*sample_0*.cif"
        )

    protenix_cifs = glob.glob(pattern)

    if not af3_cifs:
        print(f"No CIF in {af3_folder}")
        continue
    if not protenix_cifs:
        print(f"No sample_0 CIF in {protenix_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])
    protenix_cif_path = os.path.join(protenix_folder_path, protenix_cifs[0])


    print(f"Processing {af3_folder}")

    try:
        # Load reference + original mobile structure
        structure_ref = parser.get_structure("ref", af3_cif_path)

        # -------------------------
        # RMSD when aligned on binder
        # -------------------------
        structure_mob = parser.get_structure("mobA", protenix_cif_path)
        binder_rmsd_on_binder = rmsd_of_binder_after_alignment(
            structure_ref, structure_mob, align_chain="A"
        )

        # -------------------------
        # RMSD when aligned on target
        # -------------------------
        structure_mob = parser.get_structure("mobB", protenix_cif_path)
        binder_rmsd_on_target = rmsd_of_binder_after_alignment(
            structure_ref, structure_mob, align_chain="B"
        )

        results.append([
            key,
            binder_rmsd_on_binder,
            binder_rmsd_on_target,
        ])

    except Exception as e:
        print(f"Error processing {key}: {e}")
        continue


# -----------------------------------------------------------------------
# Write CSV
# -----------------------------------------------------------------------

with open("binder_rmsd_results_fixed.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Complex",
        "binder_rmsd_aligned_on_binder",
        "binder_rmsd_aligned_on_target"
    ])
    writer.writerows(results)

print("Saved binder_rmsd_results_fixed.csv")