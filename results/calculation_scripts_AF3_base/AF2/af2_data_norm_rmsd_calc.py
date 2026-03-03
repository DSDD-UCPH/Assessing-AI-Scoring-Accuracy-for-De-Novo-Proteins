"""
A script that does several things:
1. Directory setup.
2. Extracts and finds the corresponding .pdb file.
3. Matches the suffix of this .pdb file with the names of the AF3 predictions.
4. Calculates the RMSD between the best AF2 model and the corresponding AF3 predictions.
5. Generates a CSV file with the calculated values for each and every complex.
"""

import os
import json
import numpy as np
import glob
import pandas as pd
import csv
from Bio.PDB import MMCIFParser, Superimposer, PDBParser
import re

### Part 0. Functions. ###

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


def get_atom_lists_for_chain(struct_ref, struct_mob, ref_chain, mob_chain):
    """Return lists of CA atoms with matching residue numbers."""
    ca_ref = get_ca_atoms_by_resid(struct_ref, ref_chain)
    ca_mob = get_ca_atoms_by_resid(struct_mob, mob_chain)

    common = sorted(set(ca_ref.keys()) & set(ca_mob.keys()))
    if not common:
        raise ValueError(f"No overlapping residues for chains {ref_chain} / {mob_chain}")

    atoms_ref = [ca_ref[r] for r in common]
    atoms_mob = [ca_mob[r] for r in common]
    return atoms_ref, atoms_mob


def compute_rmsd_no_align(atoms1, atoms2):
    """Compute RMSD without performing another alignment."""
    diffs = np.array([a.coord - b.coord for a, b in zip(atoms1, atoms2)])
    return np.sqrt((diffs * diffs).sum() / len(diffs))


def rmsd_of_binder_after_alignment(struct_ref, struct_mob, ref_align_chain, mob_align_chain, ref_binder_chain=None, mob_binder_chain=None):
    """
    1. Align mobile structure onto reference using specified chains.
    2. Compute RMSD of binder chains without re-aligning.

    Parameters
    ----------
    struct_ref : Bio.PDB.Structure
        Reference structure (AF3)
    struct_mob : Bio.PDB.Structure
        Mobile structure (AF2)
    ref_align_chain : str
        Chain in reference used for alignment
    mob_align_chain : str
        Chain in mobile used for alignment
    ref_binder_chain : str, optional
        Chain in reference for RMSD calculation (defaults to align_chain)
    mob_binder_chain : str, optional
        Chain in mobile for RMSD calculation (defaults to align_chain)
    """
    # Default binder chains to align chains if not specified
    if ref_binder_chain is None:
        ref_binder_chain = ref_align_chain
    if mob_binder_chain is None:
        mob_binder_chain = mob_align_chain

    # 1️⃣ Get atoms for alignment
    ref_align_atoms, mob_align_atoms = get_atom_lists_for_chain(
        struct_ref, struct_mob, ref_chain=ref_align_chain, mob_chain=mob_align_chain
    )

    sup = Superimposer()
    sup.set_atoms(ref_align_atoms, mob_align_atoms)
    sup.apply(struct_mob.get_atoms())  # transform entire mobile structure

    # 2️⃣ Get atoms for RMSD calculation
    ref_binder_atoms, mob_binder_atoms = get_atom_lists_for_chain(
        struct_ref, struct_mob, ref_chain=ref_binder_chain, mob_chain=mob_binder_chain
    )

    rmsd = compute_rmsd_no_align(ref_binder_atoms, mob_binder_atoms)

    return rmsd


def normalize_string(s):
    """
    - Lowercase
    - Replace any character NOT in [A-Za-z0-9._] with '_'
    """
    s = s.lower()
    s = re.sub(r'[^a-z0-9._]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s

def normalize_for_fuzzy_match(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9.]', '', s)
    return s

def strip_egfr_prefix(folder_name):
    if folder_name.lower().startswith("egfr_"):
        return folder_name[5:]
    return folder_name


### Part 1. Directory setup. ###

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
af2_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF2/EGFR_AF2_panel"

pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

results = []

# --- AF2 lookup ---
af2_lookup = {}

for folder in os.listdir(af2_root):
    full_path = os.path.join(af2_root, folder)
    if not os.path.isdir(full_path):
        continue

    stripped = strip_egfr_prefix(folder)
    normalized = normalize_for_fuzzy_match(stripped)
    af2_lookup[normalized] = folder

# --- AF3 lookup ---
af3_lookup = {}

for folder in os.listdir(af3_root):
    full_path = os.path.join(af3_root, folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(folder)
    af3_lookup[normalized] = folder

csv_path = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AdaptyvBio/contest_pivot_table_with_units.csv"
df = pd.read_csv(csv_path)

pairs = []

for _, row in df.iterrows():

    csv_id = normalize_for_fuzzy_match(str(row["id"]))
    csv_name = normalize_for_fuzzy_match(str(row["name"]))

    af2_folder = af2_lookup.get(csv_id)
    af3_folder = af3_lookup.get(csv_name)

    if af2_folder and af3_folder:
        pairs.append((af2_folder, af3_folder))
    else:
        print(f"Could not match: id={csv_id}, name={csv_name}")

# --- Processing the complexes ---

for af2_folder, af3_folder in pairs:
    af2_folder_path = os.path.join(af2_root, af2_folder)
    af3_folder_path = os.path.join(af3_root, af3_folder)

    # Get AF3 CIF
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]

    if not af3_cifs:
        print(f"No CIF in AF3 folder {af3_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])

    # Get AF2 relaxed PDB

    pattern = os.path.join(af2_folder_path, "relaxed_model_*")
    af2_pdb_files = glob.glob(pattern)

    if not af2_pdb_files:
        print(f"No relaxed PDB in AF2 folder {af2_folder}")
        continue

    af2_pdb_path = af2_pdb_files[0]

    print(f"Processing AF2 folder {af2_folder} and AF3 folder {af3_folder}")

    try:
        # load reference (AF3)
        structure_ref = cif_parser.get_structure("ref", af3_cif_path)

        # RMSD when aligned on binder
        structure_mob = pdb_parser.get_structure("mobA", af2_pdb_path)

        binder_rmsd_on_binder = rmsd_of_binder_after_alignment(structure_ref, structure_mob,
                                                               ref_align_chain="A", # AF3 binder
                                                               mob_align_chain="B" # AF2 binder
                                                               )
        
        structure_mob = pdb_parser.get_structure("mobB", af2_pdb_path)
        
        binder_rmsd_on_target = rmsd_of_binder_after_alignment(structure_ref, structure_mob,
                                                                ref_align_chain="B", # AF3 target
                                                                mob_align_chain="A" # AF2 target
                                                                )
        
        complex_name = af3_folder

        results.append([
            complex_name,
            binder_rmsd_on_binder,
            binder_rmsd_on_target
        ])

    except Exception as e:
        print(f"Error processing {af2_folder}: {e}")
        continue

# --- Save results to CSV ---

with open("binder_rmsd_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Complex",
        "binder_rmsd_aligned_on_binder",
        "binder_rmsd_aligned_on_target"
    ])
    writer.writerows(results)

print("Saved binder_rmsd_results.csv")