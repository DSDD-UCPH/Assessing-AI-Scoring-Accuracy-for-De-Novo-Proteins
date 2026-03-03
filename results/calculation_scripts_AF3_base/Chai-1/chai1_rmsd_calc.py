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
chai1_root = "/mnt/dsdd_share/Elliott/chai_run1"

parser = MMCIFParser(QUIET=True)

results = []

# --- Chai-1 lookup ---
chai1_lookup = {}

for folder in os.listdir(chai1_root):
    full_path = os.path.join(chai1_root, folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(folder)
    chai1_lookup[normalized] = folder

# --- AF3 lookup ---
af3_lookup = {}

for af3_folder in os.listdir(af3_root):
    full_path = os.path.join(af3_root, af3_folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(af3_folder)
    af3_lookup[normalized] = af3_folder

csv_path = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AdaptyvBio/contest_pivot_table_with_units.csv"
df = pd.read_csv(csv_path)

pairs = []

for _, row in df.iterrows():

    csv_id = normalize_for_fuzzy_match(str(row["id"]))
    csv_name = normalize_for_fuzzy_match(str(row["name"]))

    chai1_folder = chai1_lookup.get(csv_id)
    af3_folder = af3_lookup.get(csv_name)

    if chai1_folder and af3_folder:
        pairs.append((chai1_folder, af3_folder))
    else:
        print(f"Could not match: id={csv_id}, name={csv_name}")

for chai1_folder, af3_folder in pairs:
    chai1_folder_path = os.path.join(chai1_root, chai1_folder)
    af3_folder_path = os.path.join(af3_root, af3_folder)

    # Get AF3 CIF
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]

    if not af3_cifs:
        print(f"No CIF in AF3 folder {af3_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])

    # Get CHAI-1 CIF

    npz_files = [f for f in os.listdir(chai1_folder_path) if f.lower().endswith(".npz") and "scores.model_idx_" in f]

    if not npz_files:
        print(f"No scored npz in CHAI-1 folder {chai1_folder}")
        continue

    scores = {}

    for npz_file in npz_files:
        npz_path = os.path.join(chai1_folder_path, npz_file)
        data = np.load(npz_path, allow_pickle=True)
        if "aggregate_score" not in data:
            print(f"No aggregate_score in {npz_file}")
            continue

        score = data["aggregate_score"].item()

        model_idx = int(npz_file.split("_")[-1].split(".")[0])

        scores[model_idx] = score

    if not scores:
        print(f"No valid aggregate_score found in {chai1_folder_path}")
    else:
        best_model_idx = max(scores, key=scores.get)
        print(f"Best model: {best_model_idx} with score {scores[best_model_idx]}")

        chai1_cifs = [f for f in os.listdir(chai1_folder_path) if f.lower().endswith(".cif") and f"pred.model_idx_{best_model_idx}" in f]

        if not chai1_cifs:
            print(f"No matching CIF for best model {best_model_idx} in {chai1_folder_path}")
        else:
            best_cif_path = os.path.join(chai1_folder_path, chai1_cifs[0])
            print(f"Selected CIF file: {best_cif_path}")
        
        print(f"Processing {af3_folder}")

    try:
        # Load reference + original mobile structure
        structure_ref = parser.get_structure("ref", af3_cif_path)

        # -------------------------
        # RMSD when aligned on binder
        # -------------------------
        structure_mob = parser.get_structure("mobA", best_cif_path)
        binder_rmsd_on_binder = rmsd_of_binder_after_alignment(
            structure_ref, structure_mob, align_chain="A"
        )

        # -------------------------
        # RMSD when aligned on target
        # -------------------------
        structure_mob = parser.get_structure("mobB", best_cif_path)
        binder_rmsd_on_target = rmsd_of_binder_after_alignment(
            structure_ref, structure_mob, align_chain="B"
        )

        complex_name = af3_folder

        results.append([
            complex_name,
            binder_rmsd_on_binder,
            binder_rmsd_on_target,
        ])

    except Exception as e:
        print(f"Error processing {complex_name}: {e}")
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