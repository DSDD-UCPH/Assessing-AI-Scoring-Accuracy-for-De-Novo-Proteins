"""
A script that does several things:
1. Directory setup.
2. Extracts and matches the corresponding .cif files.
3. Matches the suffix of this .cif file with the names of the AF3 predictions.
4. Calculates the RMSD between the best Chai-1 model and the corresponding AF3 predictions.
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
from Bio.PDB.StructureBuilder import StructureBuilder
import gemmi
import tempfile

### Part 0. Functions. ###

class PatchedMMCIFParser(MMCIFParser):
    def _build_structure(self, structure_id):
        mmcif_dict = self._mmcif_dict

        # If occupancy column missing, create default one
        if "_atom_site.occupancy" not in mmcif_dict:
            n_atoms = len(mmcif_dict["_atom_site.Cartn_x"])
            mmcif_dict["_atom_site.occupancy"] = ["1.0"] * n_atoms

        super()._build_structure(structure_id)

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


### Part 1. Directory setup. ###

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
of3_root = "/mnt/dsdd_share/Elliott/of3_run1"

parser = PatchedMMCIFParser(QUIET=True)

results = []

# --- OF3 lookup ---
of3_lookup = {}

for folder in os.listdir(of3_root):
    full_path = os.path.join(of3_root, folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(folder)
    of3_lookup[normalized] = folder

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

    of3_folder = of3_lookup.get(csv_id)
    af3_folder = af3_lookup.get(csv_name)

    if of3_folder and af3_folder:
        pairs.append((of3_folder, af3_folder))
    else:
        print(f"Could not match: id={csv_id}, name={csv_name}")

for of3_folder, af3_folder in pairs:
    of3_folder_path = os.path.join(of3_root, of3_folder)
    af3_folder_path = os.path.join(af3_root, af3_folder)

    # Get AF3 CIF
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]

    if not af3_cifs:
        print(f"No CIF in AF3 folder {af3_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])

    # OF3 CIFs

    seed_folders = [os.path.join(of3_folder_path, d) for d in os.listdir(of3_folder_path) if os.path.isdir(os.path.join(of3_folder_path, d)) and d.startswith("seed_")]

    if not seed_folders:
        print(f"No seed folder in {of3_folder}")
        continue

    seed_path = seed_folders[0]

    json_files = [
        f for f in os.listdir(seed_path)
        if f.endswith(".json") and "confidences_aggregated" in f
    ]
    if not json_files:
        print(f"No confidence JSON files in {seed_path}")
        continue

    scores = {}

    for json_file in json_files:
        json_path = os.path.join(seed_path, json_file)
        with open(json_path, "r") as jf:
            data = json.load(jf)

        if "sample_ranking_score" not in data:
            print(f"No 'sample_ranking_score' in {json_file}")
            continue

        score = data["sample_ranking_score"]

        match = re.search(r"_sample_(\d+)_", json_file)
        if not match:
            print(f"Could not parse model index from {json_file}")
            continue

        model_idx = int(match.group(1))
        scores[model_idx] = score

    if not scores:
        print(f"No valid sample_ranking_score found in {seed_path}")
        continue

    best_model_idx = max(scores, key=scores.get)
    print(f"Best model: {best_model_idx} with score {scores[best_model_idx]}")

    cif_pattern = os.path.join(seed_path, f"*sample_{best_model_idx}_model.cif")
    of3_cifs = glob.glob(cif_pattern)

    if not of3_cifs:
        print(f"No CIF for best model {best_model_idx} in {seed_path}")
        continue

    of3_cif_path = of3_cifs[0]
    print(f"Selected OF3 CIF file: {of3_cif_path}")
    print(f"Processing {af3_folder}")

    try:
        # Load reference + original mobile structure
        structure_ref = parser.get_structure("ref", af3_cif_path)

        # -------------------------
        # RMSD when aligned on binder
        # -------------------------
        structure_mob = parser.get_structure("mobA", of3_cif_path)
        binder_rmsd_on_binder = rmsd_of_binder_after_alignment(
            structure_ref, structure_mob, align_chain="A"
        )

        # -------------------------
        # RMSD when aligned on target
        # -------------------------
        structure_mob = parser.get_structure("mobB", of3_cif_path)
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