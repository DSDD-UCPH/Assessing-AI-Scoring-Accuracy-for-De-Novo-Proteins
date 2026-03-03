import MDAnalysis as mda
import os
import tempfile
import gemmi
import numpy as np
import csv
import re
import pandas as pd
import glob
import json

"""
Building a script that reads and calculates interface contacts between AF2 and AF3 while also taking file name and folder name normalization into account.
"""

### --- Functions --- ###

def load_cif_with_gemmi(cif_path):
    # read CIF
    structure = gemmi.read_structure(cif_path)

    # Write to temporary PDB
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    structure.write_pdb(tmp.name)
    tmp.close()

    # Load into MDAnalysis
    u = mda.Universe(tmp.name)

    # Remove temp file
    os.remove(tmp.name)

    return u


def get_interface_residues(cif_path, cutoff=4.5):
    u = load_cif_with_gemmi(cif_path)

    binder = u.select_atoms("chainID A and not name H*")
    target = u.select_atoms("chainID B and not name H*")

    binder_interface = set()
    target_interface = set()

    for resA in binder.residues:
        for resB in target.residues:
            
            dist = mda.lib.distances.distance_array(
                resA.atoms.positions,
                resB.atoms.positions
            )

            if np.any(dist < cutoff):
                binder_interface.add(resA.resid)
                target_interface.add(resB.resid)

    return binder_interface, target_interface

def get_interface_residues_pdb(pdb_path, cutoff=4.5):
    u = mda.Universe(pdb_path)

    binder = u.select_atoms("chainID A and not name H*")
    target = u.select_atoms("chainID B and not name H*")

    binder_interface = set()
    target_interface = set()

    for resA in binder.residues:
        for resB in target.residues:
            
            dist = mda.lib.distances.distance_array(
                resA.atoms.positions,
                resB.atoms.positions
            )

            if np.any(dist < cutoff):
                binder_interface.add(resA.resid)
                target_interface.add(resB.resid)

    return binder_interface, target_interface

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

### --- Directory setup --- ###

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
af2_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF2/EGFR_AF2_panel"

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

    af3_binder, af3_target = get_interface_residues(af3_cif_path)
    af2_binder, af2_target = get_interface_residues_pdb(af2_pdb_path)
    # binder overlap
    binder_intersection = af3_binder & af2_binder
    binder_union = af3_binder | af2_binder

    # target overlap
    target_intersection = af3_target & af2_target
    target_union = af3_target | af2_target

    binder_jaccard = len(binder_intersection) / len(binder_union) if binder_union else 0
    target_jaccard = len(target_intersection) / len(target_union) if target_union else 0

    results.append([
        af3_folder, 
        len(af3_binder), 
        len(af2_binder),
        binder_jaccard,
        len(af3_target),
        len(af2_target),
        target_jaccard
        ])
    
with open("interface_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complex",
        "AF3_binder_Interface_size",
        "AF2_binder_interface_size",
        "Binder_Jaccard",
        "AF3_Target_interface_size",
        "AF2_Target_interface_size",
        "target_jaccard"
    ])

    writer.writerows(results)