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
chai1_root = "/mnt/dsdd_share/Elliott/chai_run1"

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

        af3_binder, af3_target = get_interface_residues(af3_cif_path)
        chai1_binder, chai1_target = get_interface_residues(best_cif_path)

        # binder overlap
        binder_intersection = af3_binder & chai1_binder
        binder_union = af3_binder | chai1_binder

        # target overlap
        target_intersection = af3_target & chai1_target
        target_union = af3_target | chai1_target

        binder_jaccard = len(binder_intersection) / len(binder_union) if binder_union else 0
        target_jaccard = len(target_intersection) / len(target_union) if target_union else 0

        results.append([
            af3_folder, 
            len(af3_binder), 
            len(chai1_binder),
            binder_jaccard,
            len(af3_target),
            len(chai1_target),
            target_jaccard
            ])
    
with open("interface_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complex",
        "AF3_binder_Interface_size",
        "CHAI-1_binder_interface_size",
        "Binder_Jaccard",
        "AF3_Target_interface_size",
        "CHAI-1_Target_interface_size",
        "target_jaccard"
    ])

    writer.writerows(results)
