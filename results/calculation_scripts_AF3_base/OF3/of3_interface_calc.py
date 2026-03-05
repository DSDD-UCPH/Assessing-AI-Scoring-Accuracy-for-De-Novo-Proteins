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

def get_chain_id_by_prefix(universe, prefix):
    chain_ids = set(universe.atoms.chainIDs)
    
    for cid in chain_ids:
        if cid == prefix or cid.startswith(prefix + "-"):
            return cid
    
    raise ValueError(f"No chain starting with '{prefix}'. "
                     f"Available chains: {chain_ids}")

def load_cif_with_gemmi(cif_path):
    # read CIF
    structure = gemmi.read_structure(cif_path)

    # Rename long chain IDs to single characters
    for model in structure:
        for chain in model:
            if chain.name.startswith("A"):
                chain.name = "A"
            elif chain.name.startswith("B"):
                chain.name = "B"
            else:
                # fallback: truncate to first character
                chain.name = chain.name[0]
                
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

    binder_chain = get_chain_id_by_prefix(u, "A")
    target_chain = get_chain_id_by_prefix(u, "B")

    binder = u.select_atoms(f"chainID {binder_chain} and not name H*")
    target = u.select_atoms(f"chainID {target_chain} and not name H*")

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
of3_root = "/mnt/dsdd_share/Elliott/of3_run1"

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

    af3_binder, af3_target = get_interface_residues(af3_cif_path)
    of3_binder, of3_target = get_interface_residues(of3_cif_path)

        # binder overlap
    binder_intersection = af3_binder & of3_binder
    binder_union = af3_binder | of3_binder

        # target overlap
    target_intersection = af3_target & of3_target
    target_union = af3_target | of3_target

    binder_jaccard = len(binder_intersection) / len(binder_union) if binder_union else 0
    target_jaccard = len(target_intersection) / len(target_union) if target_union else 0

    results.append([
        af3_folder, 
        len(af3_binder), 
        len(of3_binder),
        binder_jaccard,
        len(af3_target),
        len(of3_target),
        target_jaccard
        ])
    
with open("interface_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complex",
        "AF3_binder_Interface_size",
        "OF3_binder_interface_size",
        "Binder_Jaccard",
        "AF3_Target_interface_size",
        "OF3_Target_interface_size",
        "target_jaccard"
    ])

    writer.writerows(results)