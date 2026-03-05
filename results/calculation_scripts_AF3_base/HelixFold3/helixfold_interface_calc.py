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
helixfold_root = "/mnt/dsdd_share/Elliott/helixfold_output_v2"

results = []

# --- HelixFold lookup ---
helixfold_lookup = {}

for folder in os.listdir(helixfold_root):
    full_path = os.path.join(helixfold_root, folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(folder)
    helixfold_lookup[normalized] = folder


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

    helixfold_folder = helixfold_lookup.get(csv_id)
    af3_folder = af3_lookup.get(csv_name)

    if helixfold_folder and af3_folder:
        pairs.append((helixfold_folder, af3_folder))
    else:
        print(f"Could not match: id={csv_id}, name={csv_name}")

for helixfold_folder, af3_folder in pairs:
    helixfold_folder_path = os.path.join(helixfold_root, helixfold_folder)
    af3_folder_path = os.path.join(af3_root, af3_folder)

    # Get AF3 CIF
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]

    if not af3_cifs:
        print(f"No CIF in AF3 folder {af3_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])

    # Get HelixFold3 CIF

    pattern = os.path.join(helixfold_folder_path, "*-rank1", "predicted_structure.cif")

    helixfold_cifs = glob.glob(pattern)

    if not helixfold_cifs:
        print(f"No rank1 CIF in HelixFold folder {helixfold_folder}")
        continue

    helixfold_cif_path = helixfold_cifs[0]

    print(f"Processing {af3_folder}")

    af3_binder, af3_target = get_interface_residues(af3_cif_path)
    helixfold_binder, helixfold_target = get_interface_residues(helixfold_cif_path)

        # binder overlap
    binder_intersection = af3_binder & helixfold_binder
    binder_union = af3_binder | helixfold_binder

        # target overlap
    target_intersection = af3_target & helixfold_target
    target_union = af3_target | helixfold_target

    binder_jaccard = len(binder_intersection) / len(binder_union) if binder_union else 0
    target_jaccard = len(target_intersection) / len(target_union) if target_union else 0

    results.append([
        af3_folder, 
        len(af3_binder), 
        len(helixfold_binder),
        binder_jaccard,
        len(af3_target),
        len(helixfold_target),
        target_jaccard
        ])
    
with open("interface_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complex",
        "AF3_binder_Interface_size",
        "HelixFold3_binder_interface_size",
        "Binder_Jaccard",
        "AF3_Target_interface_size",
        "HelixFold3_Target_interface_size",
        "target_jaccard"
    ])

    writer.writerows(results)