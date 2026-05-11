import os
import csv
import pandas as pd
from Bio.PDB import MMCIFParser, Superimposer

def get_ca_atoms_by_resid(structure, chain_id):
    model = structure[0]
    chain = model[chain_id]

    ca_dict = {}

    for residue in chain:
        # skip hetero residues
        if residue.id[0] != " ":
            continue

        resid = residue.id[1]

        if "CA" in residue:
            ca_dict[resid] = residue["CA"]
    return ca_dict

def compute_chain_rmsd(structure1, structure2, chain_id):
    ca1 = get_ca_atoms_by_resid(structure1, chain_id)
    ca2 = get_ca_atoms_by_resid(structure2, chain_id)

    # Match common residues
    common_resids = sorted(set(ca1.keys()) & set(ca2.keys()))

    atoms1 = [ca1[r] for r in common_resids]
    atoms2 = [ca2[r] for r in common_resids]

    if len(atoms1) == 0:
        raise ValueError(f"No matching redisues found for chain {chain_id}")
    
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)

    return sup.rms

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
boltz_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1"

parser = MMCIFParser(QUIET=True)

# Build lookup of boltz folder (case insensitive)
boltz_folders = {folder.lower(): folder
                for folder in os.listdir(boltz_root)
                if os.path.isdir(os.path.join(boltz_root, folder))
                }

results = []

for af3_folder in os.listdir(af3_root):

    af3_folder_path = os.path.join(af3_root, af3_folder)
    if not os.path.isdir(af3_folder_path):
        continue

    key = af3_folder.lower()

    if key not in boltz_folders:
        print(f"No Boltz folder match for {af3_folder}")
        continue

    boltz_folder = boltz_folders[key]

    boltz_folder_path = os.path.join(boltz_root, boltz_folder)

    # Get AF3 CIF
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]
    if len(af3_cifs) == 0:
        print(f"No CIF in {af3_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])

    # Get Boltz CIF
    boltz_cifs = [f for f in os.listdir(boltz_folder_path) if f.lower().endswith(".cif")]
    if len(boltz_cifs) == 0:
        print(f"No CIF in {boltz_folder}")

    boltz_cif_path = os.path.join(boltz_folder_path, boltz_cifs[0])

    print(f"Processing {af3_folder}")

    try:
        structure1 = parser.get_structure("ref", af3_cif_path)
        structure2 = parser.get_structure("mob", boltz_cif_path)

        binder_rmsd = compute_chain_rmsd(structure1, structure2, "A")
        target_rmsd = compute_chain_rmsd(structure1, structure2, "B")

        results.append([key, binder_rmsd, target_rmsd])

    except Exception as e:
        print(f"Error processing {key}: {e}")
        continue

    with open("rmsd_results_boltz_af3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Complex", "Binder_RMSD", "Target_RMSD"])
        writer.writerows(results)
    
