import MDAnalysis as mda
import os
import tempfile
import gemmi
import numpy as np
import csv

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

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
boltz_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1"

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
        print(f"No Boltz fodler match for {af3_folder}")
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

    af3_binder, af3_target = get_interface_residues(af3_cif_path)
    boltz_binder, boltz_target = get_interface_residues(boltz_cif_path)

    # binder overlap
    binder_intersection = af3_binder & boltz_binder
    binder_union = af3_binder | boltz_binder

    # target overlap
    target_intersection = af3_target & boltz_target
    target_union = af3_target | boltz_target

    binder_jaccard = len(binder_intersection) / len(binder_union) if binder_union else 0
    target_jaccard = len(target_intersection) / len(target_union) if target_union else 0

    results.append([
        af3_folder, 
        len(af3_binder), 
        len(boltz_binder),
        binder_jaccard,
        len(af3_target),
        len(boltz_target),
        target_jaccard
        ])
    
with open("interface_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complex",
        "AF3_binder_Interface_size",
        "Boltz_binder_interface_size",
        "Binder_Jaccard",
        "AF3_Target_interface_size",
        "Boltz_Target_interface_size",
        "target_jaccard"
    ])

    writer.writerows(results)
