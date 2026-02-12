import AF3_analysis_pipeline as af3
import pandas as pd
import glob
import os
import json
import numpy as np

rows = []

complex_dirs = sorted(glob.glob("/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output/*/"))

for complex_dir in complex_dirs:
    if not os.path.isdir(complex_dir):
        continue # skip files accidentally matched

    complex_id = os.path.basename(os.path.normpath(complex_dir))

    def find_file(folder, pattern):
        matches = glob.glob(os.path.join(folder, pattern))
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            print(f"Warning: Multiple files found for pattern {pattern} in folder {folder}. Using the first one.")
        return matches[0]

    cif_path = find_file(complex_dir, "*model.cif")
    conf_path = af3.find_real_confidences_json(complex_dir)
    if conf_path is None:
        print(f"[SKIP] {complex_dir}: no valid confidences.json found.")

    if cif_path is None or conf_path is None:
        print(f"Skipping {complex_dir} due to missing files.")
        continue

    # Extract pLDDT from CIF
    structure, plddt_df = af3.load_structure_and_plddt(cif_path)

    # Compute per-chain pLDDT
    chain_mean_plddt = plddt_df.groupby("chain")["plddt"].mean().to_dict()

    # load confidences.json (PAE, contact_probs, etc.)
    pae, contact_probs, chain_ids, res_ids = af3.load_confidences(conf_path)

    # Get chain ranges
    ranges = af3.get_chain_ranges(chain_ids)

    # compute interface metrics
    try:
        paeAB, paeBA = af3.interface_mean_pae(pae, ranges, "A", "B")
        contact_scores = af3.interface_contact_score(contact_probs, ranges, "A", "B")
    except KeyError:
        # chain naming mismatch - log it
        print(f"Warning: Chain A or B not found in {complex_dir}. Skipping interface metrics.")
        continue

    # add result row
    rows.append({
        "complex_id": complex_id,
        "mean_plddt_A": chain_mean_plddt.get("A", None),
        "mean_plddt_B": chain_mean_plddt.get("B", None),
        "interface_pae_AB": paeAB,
        "interface_contact_mean": contact_scores["mean_contact_prob"],
        "interface_contact_max": contact_scores["max_contact_prob"],
        "interface_contact_sum": contact_scores["sum_contact_prob"]
    })

# convert to dataframe

df = pd.DataFrame(rows)
output_path = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/af3_features_all_complexes.csv"
df.to_csv(output_path, index=False)