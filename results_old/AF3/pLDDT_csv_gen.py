import AF3_analysis_pipeline as af3
import pandas as pd
import glob
import os
import json
import numpy as np

rows = []
all_plddt_rows = []

root_dir = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
complex_dirs = sorted(glob.glob(os.path.join(root_dir, "*/")))

print(f"Found {len(complex_dirs)} complex directories to process.")

for complex_dir in complex_dirs:
    if not os.path.isdir(complex_dir):
        continue # skip files accidentally matched

    # Folder name is Unique ID
    complex_id = os.path.basename(os.path.normpath(complex_dir))

    cif_path = af3.find_file(complex_dir, "*model.cif")
    conf_path = af3.find_real_confidences_json(complex_dir)

    if cif_path is None or conf_path is None:
        print(f"Skipping {complex_dir} due to missing files.")
        continue

    # Extract pLDDT from CIF
    structure, plddt_df = af3.load_structure_and_plddt(cif_path)

    # Drop model index (Always 0 for AF3)

    if "model_index" in plddt_df.columns:
        plddt_df = plddt_df.drop(columns=["model_index"])

    # Add complex ID
    plddt_df.insert(0, "complex_id", complex_id)  # Insert at the beginning of the DataFrame

    all_plddt_rows.append(plddt_df.copy())

    # Compute per-chain pLDDT
    chain_mean_plddt = plddt_df.groupby("chain")["plddt"].mean().to_dict()

    # load confidences.json (PAE, contact_probs, etc.)
    pae, contact_probs, chain_ids, res_ids = af3.load_confidences(conf_path)

    # Skip if PAE or contact_probs missing
    if pae is None or contact_probs is None:
        print(f"[SKIP interface metrics] {complex_id}: PAE or contact_probs missing.")
        continue

    # Get chain ranges
    ranges = af3.get_chain_ranges(chain_ids)
    chain_list = list(ranges.keys())

    if len(chain_list) != 2:
        print(f"[SKIP interface metrics] {complex_id}: Expected 2 chains but found {chain_list}")
        continue

    chainA, chainB = chain_list # works even if chain IDs are 'A'/'B' or Numeric

    # compute interface metrics
    try:
        paeAB, _ = af3.interface_mean_pae(pae, ranges, chainA, chainB)
        contact_scores = af3.interface_contact_score(contact_probs, ranges, chainA, chainB)
    except Exception as e:
        print(f"Error extracting interface metrics for {complex_id}: {e}")
        continue

    # add result row
    rows.append({
        "complex_id": complex_id,
        "ChainA_id": chainA,
        "ChainB_id": chainB,
        "mean_plddt_chainA": chain_mean_plddt.get(chainA, None),
        "mean_plddt_chainB": chain_mean_plddt.get(chainB, None),
        "interface_pae_AB": paeAB,
        "interface_contact_mean": contact_scores["mean_contact_prob"],
        "interface_contact_max": contact_scores["max_contact_prob"],
        "interface_contact_sum": contact_scores["sum_contact_prob"]
    })

# Save all-residue pLDDT table
all_plddt = pd.concat(all_plddt_rows, ignore_index=True)
plddt_out = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/af3_plddt_all_complexes.csv"
all_plddt.to_csv(plddt_out, index=False)
print(f"Saved all-residue pLDDT table to:\n{plddt_out}")

print("Done.")