"""
A script building functions that creates a rank-1 dataframe with only binder sequences made to be rank one by the methods, 
and then calculating contact interfaces. 
"""

from itertools import combinations
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import json

from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from pairwise_rank1_rmsd import load_structure_biopython, get_chain_by_id, get_protein_residues, residue_key

def build_rank1_structures_df(
    df: pd.DataFrame,
    binder_col: str = "binder_sequence",
    method_col: str = "method",
    structure_file_col: str = "path",
    rank_col: str = "rank",
    file_type_col: str = "file_type",
    binder_id_col: str = "binder_chain_id",
    target_id_col: str = "target_chain_id",
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    if keep_cols is None:
        keep_cols = ["score", "score_name"]

    required = [
        binder_col,
        method_col,
        structure_file_col,
        rank_col,
        file_type_col,
        binder_id_col,
        target_id_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cols_to_use = required + [c for c in keep_cols if c in df.columns]
    work = df[cols_to_use].copy()

    work = work[work[rank_col] == 1].copy()
    work = work.dropna(
        subset=[
            binder_col,
            method_col,
            structure_file_col,
            file_type_col,
            binder_id_col,
            target_id_col,
        ]
    )

    dup_mask = work.duplicated(subset=[binder_col, method_col], keep=False)
    if dup_mask.any():
        raise ValueError(
            "Found multiple rank-1 rows for the same binder_sequence and method."
        )

    return work.sort_values([binder_col, method_col]).reset_index(drop=True)

def make_residue_record(residue, prefix: str) -> dict:
    hetflag, resseq, icode = residue.id
    return {
        f"{prefix}_chain_id": residue.get_parent().id,
        f"{prefix}_resname": residue.get_resname(),
        f"{prefix}_resseq": int(resseq),
        f"{prefix}_icode": "" if icode is None or icode == " " else str(icode),
        f"{prefix}_hetflag": hetflag,
    }


def extract_interface_contacts(structure, binder_chain_id, target_chain_id, cutoff=5.0):
    binder_chain = get_chain_by_id(structure, binder_chain_id)
    target_chain = get_chain_by_id(structure, target_chain_id)

    binder_residues = get_protein_residues(binder_chain)
    target_residues = get_protein_residues(target_chain)

    target_atoms = [atom for residue in target_residues for atom in residue]
    if not target_atoms:
        raise ValueError("No target atoms found.")

    ns = NeighborSearch(target_atoms)

    target_contacts = {}
    binder_contacts = {}
    contact_pairs = {}

    for binder_res in binder_residues:
        if binder_res.id[0] != " ":
            continue
        if not is_aa(binder_res, standard=False):
            continue

        binder_key = residue_key(binder_res)

        for atom in binder_res:
            nearby_residues = ns.search(atom.coord, cutoff, level="R")

            for target_res in nearby_residues:
                if target_res.get_parent().id != target_chain_id:
                    continue
                if target_res.id[0] != " ":
                    continue
                if not is_aa(target_res, standard=False):
                    continue

                target_key = residue_key(target_res)

                target_contacts[target_key] = target_res
                binder_contacts[binder_key] = binder_res
                contact_pairs[(target_key, binder_key)] = (target_res, binder_res)

    return target_contacts, binder_contacts, contact_pairs


def compute_contacts_for_row(row, cutoff=5.0):
    base = {
        "binder_sequence": row["binder_sequence"],
        "method": row["method"],
        "path": row["path"],
        "file_type": row["file_type"],
        "binder_chain_id": row["binder_chain_id"],
        "target_chain_id": row["target_chain_id"],
        "score": row.get("score", None),
        "score_name": row.get("score_name", None),
        "contact_cutoff": cutoff,
    }

    try:
        structure = load_structure_biopython(
            path=row["path"],
            file_type=row["file_type"],
            structure_id="contact_struct",
        )

        target_contacts, binder_contacts, contact_pairs = extract_interface_contacts(
            structure=structure,
            binder_chain_id=row["binder_chain_id"],
            target_chain_id=row["target_chain_id"],
            cutoff=cutoff,
        )

        summary_row = {
            **base,
            "n_target_contact_residues": len(target_contacts),
            "n_binder_contact_residues": len(binder_contacts),
            "n_residue_contact_pairs": len(contact_pairs),
            "status_contacts": "ok",
            "error_contacts": None,
        }

        target_rows = [
            {
                **base,
                **make_residue_record(target_res, "target"),
            }
            for _, target_res in sorted(target_contacts.items())
        ]

        binder_rows = [
            {
                **base,
                **make_residue_record(binder_res, "binder"),
            }
            for _, binder_res in sorted(binder_contacts.items())
        ]

        pair_rows = [
            {
                **base,
                **make_residue_record(target_res, "target"),
                **make_residue_record(binder_res, "binder"),
            }
            for _, (target_res, binder_res) in sorted(contact_pairs.items())
        ]

        return summary_row, target_rows, binder_rows, pair_rows

    except Exception as e:
        summary_row = {
            **base,
            "n_target_contact_residues": 0,
            "n_binder_contact_residues": 0,
            "n_residue_contact_pairs": 0,
            "status_contacts": "failed",
            "error_contacts": str(e),
        }
        return summary_row, [], [], []



if __name__ == "__main__":
    DATA_DIR = Path("/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/Analysis/Data")
    OUTPUT_DIR = DATA_DIR / "contact_interface_data/new_parquets"

    INPUT_PATH = DATA_DIR / "prediction_records_new.parquet"

    BINDER_LONG_PATH = OUTPUT_DIR / "rank1_binder_contacts_long_5A.parquet"
    PAIR_LONG_PATH = OUTPUT_DIR / "rank1_contact_pairs_long_5A.parquet"
    RANK1_PATH = OUTPUT_DIR / "rank1_structures.parquet"
    SUMMARY_PATH = OUTPUT_DIR / "rank1_contact_summary_5A.parquet"
    TARGET_LONG_PATH = OUTPUT_DIR / "rank1_target_contacts_long_5A.parquet"

    cutoff = 5.0

    df = pd.read_parquet(INPUT_PATH)

    rank1_df = build_rank1_structures_df(
        df,
        binder_col="binder_sequence",
        method_col="method",
        structure_file_col="path",
        rank_col="rank",
        file_type_col="file_type",
        binder_id_col="binder_chain_id",
        target_id_col="target_chain_id",
        keep_cols=["score", "score_name"],
    )
    rank1_df.to_parquet(RANK1_PATH, index=False)

    print(f"Rank-1 structures: {len(rank1_df)}")
    print(f"Unique binders: {rank1_df['binder_sequence'].nunique()}")

    results = Parallel(n_jobs=-2, verbose=10)(
        delayed(compute_contacts_for_row)(row, cutoff=cutoff)
        for _, row in rank1_df.iterrows()
    )

    summary_rows = [x[0] for x in results]
    target_rows = [row for x in results for row in x[1]]
    binder_rows = [row for x in results for row in x[2]]
    pair_rows = [row for x in results for row in x[3]]

    summary_df = pd.DataFrame(summary_rows)
    target_long_df = pd.DataFrame(target_rows)
    binder_long_df = pd.DataFrame(binder_rows)
    pair_long_df = pd.DataFrame(pair_rows)


    print("\nContact status:")
    print(summary_df["status_contacts"].value_counts(dropna=False))

    print("\nSummary head:")
    print(summary_df.head())

    print("\nTarget-contact long head:")
    print(target_long_df.head())

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(SUMMARY_PATH, index=False)
    target_long_df.to_parquet(TARGET_LONG_PATH, index=False)
    binder_long_df.to_parquet(BINDER_LONG_PATH, index=False)
    pair_long_df.to_parquet(PAIR_LONG_PATH, index=False)


    print(f"\nSaved summary parquet to: {SUMMARY_PATH}")
    print(f"Saved target-contact long parquet to: {TARGET_LONG_PATH}")
    print(f"Saved binder-contact long parquet to: {BINDER_LONG_PATH}")
    print(f"Saved contact-pair long parquet to: {PAIR_LONG_PATH}")
