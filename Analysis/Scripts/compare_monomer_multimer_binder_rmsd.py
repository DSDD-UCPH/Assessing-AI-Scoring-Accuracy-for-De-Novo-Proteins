#!/usr/bin/env python3
"""Compare rank-1 monomer and multimer binder folds by method.

This script:
1. Builds a rank-1 monomer table from a larger monomer predictions parquet.
2. Joins it to an existing rank-1 multimer table by binder_sequence and method.
3. Extracts binder CA coordinates from each structure.
4. Aligns binder-on-binder with Kabsch superposition and reports CA RMSD.
5. Writes a table and a method-wise plot.

Example:
    python compare_monomer_multimer_binder_rmsd.py \
      --monomer-parquet monomer_predictions.parquet \
      --multimer-rank1-parquet rank1_structures.parquet \
      --out-prefix monomer_vs_multimer
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser, PDBParser
import gemmi
import tempfile
import os


JOIN_KEYS = ["binder_sequence", "method"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate binder-only CA RMSD between rank-1 monomer and multimer predictions."
    )
    parser.add_argument("--monomer-parquet", required=True, help="Large monomer predictions parquet.")
    parser.add_argument(
        "--multimer-rank1-parquet",
        required=True,
        help="Existing rank1_structures parquet for multimer predictions.",
    )
    parser.add_argument(
        "--out-prefix",
        default="monomer_vs_multimer",
        help="Prefix for output files. Default: monomer_vs_multimer",
    )
    parser.add_argument(
        "--join-keys",
        nargs="+",
        default=JOIN_KEYS,
        help="Columns used to pair monomer and multimer rows. Default: binder_sequence method",
    )
    parser.add_argument(
        "--min-common-ca",
        type=int,
        default=10,
        help="Minimum number of paired CA atoms required for RMSD. Default: 10",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def build_rank1_monomers(monomers: pd.DataFrame, join_keys: list[str]) -> pd.DataFrame:
    require_columns(monomers, [*join_keys, "path"], "monomer parquet")

    df = monomers.copy()
    if "parse_success" in df.columns:
        df = df[df["parse_success"].fillna(False)]

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "rank" in df.columns:
        sort_columns.append("rank")
        ascending.append(True)
    if "score" in df.columns:
        sort_columns.append("score")
        ascending.append(False)

    if sort_columns:
        df = df.sort_values(sort_columns, ascending=ascending, na_position="last")

    keep_columns = [
        column
        for column in [
            *join_keys,
            "path",
            "file_type",
            "prediction_id",
            "rank",
            "binder_chain_id",
            "binder_length",
            "score",
            "score_name",
            "json_path",
            "pkl_path",
            "npz_path",
            "parse_success",
        ]
        if column in df.columns
    ]

    rank1 = df.drop_duplicates(subset=join_keys, keep="first")[keep_columns].reset_index(drop=True)
    return rank1

def cif_to_tmp_pdb(cif_path):
    cif_path = Path(cif_path)

    doc = gemmi.cif.read_file(str(cif_path))
    structure = gemmi.make_structure_from_block(doc.sole_block())

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    structure.write_pdb(tmp.name)
    tmp.close()
    return tmp.name

def load_structure(path: str | Path, method: str | None = None):
    path = Path(path)

    if path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
        return parser.get_structure(path.stem, str(path))

    elif path.suffix.lower() == ".cif":
        if method == "OF3":
            tmp_pdb = cif_to_tmp_pdb(path)
            try:
                parser = PDBParser(QUIET=True)
                return parser.get_structure(path.stem, tmp_pdb)
            finally:
                os.unlink(tmp_pdb)


        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(path.stem, str(path))

    else:
        raise ValueError(f"Unsupported structure file type: {path}")


def chain_candidates(chain_id: object) -> list[str]:
    if pd.isna(chain_id):
        return []
    raw = str(chain_id)
    candidates = [raw]
    if "-" in raw:
        candidates.append(raw.split("-", 1)[0])
    return list(dict.fromkeys(candidates))


def select_chain(structure, chain_id: object | None, monomer: bool):
    chains = list(structure.get_chains())
    if monomer and len(chains) == 1:
        return chains[0]

    by_id = {chain.id: chain for chain in chains}
    for candidate in chain_candidates(chain_id):
        if candidate in by_id:
            return by_id[candidate]

    if monomer and chains:
        return chains[0]

    available = ", ".join(chain.id for chain in chains)
    raise ValueError(f"Could not find chain {chain_id!r}. Available chains: {available}")


def ca_coordinates(
    path: str,
    chain_id: object | None = None,
    monomer: bool = False,
    method: str | None = None,
) -> np.ndarray:
    structure_path = Path(path)
    if not structure_path.exists():
        raise FileNotFoundError(structure_path)

    structure = load_structure(structure_path, method=method)

    chain = select_chain(structure, chain_id, monomer=monomer)
    coords = []

    for residue in chain.get_residues():
        hetflag = residue.id[0]
        if hetflag != " ":
            continue
        if "CA" in residue:
            coords.append(residue["CA"].coord.astype(float))

    if not coords:
        raise ValueError(f"No CA atoms found in {structure_path} chain {chain.id}")
    return np.asarray(coords, dtype=float)


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    if reference.shape != mobile.shape:
        raise ValueError(f"Coordinate arrays must have same shape, got {reference.shape} and {mobile.shape}")

    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref = reference - ref_center
    mob = mobile - mob_center

    covariance = mob.T @ ref
    v, _, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(v @ wt))
    rotation = v @ correction @ wt
    aligned = mob @ rotation
    diff = aligned - ref
    return float(np.sqrt((diff * diff).sum() / len(reference)))


def paired_ca_rmsd(
    monomer_path: str,
    multimer_path: str,
    multimer_binder_chain_id: object,
    min_common_ca: int,
    method: str | None = None,
) -> tuple[float, int, int, int]:
    monomer_ca = ca_coordinates(monomer_path, monomer=True, method=method)
    multimer_ca = ca_coordinates(
        multimer_path,
        chain_id=multimer_binder_chain_id,
        monomer=False,
        method=method,
    )

    common = min(len(monomer_ca), len(multimer_ca))
    if common < min_common_ca:
        raise ValueError(
            f"Too few paired CA atoms: monomer={len(monomer_ca)}, multimer={len(multimer_ca)}, common={common}"
        )

    rmsd = kabsch_rmsd(multimer_ca[:common], monomer_ca[:common])
    return rmsd, common, len(monomer_ca), len(multimer_ca)


def compare_structures(joined: pd.DataFrame, min_common_ca: int) -> pd.DataFrame:
    rows = []
    for row in joined.itertuples(index=False):
        record = row._asdict()
        try:
            rmsd, common_ca, monomer_ca, multimer_ca = paired_ca_rmsd(
                record["path_monomer"],
                record["path_multimer"],
                record.get("binder_chain_id_multimer"),
                min_common_ca,
                method=record.get("method"),
            )
            record.update(
                {
                    "binder_ca_rmsd": rmsd,
                    "common_ca_atoms": common_ca,
                    "monomer_ca_atoms": monomer_ca,
                    "multimer_ca_atoms": multimer_ca,
                    "rmsd_status": "ok",
                    "rmsd_error": "",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "binder_ca_rmsd": math.nan,
                    "common_ca_atoms": math.nan,
                    "monomer_ca_atoms": math.nan,
                    "multimer_ca_atoms": math.nan,
                    "rmsd_status": "error",
                    "rmsd_error": str(exc),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def plot_rmsd(results: pd.DataFrame, plot_path: Path) -> None:
    ok = results[results["rmsd_status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful RMSD rows to plot.")

    method_order = ok.groupby("method")["binder_ca_rmsd"].median().sort_values().index

    plt.figure(figsize=(max(8, 0.8 * len(method_order)), 5.5))
    ax = sns.boxplot(
        data=ok,
        x="method",
        y="binder_ca_rmsd",
        order=method_order,
        color="#d9e6f2",
        width=0.55,
        showfliers=False,
    )
    sns.stripplot(
        data=ok,
        x="method",
        y="binder_ca_rmsd",
        order=method_order,
        color="#1f2937",
        alpha=0.45,
        size=3,
        jitter=0.22,
        ax=ax,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Binder CA RMSD after binder-on-binder alignment (Angstrom)")
    ax.set_title("Rank-1 monomer vs rank-1 multimer binder fold agreement")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)

    monomers = pd.read_parquet(args.monomer_parquet)
    multimers = pd.read_parquet(args.multimer_rank1_parquet)

    require_columns(multimers, [*args.join_keys, "path", "binder_chain_id"], "multimer rank1 parquet")

    rank1_monomers = build_rank1_monomers(monomers, args.join_keys)
    rank1_monomers_path = out_prefix.with_name(out_prefix.name + "_rank1_monomer_structures.parquet")
    rank1_monomers.to_parquet(rank1_monomers_path, index=False)

    joined = rank1_monomers.merge(
        multimers,
        on=args.join_keys,
        how="inner",
        suffixes=("_monomer", "_multimer"),
        validate="one_to_many",
    )

    results = compare_structures(joined, min_common_ca=args.min_common_ca)
    results_path = out_prefix.with_name(out_prefix.name + "_binder_rmsd.parquet")
    csv_path = out_prefix.with_name(out_prefix.name + "_binder_rmsd.csv")
    plot_path = out_prefix.with_name(out_prefix.name + "_binder_rmsd_by_method.png")

    results.to_parquet(results_path, index=False)
    results.to_csv(csv_path, index=False)
    plot_rmsd(results, plot_path)

    summary = (
        results.groupby(["method", "rmsd_status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["method", "rmsd_status"])
    )
    ok = results[results["rmsd_status"].eq("ok")]
    method_stats = (
        ok.groupby("method")["binder_ca_rmsd"]
        .agg(["count", "median", "mean", "std", "min", "max"])
        .sort_values("median")
    )

    print(f"Wrote rank-1 monomers: {rank1_monomers_path}")
    print(f"Wrote RMSD parquet:   {results_path}")
    print(f"Wrote RMSD CSV:       {csv_path}")
    print(f"Wrote plot:           {plot_path}")
    print("\nRMSD status counts:")
    print(summary.to_string(index=False))
    print("\nMethod RMSD summary:")
    print(method_stats.to_string())


if __name__ == "__main__":
    main()
