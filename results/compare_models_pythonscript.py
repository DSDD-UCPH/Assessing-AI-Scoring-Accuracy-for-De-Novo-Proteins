#!/usr/bin/env python3
"""
Compare AF3 vs Boltz-2 predictions for the same binder+target complex.

What it does:
1) Chain mapping (user-provided or auto by sequence identity).
2) RMSD for target (AF3 vs Boltz2) and binder (AF3 vs Boltz2) using Cα atoms.
3) Interface contact maps for each model (heavy-atom cutoff, default 5.0 Å).
4) Overlap heatmap: contacts present in BOTH models.
5) Saves PNG heatmaps and CSVs of contact pairs.

Dependencies:
  pip install biopython numpy matplotlib
Optional (for prettier plots):
  pip install seaborn

References:
- AF3 outputs mmCIF with full coordinates: https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md
- Boltz outputs include mmCIF/PDB with per-residue scores: https://deepwiki.com/jwohlwend/boltz/2.4-output-formats-and-interpretation
- Biopython Superimposer (Kabsch) for RMSD: https://biopython.org/docs/1.75/api/Bio.PDB.Superimposer.html
"""

import os
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB import MMCIFParser, Polypeptide, is_aa, Superimposer
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1

# ----------------------------
# User inputs (edit these)
# ----------------------------
AF3_CIF     = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output/alex.naka.venusaur/alex.naka.venusaur_model.cif"       # Path to AlphaFold 3 mmCIF
BOLTZ2_CIF  = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1/alex.naka.Venusaur/alex.naka.Venusaur_model_0.cif"    # Path to Boltz-2 mmCIF

# Provide chain IDs if you know them; otherwise set to None and auto-map by sequence.
CHAIN_MAP = {
    "AF3":     {"binder": "A", "target": "B"},   # e.g., {"binder": "A", "target": "B"} or None
    "BOLTZ2":  {"binder": "A", "target": "B"},
}

# Contact definition (heavy-atom cutoff in Angstroms)
CONTACT_CUTOFF = 5.0

# Plot settings
PLOT_DPI = 180
OUT_PREFIX = "af3_vs_boltz2"

# ----------------------------
# Helpers
# ----------------------------
def load_structure(cif_path: str, struct_id: str):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(struct_id, cif_path)

def chain_to_sequence_and_index(chain):
    """
    Returns:
      sequence: one-letter AA string (unknowns → 'X')
      seq_map:  list of (residue, CA_atom_or_None)
    """


    seq_chars = []
    seq_map = []

    warned_nonstring = False  # to avoid spamming logs

    for res in chain:
        # Keep amino-acid residues only (filters ligands/ions/NA/solvent)
        if not is_aa(res, standard=False):
            continue

        resname = res.get_resname().strip().upper()

        # Try canonical converter first
        try:
            aa = protein_letters_3to1(resname)  # returns a 1-letter string for standard/known residues
        except Exception:
            # Try our modification table
            aa = "X"

        # Ensure 'aa' is a 1-letter *string*
        if not isinstance(aa, str):
            # Convert to string just in case; if still not 1 char, use 'X'
            if not warned_nonstring:
                print(f"[WARN] Non-string one-letter code produced for residue '{resname}': {aa} (type={type(aa)}) — coercing.")
                warned_nonstring = True
            aa = str(aa)

        if len(aa) != 1:
            if not warned_nonstring:
                print(f"[WARN] One-letter code for '{resname}' had length {len(aa)} ('{aa}'); using 'X'.")
                warned_nonstring = True
            aa = "X"

        # Grab CA if present (can be None—handled later)
        ca = res["CA"] if "CA" in res else None

        seq_chars.append(aa)
        seq_map.append((res, ca))

    return "".join(seq_chars), seq_map

def best_chain_match_by_sequence(struct1, struct2) -> List[Tuple[str, str, float]]:
    """
    Return a list of best chain matches (id1, id2, identity) by global alignment of sequences.
    """
    chains1 = [ch for ch in next(struct1.get_models())]
    chains2 = [ch for ch in next(struct2.get_models())]
    seqs1 = {ch.id: chain_to_sequence_and_index(ch)[0] for ch in chains1}
    seqs2 = {ch.id: chain_to_sequence_and_index(ch)[0] for ch in chains2}

    scores = []
    for id1, s1 in seqs1.items():
        for id2, s2 in seqs2.items():
            if len(s1) == 0 or len(s2) == 0:
                identity = 0.0
            else:
                alns = PairwiseAligner.align.globalxx(s1, s2, one_alignment_only=True, score_only=False)
                aln = alns[0]
                matches = sum((a == b) and (a != '-') and (b != '-') for a, b in zip(aln.seqA, aln.seqB))
                length = sum((a != '-') and (b != '-') for a, b in zip(aln.seqA, aln.seqB))
                identity = matches / length if length > 0 else 0.0
            scores.append((id1, id2, identity))
    # Greedy max matching
    scores.sort(key=lambda x: x[2], reverse=True)
    used1, used2, pairs = set(), set(), []
    for id1, id2, ident in scores:
        if id1 in used1 or id2 in used2:
            continue
        pairs.append((id1, id2, ident))
        used1.add(id1); used2.add(id2)
    return pairs

def alignment_index_map(seqA: str, seqB: str) -> List[Tuple[int, int]]:
    """
    Compute index pairs of aligned positions (0-based) where both are residues (not gaps).
    """
    alns = PairwiseAligner.align.globalxx(seqA, seqB, one_alignment_only=True)
    seqA_aln, seqB_aln, *_ = alns[0]
    iA = iB = 0
    idx_pairs = []
    for a, b in zip(seqA_aln, seqB_aln):
        if a != "-" and b != "-":
            idx_pairs.append((iA, iB))
        if a != "-":
            iA += 1
        if b != "-":
            iB += 1
    return idx_pairs

def get_chain(struct, chain_id):
    return next((ch for ch in next(struct.get_models()) if ch.id == chain_id), None)

def rmsd_ca(chainA, chainB) -> float:
    """
    CA-based RMSD after superposition (Kabsch).
    Uses sequence alignment to choose matched CA positions.
    """
    seqA, mapA = chain_to_sequence_and_index(chainA)
    seqB, mapB = chain_to_sequence_and_index(chainB)
    idx_pairs = alignment_index_map(seqA, seqB)
    fixed, moving = [], []
    for iA, iB in idx_pairs:
        resA, caA = mapA[iA]
        resB, caB = mapB[iB]
        if caA is None or caB is None:
            continue
        fixed.append(caA)
        moving.append(caB)
    if len(fixed) < 3:
        return float("nan")
    sup = Superimposer()
    sup.set_atoms(fixed, moving)  # fits moving to fixed
    # sup.apply can be used to transform the moving structure if needed
    return float(sup.rms)

def atom_coords_heavy(res) -> np.ndarray:
    coords = []
    for atom in res.get_atoms():
        name = atom.get_name()
        # Skip hydrogens by atom name convention (PDB/mmCIF)
        if name.startswith("H"):
            continue
        coords.append(atom.get_coord())
    if len(coords) == 0:
        return np.zeros((0,3), dtype=float)
    return np.vstack(coords)

def interface_contact_map(chain_binder, chain_target, cutoff=5.0) -> Tuple[np.ndarray, List, List]:
    """
    Returns a boolean matrix [n_binder_res, n_target_res] indicating if any heavy-atom
    pair between residues is within 'cutoff' Å.
    Also returns the residue lists used for rows/cols.
    """
    # Build residue lists (standard AAs only)
    bind_res = [r for r in chain_binder if is_aa(r, standard=True)]
    targ_res = [r for r in chain_target if is_aa(r, standard=True)]
    R = len(bind_res); C = len(targ_res)
    mtx = np.zeros((R, C), dtype=bool)

    # Pre-extract heavy atom coords
    bind_coords = [atom_coords_heavy(r) for r in bind_res]
    targ_coords = [atom_coords_heavy(r) for r in targ_res]

    cutoff2 = cutoff * cutoff
    for i in range(R):
        Bi = bind_coords[i]
        if Bi.shape[0] == 0: 
            continue
        for j in range(C):
            Tj = targ_coords[j]
            if Tj.shape[0] == 0:
                continue
            # Brute-force min distance (fast enough for typical interfaces)
            # Compute pairwise squared distances via broadcasting
            diff = Bi[:, None, :] - Tj[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            if np.any(d2 <= cutoff2):
                mtx[i, j] = True
    return mtx, bind_res, targ_res

def residue_label(res) -> str:
    """Return label like 'A:123' (chain:resnum)."""
    ch = res.get_parent().id
    het, seq, icode = res.id
    return f"{ch}:{seq}{icode.strip() or ''}"

def plot_heatmap_bool(mtx: np.ndarray, rows: List, cols: List, title: str, out_png: str):
    plt.figure(figsize=(max(6, 0.15*mtx.shape[1]), max(4, 0.15*mtx.shape[0])))
    plt.imshow(mtx, cmap="Greys", interpolation="nearest", aspect="auto", origin="lower")
    plt.colorbar(label="Contact (True/False)")
    # Only add sparse ticks to keep labels readable
    step_r = max(1, mtx.shape[0] // 20)
    step_c = max(1, mtx.shape[1] // 20)
    row_labels = [residue_label(r) for r in rows]
    col_labels = [residue_label(c) for c in cols]
    plt.yticks(range(0, mtx.shape[0], step_r), [row_labels[i] for i in range(0, mtx.shape[0], step_r)], fontsize=7)
    plt.xticks(range(0, mtx.shape[1], step_c), [col_labels[j] for j in range(0, mtx.shape[1], step_c)], rotation=90, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI)
    plt.close()

def write_contacts_csv(mtx: np.ndarray, rows: List, cols: List, out_csv: str):
    with open(out_csv, "w") as f:
        f.write("binder_residue,target_residue\n")
        for i in range(mtx.shape[0]):
            for j in range(mtx.shape[1]):
                if mtx[i, j]:
                    f.write(f"{residue_label(rows[i])},{residue_label(cols[j])}\n")

# ----------------------------
# Main
# ----------------------------
def main():
    # Load
    af3  = load_structure(AF3_CIF, "AF3")
    bol2 = load_structure(BOLTZ2_CIF, "BOLTZ2")

    # Map chains
    binder_A, target_A = None, None
    binder_B, target_B = None, None

    if CHAIN_MAP["AF3"]["binder"] and CHAIN_MAP["AF3"]["target"] \
       and CHAIN_MAP["BOLTZ2"]["binder"] and CHAIN_MAP["BOLTZ2"]["target"]:
        binder_A = get_chain(af3, CHAIN_MAP["AF3"]["binder"])
        target_A = get_chain(af3, CHAIN_MAP["AF3"]["target"])
        binder_B = get_chain(bol2, CHAIN_MAP["BOLTZ2"]["binder"])
        target_B = get_chain(bol2, CHAIN_MAP["BOLTZ2"]["target"])
        print(f"[INFO] Using user-specified chains:")
        print(f"  AF3: binder={binder_A.id if binder_A else None}, target={target_A.id if target_A else None}")
        print(f"  BOLTZ2: binder={binder_B.id if binder_B else None}, target={target_B.id if target_B else None}")
    else:
        # Auto pairing all chains, then pick two highest-identity matches and you can assign manually
        matches = best_chain_match_by_sequence(af3, bol2)
        print("[INFO] Auto chain matches by sequence identity (AF3 vs BOLTZ2):")
        for id1, id2, ident in matches:
            print(f"  {id1} ~ {id2}  identity={ident:.1%}")
        print("\n[HINT] Edit CHAIN_MAP at the top using the chain IDs above for binder/target.")
        return

    if binder_A is None or target_A is None or binder_B is None or target_B is None:
        raise ValueError("Chain mapping failed. Check CHAIN_MAP and chain IDs in your CIFs.")

    # 1) RMSDs
    tgt_rmsd = rmsd_ca(target_A, target_B)
    bnd_rmsd = rmsd_ca(binder_A, binder_B)
    print(f"\nRMSD (Cα, AFTER superposition):")
    print(f"  Target  AF3 vs Boltz-2: {tgt_rmsd:.3f} Å")
    print(f"  Binder  AF3 vs Boltz-2: {bnd_rmsd:.3f} Å")

    # 2) Interfaces (contact maps)
    m_af3, af3_bres, af3_tres = interface_contact_map(binder_A, target_A, cutoff=CONTACT_CUTOFF)
    m_b2,  b2_bres,  b2_tres  = interface_contact_map(binder_B, target_B,  cutoff=CONTACT_CUTOFF)

    # Check residue list sizes match for overlap; if not, align by residue numbers.
    # We'll create index maps by (chain, seqnum, icode).
    def res_key(r):
        het, seq, icode = r.id
        return (r.get_parent().id, seq, icode.strip() or "")

    # Build index on AF3 and Boltz2 residue keys
    af3_bind_idx = {res_key(r): i for i, r in enumerate(af3_bres)}
    af3_targ_idx = {res_key(r): j for j, r in enumerate(af3_tres)}
    b2_bind_idx  = {res_key(r): i for i, r in enumerate(b2_bres)}
    b2_targ_idx  = {res_key(r): j for j, r in enumerate(b2_tres)}

    # Common residues by key
    common_bind_keys = [k for k in af3_bind_idx.keys() if k in b2_bind_idx]
    common_targ_keys = [k for k in af3_targ_idx.keys() if k in b2_targ_idx]

    # Build reduced matrices over common residue sets in same order
    def submatrix(M, row_map, col_map, row_keys, col_keys):
        R = len(row_keys); C = len(col_keys)
        out = np.zeros((R, C), dtype=bool)
        for i, rk in enumerate(row_keys):
            for j, ck in enumerate(col_keys):
                out[i, j] = M[row_map[rk], col_map[ck]]
        return out

    m_af3_common = submatrix(m_af3, af3_bind_idx, af3_targ_idx, common_bind_keys, common_targ_keys)
    m_b2_common  = submatrix(m_b2,  b2_bind_idx,  b2_targ_idx,  common_bind_keys, common_targ_keys)

    overlap = m_af3_common & m_b2_common

    # 3) Plots & CSVs
    os.makedirs("plots", exist_ok=True)
    os.makedirs("contacts", exist_ok=True)

    plot_heatmap_bool(m_af3, af3_bres, af3_tres,
                      title=f"AF3 interface contacts (cutoff={CONTACT_CUTOFF} Å)",
                      out_png=f"plots/{OUT_PREFIX}_AF3_contacts.png")
    plot_heatmap_bool(m_b2, b2_bres, b2_tres,
                      title=f"Boltz-2 interface contacts (cutoff={CONTACT_CUTOFF} Å)",
                      out_png=f"plots/{OUT_PREFIX}_Boltz2_contacts.png")

    # For overlap, we need the aligned residue lists (common keys)
    common_bind_res = [af3_bres[af3_bind_idx[k]] for k in common_bind_keys]
    common_targ_res = [af3_tres[af3_targ_idx[k]] for k in common_targ_keys]
    plot_heatmap_bool(overlap, common_bind_res, common_targ_res,
                      title=f"Overlap of contacts (AF3 ∩ Boltz-2), cutoff={CONTACT_CUTOFF} Å",
                      out_png=f"plots/{OUT_PREFIX}_overlap_contacts.png")

    # CSV export of contacts
    write_contacts_csv(m_af3, af3_bres, af3_tres, f"contacts/{OUT_PREFIX}_AF3_contacts.csv")
    write_contacts_csv(m_b2,  b2_bres,  b2_tres,  f"contacts/{OUT_PREFIX}_Boltz2_contacts.csv")
    write_contacts_csv(overlap, common_bind_res, common_targ_res, f"contacts/{OUT_PREFIX}_overlap_contacts.csv")

    # Summary
    print("\n=== Summary ===")
    print(f"Target RMSD (Cα): {tgt_rmsd:.3f} Å")
    print(f"Binder RMSD (Cα): {bnd_rmsd:.3f} Å")
    print(f"AF3 contacts:      {m_af3.sum()} pairs")
    print(f"Boltz-2 contacts:  {m_b2.sum()} pairs")
    print(f"Overlap contacts:  {overlap.sum()} pairs")
    print("\nFiles written:")
    print(f"  plots/{OUT_PREFIX}_AF3_contacts.png")
    print(f"  plots/{OUT_PREFIX}_Boltz2_contacts.png")
    print(f"  plots/{OUT_PREFIX}_overlap_contacts.png")
    print(f"  contacts/{OUT_PREFIX}_AF3_contacts.csv")
    print(f"  contacts/{OUT_PREFIX}_Boltz2_contacts.csv")
    print(f"  contacts/{OUT_PREFIX}_overlap_contacts.csv")

if __name__ == "__main__":
    main()