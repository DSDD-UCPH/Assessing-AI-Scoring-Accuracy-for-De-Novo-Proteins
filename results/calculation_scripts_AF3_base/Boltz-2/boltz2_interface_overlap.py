#!/usr/bin/env python3

from itertools import chain
import os
import glob
import csv
import numpy as np

from Bio.PDB import MMCIFParser, Polypeptide, is_aa, Superimposer
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1
from Bio import pairwise2

# ----------------------------
# User inputs (edit these)
# ----------------------------
AF3_ROOT     =  "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"     # Path to AlphaFold 3 directory
BOLTZ2_ROOT  = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1"    # Path to Boltz-2 directory
AF3_GLOB = "*/*_model.cif"
BOLTZ2_GLOB = "*/*_model_0.cif"
SUMMARY_CSV = "contact_summary_af3_boltz2.csv"

CONTACT_CUTOFF = 5.0

OUT_PREFIX = "af3_vs_boltz2"

# ----------------------------
# Helpers
# ----------------------------
def load_structure(path, sid):
    return MMCIFParser(QUIET=True).get_structure(sid, path)

def chainAA(chain):
    seq = []
    std = []

    for r in chain:
        if not is_aa(r, standard=False):
            continue

        try:
            aa = protein_letters_3to1(r.get_resname().strip().upper())
        except:
            aa = "X"
        seq.append( aa if len(aa)==1 else "X")
        if is_aa(r, standard=True):
            std.append(r)
        return "".join(seq), std


def aa1(res):
    """Return 1-letter AA; fallback to 'X' for unknowns/modified residues."""
    resname = res.get_resname().strip().upper()
    # protein_letters_3to1 is a dict → use .get(..., 'X') to avoid None
    aa = protein_letters_3to1.get(resname, "X")
    # Ensure it's a single character string
    return aa if isinstance(aa, str) and len(aa) == 1 else "X"


def heavy_coords(res):
    coords = []
    for atom in res.get_atoms():
        if not atom.get_name().startswith("H"):
            coords.append(atom.get_coord())
    return np.vstack(coords) if coords else np.zeros((0, 3))

def contact_map(chain_binder, chain_target, cutoff=CONTACT_CUTOFF):
    b_res = [r for r in chain_binder if is_aa(r, standard=True)]
    t_res = [r for r in chain_target if is_aa(r, standard=True)]
    B, T = len(b_res), len(t_res)
    M = np.zeros((B,T), dtype=bool)

    bxyz = [heavy_coords(r) for r in b_res]
    txyz = [heavy_coords(r) for r in t_res]

    c2 = cutoff * cutoff
    for i in range(B):
        for j in range(T):
            if bxyz[i].size==0 or txyz[j].size==0:
                continue
            diff = bxyz[i][:,None,:] - txyz[j][None,:,:]
            if np.any(np.sum(diff*diff,axis=2) <= c2):
                M[i,j] = True
    return M, b_res, t_res

def align_pairs(seqA, seqB):
    """Return list of (iA, iB) where both aligned positions are residues"""
    aln = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)[0]
    a, b = aln.seqA, aln.seqB
    iA=iB=0
    out=[]
    for x,y in zip(a,b):
        if x !="-" and y!="-":
            out.append((iA,iB))
        if x!="-": iA+=1
        if y!="-": iB+=1
    return out

def overlap_contacts(MA, MB, seqA_b, seqA_t, seqB_b, seqB_t):
    """Compute contact overlap via sequence alignment"""
    bind_pairs = align_pairs(seqA_b, seqB_b)
    targ_pairs = align_pairs(seqA_t, seqB_t)
    map_b = dict(bind_pairs)
    map_t = dict(targ_pairs)

    overlap = 0
    for iA in range(MA.shape[0]):
        iB = map_b.get(iA)
        if iB is None or iB >= MB.shape[0]:
            continue
        js = np.where(MA[iA]) [0]
        for jA in js:
            jB = map_t.get(jA)
            if jB is None or jB >= MB.shape[1]:
                continue
            if MB[iB, jB]:
                overlap += 1
    return overlap

# ----------------------------
# Batch loop
# ----------------------------
def main():
    af3_files = glob.glob(os.path.join(AF3_ROOT, AF3_GLOB))
    boltz_files = glob.glob(os.path.join(BOLTZ2_ROOT, BOLTZ2_GLOB))

    # Match by filename stem
    def stem(path):
        folder = os.path.basename(os.path.dirname(path)).strip().lower()
        file = os.path.basename(path).strip().lower()
        return folder

    
    bol_map = {stem(f): f for f in boltz_files}

    # Write summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["complex_id", "af3_contacts", "boltz_contacts", "overlap_contacts"])

        for af3 in af3_files:
            sid = stem(af3)
            if sid not in bol_map:
                print(f"[WARN] no Boltz file for {sid}")
                continue

            bol = bol_map[sid]
            print(f"[INFO] Processing {sid}")

            # Load
            A = load_structure(af3, sid+"_A")
            B = load_structure(bol, sid+"_B")

            # Use consistent chain IDs (A=binder, B=target)
            A_b = A[0]["A"]; A_t = A[0]["B"]
            B_b = B[0]["A"]; B_t = B[0]["B"]

            # Contact maps
            MA, A_bres, A_tres = contact_map(A_b, A_t, CONTACT_CUTOFF)
            MB, B_bres, B_tres = contact_map(B_b, B_t, CONTACT_CUTOFF)

            # Seq for alignment (standard AAs only)
            seqA_b = "".join(aa1(r) for r in A_bres)
            seqA_t = "".join(aa1(r) for r in A_tres)
            seqB_b = "".join(aa1(r) for r in B_bres)
            seqB_t = "".join(aa1(r) for r in B_tres)

            ov1 = overlap_contacts(MA, MB, seqA_b, seqA_t, seqB_b, seqB_t)

            w.writerow([sid, int(MA.sum()), int(MB.sum()), ov1])

    print(f"Done. Summary written to {SUMMARY_CSV}")

if __name__ == "__main__":
    main()