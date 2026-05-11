#!/usr/bin/env python3

from itertools import chain
import os
import glob
import csv
import numpy as np
import re 
import pandas as pd

from Bio.PDB import MMCIFParser, Polypeptide, is_aa, Superimposer, PDBParser
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1
from Bio import pairwise2

"""
Building a script that finds interface contact points that overlap as well as how many unique contacts each methods has.
"""

# ----------------------------
# User inputs (edit these)
# ----------------------------
AF3_ROOT     =  "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"     # Path to AlphaFold 3 directory
HELIXFOLD_ROOT  = "/mnt/dsdd_share/Elliott/helixfold_output_v2"    # Path to HelixFold3 directory
AF3_GLOB = "*/*_model.cif"
HELIXFOLD_GLOB = "*/*-rank1/predicted_structure.cif"
SUMMARY_CSV = "contact_summary_af3_helixfold.csv"

CONTACT_CUTOFF = 5.0

OUT_PREFIX = "af3_vs_helixfold"

# ----------------------------
# Helpers
# ----------------------------

def get_chain_by_prefix(model, chain_id_prefix):
    for cid in model.child_dict.keys():
        if cid.startswith(chain_id_prefix):
            return model.child_dict[cid]
    raise KeyError(f"No chain starting with '{chain_id_prefix}'. "
                   f"Available chains: {list(model.child_dict.keys())}")

def load_structure(path, sid):
    if path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unknown structure format: {path}")

    return parser.get_structure(sid, path)

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

# --- HelixFold lookup ---
parser = MMCIFParser(QUIET=True)

results = []

# --- HelixFold lookup ---
helixfold_lookup = {}

for folder in os.listdir(HELIXFOLD_ROOT):
    full_path = os.path.join(HELIXFOLD_ROOT, folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(folder)
    helixfold_lookup[normalized] = folder


# --- AF3 lookup ---
af3_lookup = {}

for af3_folder in os.listdir(AF3_ROOT):
    full_path = os.path.join(AF3_ROOT, af3_folder)
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

def main(): 
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["complex_id", "af3_contacts", "helixfold_contacts", "overlap_contacts"])

        for helixfold_folder, af3_folder in pairs:
            helixfold_folder_path = os.path.join(HELIXFOLD_ROOT, helixfold_folder)
            af3_folder_path = os.path.join(AF3_ROOT, af3_folder)

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

            # Load structures
            # -------------------------
            A = load_structure(af3_cif_path, af3_folder + "_AF3")
            B = load_structure(helixfold_cif_path, helixfold_folder + "_HELIXFOLD")

            # Use consistent chain IDs (A=binder, B=target)
            model_A = A[0]
            model_B = B[0]

            A_b = get_chain_by_prefix(model_A, "A")
            A_t = get_chain_by_prefix(model_A, "B") 

            B_b = get_chain_by_prefix(model_B, "A")
            B_t = get_chain_by_prefix(model_B, "B")

            # -------------------------
            # Contact maps
            # -------------------------
            MA, A_bres, A_tres = contact_map(A_b, A_t, CONTACT_CUTOFF)
            MB, B_bres, B_tres = contact_map(B_b, B_t, CONTACT_CUTOFF)

            # Seq for alignment (standard AAs only)
            seqA_b = "".join(aa1(r) for r in A_bres)
            seqA_t = "".join(aa1(r) for r in A_tres)
            seqB_b = "".join(aa1(r) for r in B_bres)
            seqB_t = "".join(aa1(r) for r in B_tres)

            # -------------------------
            # Overlap
            # -------------------------
            ov1 = overlap_contacts(MA, MB, seqA_b, seqA_t, seqB_b, seqB_t)

            # Use AF3 name as complex ID (or normalize if desired)
            complex_id = af3_folder

            w.writerow([
                complex_id,
                int(MA.sum()),
                int(MB.sum()),
                ov1
            ])

    print(f"Done. Summary written to {SUMMARY_CSV}")

if __name__ == "__main__":
    main()