from itertools import chain
import os
import glob
import csv
import json
import numpy as np
import re 
import pandas as pd
import gemmi
import tempfile

from Bio.PDB import MMCIFParser, Polypeptide, is_aa, Superimposer, PDBParser
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1
from Bio import pairwise2

"""
Building a script that finds interface contact points that overlap as well as how many unique contacts each methods has.
"""

CONTACT_CUTOFF = 5.0 

# ----------------------------
# Helpers
# ----------------------------

def cif_to_clean_pdb(cif_path):
    doc = gemmi.cif.read_file(cif_path)
    structure = gemmi.make_structure_from_block(doc.sole_block())

    # Optional: fix long chain IDs (A-1 → A)
    for model in structure:
        for chain in model:
            if len(chain.name) > 1:
                chain.name = chain.name[0]

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    structure.write_pdb(tmp.name)
    return tmp.name

def get_chain_by_prefix(model, chain_id_prefix):
    for cid in model.child_dict.keys():
        if cid.startswith(chain_id_prefix):
            return model.child_dict[cid]
    raise KeyError(f"No chain starting with '{chain_id_prefix}'. "
                   f"Available chains: {list(model.child_dict.keys())}")

def load_structure(path, sid):
    if path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(sid, path)
    elif path.lower().endswith(".pdb"):
        parser = PDBParser(QUIET=True)
        return parser.get_structure(sid, path)
    else:
        raise ValueError(f"Unknown structure format: {path}")

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

### Part 1. Directory setup. ###

af3_root = "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output"
of3_root = "/mnt/dsdd_share/Elliott/of3_run1"

parser = MMCIFParser(QUIET=True)
pdb_parser = PDBParser(QUIET=True)

results = []

# --- OF3 lookup ---
of3_lookup = {}

for folder in os.listdir(of3_root):
    full_path = os.path.join(of3_root, folder)
    if not os.path.isdir(full_path):
        continue

    normalized = normalize_for_fuzzy_match(folder)
    of3_lookup[normalized] = folder

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

    of3_folder = of3_lookup.get(csv_id)
    af3_folder = af3_lookup.get(csv_name)

    if of3_folder and af3_folder:
        pairs.append((of3_folder, af3_folder))
    else:
        print(f"Could not match: id={csv_id}, name={csv_name}")

for of3_folder, af3_folder in pairs:
    of3_folder_path = os.path.join(of3_root, of3_folder)
    af3_folder_path = os.path.join(af3_root, af3_folder)

    # Get AF3 CIF
    af3_cifs = [f for f in os.listdir(af3_folder_path) if f.lower().endswith(".cif")]

    if not af3_cifs:
        print(f"No CIF in AF3 folder {af3_folder}")
        continue

    af3_cif_path = os.path.join(af3_folder_path, af3_cifs[0])

    # OF3 CIFs

    seed_folders = [os.path.join(of3_folder_path, d) for d in os.listdir(of3_folder_path) if os.path.isdir(os.path.join(of3_folder_path, d)) and d.startswith("seed_")]

    if not seed_folders:
        print(f"No seed folder in {of3_folder}")
        continue

    seed_path = seed_folders[0]

    json_files = [
        f for f in os.listdir(seed_path)
        if f.endswith(".json") and "confidences_aggregated" in f
    ]
    if not json_files:
        print(f"No confidence JSON files in {seed_path}")
        continue

    scores = {}

    for json_file in json_files:
        json_path = os.path.join(seed_path, json_file)
        with open(json_path, "r") as jf:
            data = json.load(jf)

        if "sample_ranking_score" not in data:
            print(f"No 'sample_ranking_score' in {json_file}")
            continue

        score = data["sample_ranking_score"]

        match = re.search(r"_sample_(\d+)_", json_file)
        if not match:
            print(f"Could not parse model index from {json_file}")
            continue

        model_idx = int(match.group(1))
        scores[model_idx] = score

    if not scores:
        print(f"No valid sample_ranking_score found in {seed_path}")
        continue

    best_model_idx = max(scores, key=scores.get)
    print(f"Best model: {best_model_idx} with score {scores[best_model_idx]}")

    cif_pattern = os.path.join(seed_path, f"*sample_{best_model_idx}_model.cif")
    of3_cifs = glob.glob(cif_pattern)

    if not of3_cifs:
        print(f"No CIF for best model {best_model_idx} in {seed_path}")
        continue

    of3_cif_path = of3_cifs[0]
    print(f"Selected OF3 CIF file: {of3_cif_path}")
    print(f"Processing {af3_folder}")

    try:
        # Load structures
        # -------------------------
        A = load_structure(af3_cif_path, af3_folder + "_AF3")
        of3_pdb_path = cif_to_clean_pdb(of3_cif_path)
        B = load_structure(of3_pdb_path, of3_folder + "_OF3")

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

        results.append([
            complex_id,
            int(MA.sum()),
            int(MB.sum()),
            ov1
        ])
    except Exception as e:
        print(f"Error processing {complex_id}: {e}")
        continue

# -----------------------------------------------------------------------
# Write CSV
# -----------------------------------------------------------------------

with open("contact_overlap.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Complex",
        "af3_contacts",
        "of3_contacts",
        "overlap_contacts"
    ])
    writer.writerows(results)

print("Saved contact_overlap.csv")