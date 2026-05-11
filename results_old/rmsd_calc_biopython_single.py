from Bio.PDB import MMCIFParser, Superimposer

parser = MMCIFParser()

# Extract backbome atoms for chain A

structure1 = parser.get_structure("ref", "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output/alex.naka.venusaur/alex.naka.venusaur_model.cif")      # Path to AlphaFold 3 mmCIF
structure2  = parser.get_structure("mob", "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1/alex.naka.Venusaur/alex.naka.Venusaur_model_0.cif")    # Path to Boltz-2 mmCIF

atoms1A = []
atoms2A = []

for r1, r2 in zip(structure1[0]["A"], structure2[0]["A"]):

    if r1.id[1] != r2.id[1]:
        raise ValueError("Residue mismatch between models")

    if "CA" in r1 and "CA" in r2:
        atoms1A.append(r1["CA"])
        atoms2A.append(r2["CA"])

supA = Superimposer()
supA.set_atoms(atoms1A, atoms2A)

print("Binder RMSD (aligned on binder):", supA.rms)

# Extract for chain B

structure1 = parser.get_structure("ref", "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/AF3/AF3_run1_output/alex.naka.venusaur/alex.naka.venusaur_model.cif")      # Path to AlphaFold 3 mmCIF
structure2  = parser.get_structure("mob", "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1/alex.naka.Venusaur/alex.naka.Venusaur_model_0.cif")    # Path to Boltz-2 mmCIF

atoms1B = []
atoms2B = []

for r1, r2 in zip(structure1[0]["B"], structure2[0]["B"]):

    if r1.id[1] != r2.id[1]:
        raise ValueError("Residue mismatch between models")
    
    if "CA" in r1 and "CA" in r2:
        atoms1B.append(r1["CA"])
        atoms2B.append(r2["CA"])

supB = Superimposer()
supB.set_atoms(atoms1B, atoms2B)

print("Target RMSD (aligned on target):", supB.rms)