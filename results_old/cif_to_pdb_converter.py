from Bio.PDB import MMCIFParser, PDBIO

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("model", "/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Boltz-2/Boltz-2_run1/alex.naka.Venusaur/alex.naka.Venusaur_model_0.cif")

io = PDBIO()
io.set_structure(structure)
io.save("example1_boltz2.pdb")
