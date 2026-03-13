import json
from pathlib import Path
import pandas as pd

base_dir = Path("/home/postyr/Assessing-AI-Scoring-Accuracy-for-De-Novo-Proteins/results/Protenix/Protenix_run_1")

rows = []

for json_file in base_dir.rglob("*sample_0.json"):
    folder_name = json_file.parts[-4]

    complex_id = folder_name.rsplit("_EGFR", 1)[0] if folder_name.endswith("_EGFR") else folder_name

    with open(json_file) as f:
        data = json.load(f)

    data["complex_id"] = complex_id

    rows.append(data)

df = pd.DataFrame(rows)

cols = ["complex_id"] + [c for c in df.columns if c != "complex_id"]
df = df[cols]

df.to_csv("protenix_compiled.csv", index=False)

print(f"Saved {len(df)} rows to protenix_compiled.csv")

