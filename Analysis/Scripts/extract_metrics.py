from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd


class NativeMetricExtractor:
    def __init__(self):
        pass

    def _safe_float(self, value):
        try:
            if isinstance(value, (list, tuple, dict)):
                return None
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                return None
            return float(value)
        except Exception:
            return None

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _load_pkl(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_npz(self, path):
        return dict(np.load(path, allow_pickle=True))

    def _flatten_dict(self, d, parent_key=""):
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else str(k)
                if isinstance(v, dict):
                    items.update(self._flatten_dict(v, new_key))
                else:
                    items[new_key] = v
        return items

    def _extract_selected_numeric_keys(self, flat_dict, keys_to_keep, prefix=None):
        out = {}
        for key in keys_to_keep:
            if key in flat_dict:
                val = self._safe_float(flat_dict[key])
                if val is not None:
                    out[f"{prefix}_{key}" if prefix else key] = val
        return out

    def _extract_all_numeric_keys(self, flat_dict, prefix=None, exclude_substrings=None):
        out = {}
        exclude_substrings = exclude_substrings or []

        for key, value in flat_dict.items():
            if any(x in key.lower() for x in exclude_substrings):
                continue
            val = self._safe_float(value)
            if val is not None:
                clean_key = key.replace(".", "__")
                out[f"{prefix}_{clean_key}" if prefix else clean_key] = val
        return out

    def extract_row_metrics(self, row):
        method = row["method"]

        dispatch = {
            "AF2": self._extract_af2_metrics,
            "AF3": self._extract_af3_metrics,
            "HelixFold3": self._extract_helixfold3_metrics,
            "Chai-1": self._extract_chai1_metrics,
            "SeedFold": self._extract_seedfold_metrics,
            "Protenix": self._extract_protenix_metrics,
            "OF3": self._extract_of3_metrics,
            "Boltz-2": self._extract_boltz2_metrics,
            "OF2": self._extract_of2_metrics,
        }

        func = dispatch.get(method, self._extract_default_metrics)
        return func(row)

    def _extract_default_metrics(self, row):
        return {}

    def _extract_af2_metrics(self, row):
        out = {}
        pkl_path = row.get("pkl_path")

        if pd.notna(pkl_path) and Path(pkl_path).exists():
            data = self._load_pkl(Path(pkl_path))

            if "plddt" in data:
                plddt = np.asarray(data["plddt"])
                out["af2_plddt_mean"] = self._safe_float(np.mean(plddt))
                out["af2_plddt_median"] = self._safe_float(np.median(plddt))
                out["af2_plddt_min"] = self._safe_float(np.min(plddt))
                out["af2_plddt_max"] = self._safe_float(np.max(plddt))

            for key in ["ptm", "iptm"]:
                if key in data:
                    out[f"af2_{key}"] = self._safe_float(data.get(key))

        return out

    def _extract_af3_metrics(self, row):
        out = {}
        json_path = row.get("json_path")

        if pd.notna(json_path) and Path(json_path).exists():
            data = self._load_json(Path(json_path))
            flat = self._flatten_dict(data)

            # First: pull the most likely important AF3 metrics explicitly
            keys = [
                "ranking_score",
                "ptm",
                "iptm",
                "fraction_disordered",
                "has_clash",
                "disorder",
                "chain_pair_pae_min",
            ]
            out.update(self._extract_selected_numeric_keys(flat, keys, prefix="af3"))

            # Then: optionally pull every numeric value in the summary JSON
            # This is useful because AF3 JSONs can vary slightly
            extra = self._extract_all_numeric_keys(
                flat,
                prefix="af3",
                exclude_substrings=["seed", "sample", "token", "atom", "residue", "chain"]
            )

            # Don't overwrite explicit keys
            for k, v in extra.items():
                out.setdefault(k, v)

        return out

    def _extract_helixfold3_metrics(self, row):
        out = {}
        json_path = row.get("json_path")

        if pd.notna(json_path) and Path(json_path).exists():
            data = self._load_json(Path(json_path))
            flat = self._flatten_dict(data)

            # Do NOT use folder rank here
            keys = [
                "ranking_score",
                "ptm",
                "iptm",
                "plddt",
            ]
            out.update(self._extract_selected_numeric_keys(flat, keys, prefix="helixfold3"))

            extra = self._extract_all_numeric_keys(
                flat,
                prefix="helixfold3",
                exclude_substrings=["seed", "sample", "token", "atom", "residue", "chain", "rank"]
            )

            for k, v in extra.items():
                out.setdefault(k, v)

        return out

    def _extract_chai1_metrics(self, row):
        out = {}
        npz_path = row.get("npz_path")

        if pd.notna(npz_path) and Path(npz_path).exists():
            data = self._load_npz(Path(npz_path))
            for key, value in data.items():
                val = self._safe_float(value)
                if val is not None:
                    out[f"chai1_{key}"] = val

        return out

    def _extract_seedfold_metrics(self, row):
        out = {}
        json_path = row.get("json_path")

        if pd.notna(json_path) and Path(json_path).exists():
            data = self._load_json(Path(json_path))
            flat = self._flatten_dict(data)
            extra = self._extract_all_numeric_keys(flat, prefix="seedfold")
            out.update(extra)

        return out

    def _extract_protenix_metrics(self, row):
        out = {}
        json_path = row.get("json_path")

        if pd.notna(json_path) and Path(json_path).exists():
            data = self._load_json(Path(json_path))
            flat = self._flatten_dict(data)
            extra = self._extract_all_numeric_keys(flat, prefix="protenix")
            out.update(extra)

        return out

    def _extract_of3_metrics(self, row):
        out = {}
        json_path = row.get("json_path")

        if pd.notna(json_path) and Path(json_path).exists():
            data = self._load_json(Path(json_path))
            flat = self._flatten_dict(data)
            extra = self._extract_all_numeric_keys(flat, prefix="of3")
            out.update(extra)

        return out

    def _extract_boltz2_metrics(self, row):
        out = {}

        json_path = row.get("json_path")
        path = row.get("path")

        candidate_jsons = []

        if pd.notna(json_path) and Path(json_path).exists():
            candidate_jsons.append(Path(json_path))

        if pd.notna(path):
            structure_path = Path(path)
            if structure_path.exists():
                candidate_jsons.extend(sorted(structure_path.parent.glob("*.json")))

        seen = set()
        candidate_jsons = [p for p in candidate_jsons if not (str(p) in seen or seen.add(str(p)))]

        for jp in candidate_jsons:
            try:
                data = self._load_json(jp)
                flat = self._flatten_dict(data)

                numeric = self._extract_all_numeric_keys(
                    flat,
                    prefix="boltz2",
                    exclude_substrings=["seed", "sample", "token", "atom", "residue", "chain"]
                )

                if numeric:
                    out.update(numeric)
                    out["boltz2_source_json"] = str(jp)
                    break
            except Exception:
                continue

        return out

    def _extract_of2_metrics(self, row):
        # relaxed pdb only: no native metrics recoverable here
        return {}