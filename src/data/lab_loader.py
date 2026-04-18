# src/data/lab_loader.py
"""
Merge NHANES .xpt panels from several cycles into one table (one row per participant).

Usage:
    from src.data.lab_loader import load_nhanes
    features_df, full_df, feature_cols = load_nhanes()
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Tuple


def _load_cycle(
    cycle: dict,
    data_dir: Path,
    file_bases: dict,
    all_feature_cols: List[str],
    validation_file_bases: dict = None,
    validation_columns: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and merge all panels for a single NHANES cycle.

    Returns (lab_df, demo_df, val_df):
      lab_df  — training feature columns only
      demo_df — age/sex/BMI/ethnicity for context
      val_df  — validation-only columns (HbA1c, diagnosis flags) — never in features
    """
    year   = cycle["year"]
    prefix = cycle.get("prefix", "")
    suffix = cycle.get("suffix", "")

    dfs = {}
    for panel_name, base in file_bases.items():
        path = data_dir / f"{prefix}{base}{suffix}.xpt"
        if not path.exists():
            continue
        df = pd.read_sas(str(path))
        df.columns = [c.upper() for c in df.columns]
        df["record_id"] = year + "_" + df["SEQN"].astype(int).astype(str)
        df["cycle_year"] = year
        dfs[panel_name] = df

    if not dfs:
        return None, None, None

    loaded = sorted(dfs.keys())
    print(f"    {year}:  {', '.join(f'{k}({dfs[k].shape[0]:,})' for k in loaded)}")

    # Merge lab panels on record_id (training features only)
    lab_df = None
    for panel_name, df in dfs.items():
        if panel_name == "demographics":
            continue
        keep = ["record_id", "cycle_year"] + [
            c for c in all_feature_cols if c in df.columns
        ]
        sub = df[keep]
        lab_df = sub if lab_df is None else lab_df.merge(sub, on=["record_id", "cycle_year"], how="outer")

    # Demographics
    demo_df = None
    if "demographics" in dfs:
        demo_cols = ["record_id", "cycle_year", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI"]
        demo_cols = [c for c in demo_cols if c in dfs["demographics"].columns]
        demo_df = dfs["demographics"][demo_cols].copy()

    # Validation-only files (HbA1c, diagnosis labels) — never merged into features
    val_df = None
    if validation_file_bases and validation_columns:
        for file_key, base in validation_file_bases.items():
            path = data_dir / f"{prefix}{base}{suffix}.xpt"
            if not path.exists():
                continue
            vdf = pd.read_sas(str(path))
            vdf.columns = [c.upper() for c in vdf.columns]
            vdf["record_id"] = year + "_" + vdf["SEQN"].astype(int).astype(str)
            cols_to_keep = ["record_id"] + [
                c for c in validation_columns.get(file_key, {}).keys()
                if c in vdf.columns
            ]
            vdf = vdf[cols_to_keep]
            val_df = vdf if val_df is None else val_df.merge(vdf, on="record_id", how="outer")

    return lab_df, demo_df, val_df


def load_nhanes(
    config_path: str = "configs/config_lab.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load and merge all configured NHANES cycles into one DataFrame.

    Returns:
        features_df  — record_id + cycle_year + lab feature columns  (N × M+2)
        full_df      — features_df + demographics columns (for validation only)
        feature_cols — ordered list of feature column names
    """
    cfg = yaml.safe_load(open(config_path))
    dcfg = cfg["lab_data"]
    data_dir = Path(dcfg["data_dir"])
    min_valid_ratio = dcfg.get("min_valid_ratio", 0.70)
    cycles = dcfg["cycles"]
    file_bases = dcfg["file_bases"]
    validation_file_bases = dcfg.get("validation_files", {})
    validation_columns    = dcfg.get("validation_columns", {})

    # All feature columns defined across panels
    all_feature_cols = []
    for panel in dcfg["panels"].values():
        all_feature_cols.extend(list(panel["features"].keys()))

    print(f"Loading {len(cycles)} NHANES cycle(s) from {data_dir}…")

    all_lab, all_demo, all_val = [], [], []
    for cycle in cycles:
        lab_df, demo_df, val_df = _load_cycle(
            cycle, data_dir, file_bases, all_feature_cols,
            validation_file_bases, validation_columns,
        )
        if lab_df is not None:
            all_lab.append(lab_df)
        if demo_df is not None:
            all_demo.append(demo_df)
        if val_df is not None:
            all_val.append(val_df)

    if not all_lab:
        abs_dir = data_dir.resolve()
        hint = (
            f"Nothing loaded from:\n  {abs_dir}\n\n"
            "Names should match config_lab.yaml (e.g. CBC_J.xpt / BIOPRO_J.xpt for the 2017–18 cycle). "
            "Generate curls with:\n  python experiments/run_lab_pipeline.py --stage download\n"
            "then save files under that directory.\n"
        )
        if data_dir.exists():
            xpts = list(data_dir.glob("*.xpt")) + list(data_dir.glob("*.XPT"))
            if xpts:
                hint += f"\nFound these (if names differ from the config, fix or trim cycles): {[p.name for p in xpts[:15]]}"
            else:
                hint += "\nFolder is empty or has no .xpt files."
        else:
            hint += f"\nmkdir -p {abs_dir}"
        raise FileNotFoundError(hint)

    merged     = pd.concat(all_lab, ignore_index=True)
    demo_merged = pd.concat(all_demo, ignore_index=True) if all_demo else None
    val_merged  = pd.concat(all_val,  ignore_index=True) if all_val  else None

    # Only keep feature columns actually present
    feature_cols = [c for c in all_feature_cols if c in merged.columns]
    missing_cols = set(all_feature_cols) - set(feature_cols)
    if missing_cols:
        print(f"\n  ⚠  {len(missing_cols)} expected columns not found: {sorted(missing_cols)[:5]}…")

    # Merge demographics + validation labels into full_df
    full_df = merged.copy()
    if demo_merged is not None:
        full_df = full_df.merge(demo_merged, on=["record_id", "cycle_year"], how="left")
    if val_merged is not None:
        full_df = full_df.merge(val_merged, on="record_id", how="left")

    # Drop records missing too many feature values
    n_before = len(merged)
    valid_ratio = merged[feature_cols].notna().mean(axis=1)
    keep_mask = valid_ratio >= min_valid_ratio
    merged = merged[keep_mask].reset_index(drop=True)
    full_df = full_df[keep_mask].reset_index(drop=True)
    print(f"\n  Dropped {n_before - len(merged):,} records with >{(1-min_valid_ratio)*100:.0f}% missing")

    # Impute remaining missing values with column medians (per cycle to avoid leakage)
    imputed = 0
    for col in feature_cols:
        n_miss = merged[col].isna().sum()
        if n_miss > 0:
            median_val = merged[col].median()
            merged[col] = merged[col].fillna(median_val)
            full_df[col] = full_df[col].fillna(median_val)
            imputed += n_miss
    if imputed:
        print(f"  Imputed {imputed:,} remaining missing values with column medians")

    features_df = merged[["record_id", "cycle_year"] + feature_cols].copy()

    # Summary
    print(f"\n  ✓  Final dataset: {len(features_df):,} records × {len(feature_cols)} features")
    for yr, grp in features_df.groupby("cycle_year"):
        print(f"     {yr}: {len(grp):,} records")
    for panel_name, panel_cfg in dcfg["panels"].items():
        n = len([c for c in panel_cfg["features"] if c in feature_cols])
        print(f"     panel {panel_name}: {n} features")

    return features_df, full_df, feature_cols


def get_feature_labels(config_path: str = "configs/config_lab.yaml") -> dict:
    cfg = yaml.safe_load(open(config_path))
    labels = {}
    for panel in cfg["lab_data"]["panels"].values():
        labels.update(panel["features"])
    return labels


def get_clinical_ranges(config_path: str = "configs/config_lab.yaml") -> dict:
    cfg = yaml.safe_load(open(config_path))
    return cfg["embedding"]["clinical_ranges"]
