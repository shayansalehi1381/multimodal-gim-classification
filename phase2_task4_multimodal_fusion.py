"""
Task 4.2 – Multimodal Fusion
=================================================
Input:
  - data/processed/patient_visual_features_n17.csv (17x513)
  - data/processed/cleaned_metadata_n17.xlsx
Output:
  - data/processed/multimodal_dataset_n17.csv
  - data/processed/multimodal_dataset_n17.pkl
"""

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

print("=" * 70)
print("   Task 4.2 – Multimodal Fusion (n=17)")
print("=" * 70)

# Paths
OUTPUT_DIR = os.path.join("data", "processed")
VIS_CSV    = os.path.join(OUTPUT_DIR, "patient_visual_features_n17.csv")
CLIN_PATH  = os.path.join(OUTPUT_DIR, "cleaned_metadata_n17.xlsx")
OUT_CSV    = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.csv")
OUT_PKL    = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.pkl")

# 1. Load Data
print("[1/4] Loading visual and clinical features...")
try:
    df_vis = pd.read_csv(VIS_CSV)
    df_clin = pd.read_excel(CLIN_PATH)
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

# Ensure Case Code is standard string format in both to avoid mismatches
df_vis['Case Code'] = df_vis['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True)
df_clin['Case Code'] = df_clin['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True)

# Handle NaNs in clinical metadata (so ML models don't crash)
for col in df_clin.columns:
    if df_clin[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_clin[col]):
            df_clin[col] = df_clin[col].fillna(0)  # Default fallback for missing numerics
        else:
            df_clin[col] = df_clin[col].fillna("Unknown")

# Rename visual feature columns to explicitly start with 'vis_feat_' if they don't already
rename_map = {col: col.replace('feat_', 'vis_feat_') for col in df_vis.columns if col.startswith('feat_')}
df_vis.rename(columns=rename_map, inplace=True)

print(f"      Visual Features Dataframe : {df_vis.shape}")
print(f"      Clinical Metadata Dataframe: {df_clin.shape}")

# 2. Merge Data
print("[2/4] Performing Multimodal Fusion (Merge by Case Code)...")
df_merged = pd.merge(df_clin, df_vis, on="Case Code", how="inner")

print(f"      Merged Dataframe Shape: {df_merged.shape}")

# 3. Validation
print("[3/4] Validating fused dataset...")
n_rows, n_cols = df_merged.shape

if n_rows != 17:
    print(f"      [WARNING] Shape mismatch! Expected 17 rows, got {n_rows}")
else:
    print("      ✔ Row count is exactly 17.")

null_counts = df_merged.isnull().sum().sum()
if null_counts > 0:
    print(f"      [WARNING] Found {null_counts} NaN/missing values across the dataset.")
else:
    print("      ✔ No NaN or missing values found in the merged dataset.")

# Target variable distribution
label_col = 'Subtype'
if label_col in df_merged.columns:
    dist = df_merged[label_col].value_counts().to_dict()
    print(f"      Target Distribution ('{label_col}'): {dist}")
else:
    print(f"      [WARNING] Target column '{label_col}' not found.")

# 4. Save Outputs
print("[4/4] Saving fused multimodal dataset...")
df_merged.to_csv(OUT_CSV, index=False)
df_merged.to_pickle(OUT_PKL)
print(f"      Saved CSV : {OUT_CSV}")
print(f"      Saved PKL : {OUT_PKL}")

# Summary Output
print("\n--- Summary ---")
print(f"  Total Rows    : {n_rows}")
print(f"  Total Columns : {n_cols}")
print(f"  Metadata Cols : {df_clin.shape[1]}")
print(f"  Visual Cols   : {df_vis.shape[1] - 1}") # subtract Case Code

vis_cols = [c for c in df_merged.columns if c.startswith('vis_feat_')]
print(f"  Visual Feat Count: {len(vis_cols)} (e.g., {vis_cols[0]}, {vis_cols[1]}, ..., {vis_cols[-1]})")

print("\n" + "=" * 70)
print("🎉 Task 4.2 COMPLETE – Multimodal fusion successful.")
print("=" * 70)
