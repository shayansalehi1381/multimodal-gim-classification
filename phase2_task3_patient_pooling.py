"""
Task 3.2 – Patient-Level Mean-Pooling
=================================================
Input:
  - data/processed/frame_embeddings_1976x512.npy
  - data/processed/frame_metadata.csv
  - data/processed/cleaned_metadata_n17.xlsx (contains Case Code)
Output:
  - data/processed/patient_visual_features_n17.npy
  - data/processed/patient_visual_features_n17.csv
"""

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

print("=" * 70)
print("   Task 3.2 – Patient-Level Mean-Pooling (n=17)")
print("=" * 70)

# Paths
OUTPUT_DIR = os.path.join("data", "processed")
EMB_PATH   = os.path.join(OUTPUT_DIR, "frame_embeddings_1976x512.npy")
META_PATH  = os.path.join(OUTPUT_DIR, "frame_metadata.csv")
CLIN_PATH  = os.path.join(OUTPUT_DIR, "cleaned_metadata_n17.xlsx")
OUT_NPY    = os.path.join(OUTPUT_DIR, "patient_visual_features_n17.npy")
OUT_CSV    = os.path.join(OUTPUT_DIR, "patient_visual_features_n17.csv")

# 1. Load Data
print("[1/5] Loading embeddings and metadata...")
frame_embs = np.load(EMB_PATH)
frame_meta = pd.read_csv(META_PATH)
print(f"      Frame embeddings : {frame_embs.shape}")
print(f"      Frame metadata   : {frame_meta.shape}")

print("[2/5] Loading clinical metadata (n=17)...")
clinical_meta = pd.read_excel(CLIN_PATH)

# We want to match case_code between metadata (where they could be string)
# and clinical metadata (where it might be float/int)
valid_cases = clinical_meta['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True).tolist()
print(f"      Found {len(valid_cases)} valid patient case codes.")

# 2. Extract Valid Patient Frames
print("[3/5] Performing Patient-Level Mean-Pooling...")
frame_meta['case_code'] = frame_meta['case_code'].astype(str).str.replace(r'\.0$', '', regex=True)

patient_embs = []
patient_codes = []

missing_cases = []

for case in valid_cases:
    # Get frame indices for this case
    case_indices = frame_meta[frame_meta['case_code'] == case]['embedding_idx'].values
    if len(case_indices) == 0:
        missing_cases.append(case)
        continue
    
    # Retrieve embeddings
    case_frame_embs = frame_embs[case_indices] # shape: (num_frames, 512)
    
    # Compute mean across frames
    mean_emb = np.mean(case_frame_embs, axis=0) # shape: (512,)
    
    patient_embs.append(mean_emb)
    patient_codes.append(case)
    
if missing_cases:
    print(f"      WARNING: {len(missing_cases)} cases had no frames: {missing_cases}")

patient_embs_np = np.array(patient_embs, dtype=np.float32)
print(f"      Pooled matrix shape before norm: {patient_embs_np.shape}")

# 3. L2 Normalization
print("[4/5] Applying L2 Normalization to patient-level vectors...")
patient_embs_tensor = torch.tensor(patient_embs_np)
patient_embs_norm = F.normalize(patient_embs_tensor, p=2, dim=1).numpy()

print(f"      Pooled matrix shape after norm : {patient_embs_norm.shape}")

# 4. Save Outputs
print("[5/5] Saving outputs...")
np.save(OUT_NPY, patient_embs_norm)
print(f"      Saved NumPy array : {OUT_NPY}")

# Convert to dataframe for CSV
cols = [f"feat_{i}" for i in range(patient_embs_norm.shape[1])]
df_out = pd.DataFrame(patient_embs_norm, columns=cols)
df_out.insert(0, "Case Code", patient_codes)

df_out.to_csv(OUT_CSV, index=False)
print(f"      Saved CSV file    : {OUT_CSV}")

# Verification
print("\n--- Verification ---")
print(f"  Final Matrix Shape : {patient_embs_norm.shape}")
print(f"  Final dtype        : {patient_embs_norm.dtype}")
print(f"  L2 Norm (row 0)    : {np.linalg.norm(patient_embs_norm[0]):.6f} (expected ~1.0)")
print(f"  Number of Patients : {len(patient_codes)}")
if len(patient_codes) == 17:
    print("  ✔ Exact count (n=17) matched!")
else:
    print(f"  ⚠ Count mismatch: Expected 17, Got {len(patient_codes)}")

print("\n" + "=" * 70)
print("🎉 Task 3.2 COMPLETE – Patient-level pooling finished.")
print("=" * 70)
