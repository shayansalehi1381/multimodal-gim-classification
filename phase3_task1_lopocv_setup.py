"""
Task 1.3 – Leave-One-Patient-Out Cross-Validation (LOPOCV) Setup
=================================================================
Input:
  - data/processed/multimodal_dataset_n17.csv
Output:
  - data/processed/lopocv_splits_n17.pkl
"""

import sys
import os
import pickle

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sklearn.model_selection import LeaveOneGroupOut

print("=" * 70)
print("   Task 1.3 – LOPOCV Pipeline Setup (n=17)")
print("=" * 70)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("data", "processed")
DATA_PATH  = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.csv")
SPLITS_PATH = os.path.join(OUTPUT_DIR, "lopocv_splits_n17.pkl")

print("\n[1/5] Loading multimodal dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")
print(f"      Columns: {df.shape[1]}")

# ── 2. Target Label Vector y ─────────────────────────────────────────────────
print("\n[2/5] Setting up target label vector y...")

# Map: Complete -> 0, Incomplete -> 1
label_map = {'Complete': 0, 'Incomplete': 1}
df['y'] = df['Subtype'].map(label_map)

y = df['y'].values
print(f"      Label mapping: {label_map}")
print(f"      y shape: {y.shape}")
print(f"      Class distribution:")
for label_name, label_val in label_map.items():
    count = (y == label_val).sum()
    print(f"        {label_name} (y={label_val}): {count} patients")

# Verify class balance
n_complete = (y == 0).sum()
n_incomplete = (y == 1).sum()
if n_complete == 8 and n_incomplete == 9:
    print("      ✔ Class balance verified: 8 Complete, 9 Incomplete")
else:
    print(f"      ⚠ Unexpected balance: {n_complete} Complete, {n_incomplete} Incomplete")

# ── 3. Separate Features X ──────────────────────────────────────────────────
print("\n[3/5] Separating feature matrices...")

# Identify column groups
vis_cols = [c for c in df.columns if c.startswith('vis_feat_')]

# Clinical/IEE features = binary IEE features used in the study
iee_cols = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM']

# Additional clinical features (numeric only, exclude identifiers/strings)
extra_clinical = ['Age', 'Subtype_Binary']

# Combine all clinical feature columns
clinical_cols = iee_cols + extra_clinical

# Full feature set = clinical + visual
all_feature_cols = clinical_cols + vis_cols

# Grouping identifier
groups = df['Case Code'].values

# Build feature matrix
X = df[all_feature_cols].values.astype(np.float32)

print(f"      Clinical/IEE features : {len(clinical_cols)} cols  {clinical_cols}")
print(f"      Visual features       : {len(vis_cols)} cols  (vis_feat_0 ... vis_feat_511)")
print(f"      Total feature dims    : {X.shape[1]}")
print(f"      X shape               : {X.shape}")
print(f"      Groups (Case Codes)   : {groups.tolist()}")

# ── 4. LOPOCV Split Generator ───────────────────────────────────────────────
print("\n[4/5] Setting up Leave-One-Patient-Out Cross-Validation...")

logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, groups)
print(f"      Total CV folds: {n_splits}")

if n_splits != 17:
    print(f"      ⚠ Expected 17 folds, got {n_splits}")
else:
    print("      ✔ Exactly 17 folds confirmed (one per patient)")

# Store fold indices and verify
fold_data = []

print("\n      Fold Verification Log:")
print("      " + "-" * 62)
print(f"      {'Fold':>4}  {'Test Case':>10}  {'Test Label':>10}  "
      f"{'Train(C)':>8}  {'Train(I)':>8}  {'Train N':>7}")
print("      " + "-" * 62)

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    test_case = groups[test_idx[0]]
    test_label = 'Complete' if y[test_idx[0]] == 0 else 'Incomplete'
    train_y = y[train_idx]
    n_train_complete = (train_y == 0).sum()
    n_train_incomplete = (train_y == 1).sum()

    print(f"      {fold_idx+1:4d}  {str(test_case):>10s}  {test_label:>10s}  "
          f"{n_train_complete:8d}  {n_train_incomplete:8d}  {len(train_idx):7d}")

    # Verify zero data leakage
    train_cases = set(groups[train_idx])
    test_cases = set(groups[test_idx])
    assert train_cases.isdisjoint(test_cases), \
        f"DATA LEAKAGE in fold {fold_idx+1}! Overlap: {train_cases & test_cases}"

    # Verify sizes
    assert len(test_idx) == 1, f"Test set should have 1 patient, got {len(test_idx)}"
    assert len(train_idx) == 16, f"Train set should have 16 patients, got {len(train_idx)}"

    fold_data.append({
        'fold': fold_idx + 1,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'test_case_code': test_case,
        'test_label': test_label,
    })

print("      " + "-" * 62)
print("      ✔ All 17 folds verified: no data leakage, correct split sizes")

# ── 5. Save Splits ──────────────────────────────────────────────────────────
print(f"\n[5/5] Saving LOPOCV split indices...")

splits_output = {
    'fold_data': fold_data,
    'feature_columns': all_feature_cols,
    'clinical_cols': clinical_cols,
    'visual_cols': vis_cols,
    'label_map': label_map,
    'groups': groups,
    'X': X,
    'y': y,
}

with open(SPLITS_PATH, 'wb') as f:
    pickle.dump(splits_output, f)

size_kb = os.path.getsize(SPLITS_PATH) / 1024
print(f"      Saved: {SPLITS_PATH}  ({size_kb:.1f} KB)")

print("\n" + "=" * 70)
print("🎉 Task 1.3 COMPLETE – LOPOCV pipeline is set up and verified.")
print("=" * 70)
