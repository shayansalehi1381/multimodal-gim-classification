"""
Task 5.3 – Sub-cohort Exploratory Analysis on OLGIM Stages
=============================================================
Input:
  - data/processed/cleaned_metadata_n17.xlsx
  - data/processed/multimodal_classifier_predictions.csv
Output:
  - data/processed/olgim_subcohort_analysis.csv
"""

import sys
import os
import warnings

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

print("=" * 70)
print("   Task 5.3 – OLGIM Sub-cohort Exploratory Analysis")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("data", "processed")
CLIN_PATH  = os.path.join(OUTPUT_DIR, "cleaned_metadata_n17.xlsx")
PRED_PATH  = os.path.join(OUTPUT_DIR, "multimodal_classifier_predictions.csv")
OUT_PATH   = os.path.join(OUTPUT_DIR, "olgim_subcohort_analysis.csv")

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")
df_clin = pd.read_excel(CLIN_PATH)
df_preds = pd.read_csv(PRED_PATH)

df_clin['Case Code'] = df_clin['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True)
df_preds['test_case_code'] = df_preds['test_case_code'].astype(str).str.replace(r'\.0$', '', regex=True)

print(f"      Clinical metadata : {df_clin.shape}")
print(f"      All predictions   : {df_preds.shape}")

# ── 2. Isolate OLGIM Sub-cohort ──────────────────────────────────────────────
print("\n[2/5] Isolating patients with valid OLGIM stages...")

df_olgim = df_clin[df_clin['OLGIM'].notna()].copy()
n_olgim = len(df_olgim)
print(f"      Patients with valid OLGIM: {n_olgim}")

# Map OLGIM to risk category
# Low Risk: Stage 0, I, II  |  High Risk: Stage III, IV
def olgim_risk(stage):
    if pd.isna(stage):
        return None
    s = int(stage)
    if s <= 2:
        return 'Low Risk'
    else:
        return 'High Risk'

df_olgim['OLGIM_Stage'] = df_olgim['OLGIM'].astype(int)
df_olgim['OLGIM_Risk'] = df_olgim['OLGIM'].apply(olgim_risk)

print(f"\n      OLGIM Stage Distribution:")
for stage in sorted(df_olgim['OLGIM_Stage'].unique()):
    count = (df_olgim['OLGIM_Stage'] == stage).sum()
    risk = olgim_risk(stage)
    print(f"        Stage {stage} ({risk}): {count} patients")

print(f"\n      Risk Category Distribution:")
print(f"        Low Risk  (0-II)  : {(df_olgim['OLGIM_Risk'] == 'Low Risk').sum()}")
print(f"        High Risk (III-IV): {(df_olgim['OLGIM_Risk'] == 'High Risk').sum()}")

# ── 3. Cross-tabulation: OLGIM × Subtype ────────────────────────────────────
print("\n[3/5] Cross-tabulation: OLGIM Stage × GIM Subtype...")

olgim_cases = df_olgim['Case Code'].tolist()

print(f"\n      {'Case':>6}  {'OLGIM':>5}  {'Risk':>10}  {'Subtype':>10}")
print(f"      " + "-" * 40)
for _, row in df_olgim.iterrows():
    print(f"      {row['Case Code']:>6}  {int(row['OLGIM']):>5}  "
          f"{row['OLGIM_Risk']:>10}  {row['Subtype']:>10}")

# Cross-tab
print(f"\n      Cross-tabulation (OLGIM Risk × Subtype):")
ct = pd.crosstab(df_olgim['OLGIM_Risk'], df_olgim['Subtype'], margins=True)
print(f"\n{ct.to_string()}")

# Concordance analysis
print(f"\n      Concordance Analysis:")
# High Risk OLGIM → Incomplete?
high_risk = df_olgim[df_olgim['OLGIM_Risk'] == 'High Risk']
if len(high_risk) > 0:
    n_hr_incomplete = (high_risk['Subtype'] == 'Incomplete').sum()
    print(f"        High-Risk OLGIM patients: {len(high_risk)}")
    print(f"        Of those, Incomplete subtype: {n_hr_incomplete}/{len(high_risk)} "
          f"({n_hr_incomplete/len(high_risk)*100:.0f}%)")
else:
    print(f"        No High-Risk OLGIM patients found")

low_risk = df_olgim[df_olgim['OLGIM_Risk'] == 'Low Risk']
if len(low_risk) > 0:
    n_lr_incomplete = (low_risk['Subtype'] == 'Incomplete').sum()
    n_lr_complete = (low_risk['Subtype'] == 'Complete').sum()
    print(f"        Low-Risk OLGIM patients: {len(low_risk)}")
    print(f"        Of those, Complete subtype: {n_lr_complete}/{len(low_risk)} "
          f"({n_lr_complete/len(low_risk)*100:.0f}%)")
    print(f"        Of those, Incomplete subtype: {n_lr_incomplete}/{len(low_risk)} "
          f"({n_lr_incomplete/len(low_risk)*100:.0f}%)")

# ── 4. RF Model Performance on OLGIM Sub-cohort ─────────────────────────────
print(f"\n[4/5] Random Forest performance on OLGIM sub-cohort (n={n_olgim})...")

# Get RF predictions for OLGIM patients only
df_rf = df_preds[df_preds['model'] == 'RandomForest'].copy()
df_rf_olgim = df_rf[df_rf['test_case_code'].isin(olgim_cases)].copy()

# Merge OLGIM info
df_rf_olgim = df_rf_olgim.merge(
    df_olgim[['Case Code', 'OLGIM_Stage', 'OLGIM_Risk']],
    left_on='test_case_code', right_on='Case Code', how='left'
)

print(f"\n      {'Case':>6}  {'OLGIM':>5}  {'Risk':>10}  {'True':>10}  "
      f"{'Pred':>10}  {'P(Inc)':>7}  {'':>3}")
print(f"      " + "-" * 65)
for _, row in df_rf_olgim.iterrows():
    mark = '✔' if row['correct'] else '✘'
    print(f"      {row['test_case_code']:>6}  {int(row['OLGIM_Stage']):>5}  "
          f"{row['OLGIM_Risk']:>10}  {row['true_label']:>10}  "
          f"{row['pred_label']:>10}  {row['prob_incomplete']:>7.4f}  {mark:>3}")

# Compute metrics on sub-cohort
y_true_sub = df_rf_olgim['true_y'].values
y_pred_sub = df_rf_olgim['pred_y'].values
y_prob_sub = df_rf_olgim['prob_incomplete'].values

acc_sub = accuracy_score(y_true_sub, y_pred_sub)
n_correct = int((y_true_sub == y_pred_sub).sum())

# Check if both classes are present in sub-cohort predictions
unique_true = np.unique(y_true_sub)
unique_pred = np.unique(y_pred_sub)

print(f"\n      Sub-cohort Metrics (RF, n={n_olgim}):")
print(f"        Accuracy      : {acc_sub:.4f}  ({n_correct}/{n_olgim})")

if len(unique_true) > 1:
    sens_sub = recall_score(y_true_sub, y_pred_sub, pos_label=1)
    spec_sub = recall_score(y_true_sub, y_pred_sub, pos_label=0)
    f1_sub   = f1_score(y_true_sub, y_pred_sub)
    print(f"        Sensitivity   : {sens_sub:.4f}  (Recall for Incomplete)")
    print(f"        Specificity   : {spec_sub:.4f}  (Recall for Complete)")
    print(f"        F1-Score      : {f1_sub:.4f}")
else:
    print(f"        [Note] Only one true class in sub-cohort; Sens/Spec/F1 not meaningful")

# Concordance: Does the model's prediction align with OLGIM risk?
print(f"\n      Prediction–OLGIM Concordance:")
for _, row in df_rf_olgim.iterrows():
    olgim_suggests = 'Incomplete' if row['OLGIM_Risk'] == 'High Risk' else 'ambiguous'
    model_pred = row['pred_label']
    true_label = row['true_label']
    concordant = '—'
    if row['OLGIM_Risk'] == 'High Risk':
        concordant = '✔' if model_pred == 'Incomplete' else '✘'
    print(f"        Case {row['test_case_code']}: OLGIM={int(row['OLGIM_Stage'])} "
          f"({row['OLGIM_Risk']}), Pred={model_pred}, True={true_label}  {concordant}")

# ── 5. Export ────────────────────────────────────────────────────────────────
print(f"\n[5/5] Saving sub-cohort analysis...")

export_cols = ['test_case_code', 'OLGIM_Stage', 'OLGIM_Risk',
               'true_label', 'pred_label', 'prob_incomplete', 'correct']
df_export = df_rf_olgim[export_cols].copy()
df_export.to_csv(OUT_PATH, index=False)
print(f"      Saved: {OUT_PATH}")

print("\n" + "=" * 70)
print(f"🎉 Task 5.3 COMPLETE – OLGIM sub-cohort analysis (n={n_olgim}) done.")
print("=" * 70)
