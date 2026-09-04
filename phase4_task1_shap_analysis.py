"""
Task 1.4 – Clinical Explainability (SHAP Values Computation)
============================================================
Input:
  - data/processed/multimodal_dataset_n17.csv
Output:
  - data/processed/shap_feature_importance.csv
  - data/processed/shap_values_cache.pkl
"""

import sys
import os
import warnings
import pickle

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
import shap

print("=" * 70)
print("   Task 1.4 – Clinical Explainability (SHAP Values)")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("data", "processed")
DATA_PATH  = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.csv")
CSV_OUT    = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
PKL_OUT    = os.path.join(OUTPUT_DIR, "shap_values_cache.pkl")

CLINICAL_COLS = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM', 'Age']
PCA_K = 5
LABEL_MAP = {'Complete': 0, 'Incomplete': 1}
RANDOM_STATE = 42

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading multimodal dataset...")
df = pd.read_csv(DATA_PATH)

vis_cols = sorted([c for c in df.columns if c.startswith('vis_feat_')],
                  key=lambda c: int(c.split('_')[-1]))

X_clin = df[CLINICAL_COLS].values.astype(np.float64)
X_vis  = df[vis_cols].values.astype(np.float64)
y      = df['Subtype'].map(LABEL_MAP).values
groups = df['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True).values

multimodal_feature_names = CLINICAL_COLS + [f"Visual_PC{i+1}" for i in range(PCA_K)]

print(f"      Clinical features : {len(CLINICAL_COLS)}")
print(f"      Visual components : {PCA_K}")
print(f"      Total features    : {len(multimodal_feature_names)}")

# ── 2. Compute SHAP Values (LOPOCV) ──────────────────────────────────────────
print(f"\n[2/4] Computing SHAP values across 17 LOPOCV folds...")

logo = LeaveOneGroupOut()

# To accumulate SHAP values for the whole dataset (17 test samples)
# We will store them aligned with the original dataframe order
shap_values_lr_all = np.zeros((len(df), len(CLINICAL_COLS)))
shap_values_rf_all = np.zeros((len(df), len(multimodal_feature_names)))
test_X_clin_all = np.zeros((len(df), len(CLINICAL_COLS)))
test_X_multi_all = np.zeros((len(df), len(multimodal_feature_names)))

for train_idx, test_idx in logo.split(X_vis, y, groups):
    y_train, y_test = y[train_idx], y[test_idx]
    
    # ── In-Fold Preprocessing ────────────────────────────────────────────
    # Clinical scaling
    scaler_clin = StandardScaler()
    X_clin_tr = scaler_clin.fit_transform(X_clin[train_idx])
    X_clin_te = scaler_clin.transform(X_clin[test_idx])
    
    # Visual scaling + PCA
    scaler_vis = StandardScaler()
    X_vis_tr_sc = scaler_vis.fit_transform(X_vis[train_idx])
    X_vis_te_sc = scaler_vis.transform(X_vis[test_idx])
    
    pca = PCA(n_components=PCA_K, random_state=RANDOM_STATE)
    X_vis_tr_pca = pca.fit_transform(X_vis_tr_sc)
    X_vis_te_pca = pca.transform(X_vis_te_sc)
    
    X_multi_tr = np.hstack([X_clin_tr, X_vis_tr_pca])
    X_multi_te = np.hstack([X_clin_te, X_vis_te_pca])
    
    # Store transformed test features for global SHAP analysis (optional)
    test_X_clin_all[test_idx] = X_clin_te
    test_X_multi_all[test_idx] = X_multi_te

    # ── Model 1: Baseline LR (Clinical Only) ─────────────────────────────
    lr = LogisticRegression(penalty='l2', solver='lbfgs', random_state=RANDOM_STATE)
    lr.fit(X_clin_tr, y_train)
    
    # For Logistic Regression, shap.LinearExplainer works well
    explainer_lr = shap.LinearExplainer(lr, X_clin_tr)
    shap_vals_lr = explainer_lr.shap_values(X_clin_te)
    shap_values_lr_all[test_idx] = shap_vals_lr
    
    # ── Model 2: RandomForest (Multimodal) ───────────────────────────────
    rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE)
    rf.fit(X_multi_tr, y_train)
    
    # For Random Forest, shap.TreeExplainer is preferred
    explainer_rf = shap.TreeExplainer(rf)
    shap_vals_rf = explainer_rf.shap_values(X_multi_te)
    
    # TreeExplainer for classification returns a list of arrays (one per class). 
    # We want the SHAP values for the positive class (class 1).
    if isinstance(shap_vals_rf, list):
        shap_vals_rf_class1 = shap_vals_rf[1]
    elif len(shap_vals_rf.shape) == 3:
        # Some versions of shap return shape (n_samples, n_features, n_classes)
        shap_vals_rf_class1 = shap_vals_rf[:, :, 1]
    else:
        shap_vals_rf_class1 = shap_vals_rf
        
    shap_values_rf_all[test_idx] = shap_vals_rf_class1

print("      ✔ SHAP values computed for all 17 folds.")

# ── 3. Feature Importance Calculation ────────────────────────────────────────
print("\n[3/4] Calculating mean absolute SHAP values...")

# Baseline LR
mean_abs_shap_lr = np.mean(np.abs(shap_values_lr_all), axis=0)
df_imp_lr = pd.DataFrame({
    'Feature': CLINICAL_COLS,
    'Mean_Abs_SHAP': mean_abs_shap_lr,
    'Model': 'L2-LogReg (Clinical)'
}).sort_values('Mean_Abs_SHAP', ascending=False)
df_imp_lr['Rank'] = range(1, len(df_imp_lr) + 1)

# RandomForest Multimodal
mean_abs_shap_rf = np.mean(np.abs(shap_values_rf_all), axis=0)
df_imp_rf = pd.DataFrame({
    'Feature': multimodal_feature_names,
    'Mean_Abs_SHAP': mean_abs_shap_rf,
    'Model': 'RandomForest (Multimodal)'
}).sort_values('Mean_Abs_SHAP', ascending=False)
df_imp_rf['Rank'] = range(1, len(df_imp_rf) + 1)

# Combine
df_importance = pd.concat([df_imp_lr, df_imp_rf], ignore_index=True)

# Print Tables
print(f"\n      L2-LogReg (Clinical-Only) Feature Importance:")
print(f"      ┌{'─'*38}┐")
print(f"      │ {'Rank':<4s} │ {'Feature':<15s} │ {'|SHAP|':<10s} │")
print(f"      ├{'─'*38}┤")
for _, row in df_imp_lr.iterrows():
    print(f"      │ {int(row['Rank']):<4d} │ {row['Feature']:<15s} │ {row['Mean_Abs_SHAP']:<10.4f} │")
print(f"      └{'─'*38}┘")

print(f"\n      RandomForest (Multimodal) Feature Importance:")
print(f"      ┌{'─'*38}┐")
print(f"      │ {'Rank':<4s} │ {'Feature':<15s} │ {'|SHAP|':<10s} │")
print(f"      ├{'─'*38}┤")
for _, row in df_imp_rf.iterrows():
    # Highlight visual features
    is_visual = "Visual" in row['Feature']
    marker = "*" if is_visual else " "
    print(f"      │ {int(row['Rank']):<4d} │ {row['Feature']+marker:<15s} │ {row['Mean_Abs_SHAP']:<10.4f} │")
print(f"      └{'─'*38}┘")
print("        * Visual PCA Component")

# Analyze relative contribution of Visual vs Clinical in RF
visual_shap = df_imp_rf[df_imp_rf['Feature'].str.contains('Visual')]['Mean_Abs_SHAP'].sum()
clinical_shap = df_imp_rf[~df_imp_rf['Feature'].str.contains('Visual')]['Mean_Abs_SHAP'].sum()
total_shap = visual_shap + clinical_shap

print(f"\n      Relative Contribution in Multimodal Model:")
print(f"        Visual Features   : {visual_shap/total_shap*100:.1f}%")
print(f"        Clinical Features : {clinical_shap/total_shap*100:.1f}%")

# ── 4. Save Outputs ──────────────────────────────────────────────────────────
print("\n[4/4] Saving SHAP outputs...")
df_importance.to_csv(CSV_OUT, index=False)

cache_data = {
    'lr': {
        'shap_values': shap_values_lr_all,
        'features': test_X_clin_all,
        'feature_names': CLINICAL_COLS
    },
    'rf': {
        'shap_values': shap_values_rf_all,
        'features': test_X_multi_all,
        'feature_names': multimodal_feature_names
    }
}
with open(PKL_OUT, 'wb') as f:
    pickle.dump(cache_data, f)

print(f"      Feature Rankings : {CSV_OUT}")
print(f"      SHAP Cache       : {PKL_OUT}")

print("\n" + "=" * 70)
print("🎉 Task 1.4 COMPLETE – SHAP values computed and ranked.")
print("=" * 70)
