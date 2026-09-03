"""
Task 3.3 – In-Fold PCA Dimension Reduction Pipeline
=====================================================
Features:
  - Clinical/IEE (7 dims): LBC, MTB, WOS, TVF, MLE, TM, Age
  - Visual (512 dims): vis_feat_0 … vis_feat_511
Target : Subtype  (Complete=0, Incomplete=1)
CV     : Leave-One-Patient-Out (17 folds)
PCA    : Fitted ONLY on train visual features per fold (zero leakage)
Strategies:
  A) Fixed n_components=5
  B) Variance threshold >= 0.90
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
from sklearn.model_selection import LeaveOneGroupOut
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
from sklearn.decomposition import PCA

print("=" * 70)
print("   Task 3.3 – In-Fold PCA Dimension Reduction Pipeline")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.join("data", "processed")
DATA_PATH    = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.pkl")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "infold_pca_summary.csv")

CLINICAL_COLS = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM', 'Age']
LABEL_MAP     = {'Complete': 0, 'Incomplete': 1}

PCA_FIXED_K     = 5       # Strategy A: fixed number of components
PCA_VAR_THRESH  = 0.90    # Strategy B: cumulative variance threshold

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_pickle(DATA_PATH)

vis_cols = sorted([c for c in df.columns if c.startswith('vis_feat_')],
                  key=lambda c: int(c.split('_')[-1]))

X_clin = df[CLINICAL_COLS].values.astype(np.float64)
X_vis  = df[vis_cols].values.astype(np.float64)
y      = df['Subtype'].map(LABEL_MAP).values
groups = df['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True).values

print(f"      Clinical features : {X_clin.shape}")
print(f"      Visual features   : {X_vis.shape}")
print(f"      Target y          : {y.shape}  (Complete=0:{(y==0).sum()}, Incomplete=1:{(y==1).sum()})")
print(f"      Groups            : {len(np.unique(groups))} patients")

# ── 2. LOPOCV In-Fold PCA ────────────────────────────────────────────────────
print("\n[2/4] Running In-Fold PCA across 17 LOPOCV folds...")
print(f"      Strategy A: Fixed PCA (n_components={PCA_FIXED_K})")
print(f"      Strategy B: Variance threshold (>= {PCA_VAR_THRESH*100:.0f}%)")

logo = LeaveOneGroupOut()

summary_rows = []

print("\n      " + "-" * 90)
print(f"      {'Fold':>4}  {'TestCase':>8}  "
      f"{'Strat_A(k)':>10}  {'A_VarExpl':>9}  "
      f"{'Strat_B(k)':>10}  {'B_VarExpl':>9}  "
      f"{'Train_X':>10}  {'Test_X':>10}")
print("      " + "-" * 90)

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_vis, y, groups)):
    test_case = groups[test_idx[0]]

    # ── Clinical features: scale on train only ───────────────────────────
    scaler_clin = StandardScaler()
    X_clin_train = scaler_clin.fit_transform(X_clin[train_idx])
    X_clin_test  = scaler_clin.transform(X_clin[test_idx])

    # ── Visual features: scale on train only ─────────────────────────────
    scaler_vis = StandardScaler()
    X_vis_train_sc = scaler_vis.fit_transform(X_vis[train_idx])
    X_vis_test_sc  = scaler_vis.transform(X_vis[test_idx])

    # ── Strategy A: Fixed n_components ───────────────────────────────────
    pca_a = PCA(n_components=PCA_FIXED_K, random_state=42)
    X_vis_train_a = pca_a.fit_transform(X_vis_train_sc)
    X_vis_test_a  = pca_a.transform(X_vis_test_sc)
    var_expl_a = pca_a.explained_variance_ratio_.sum()

    # Concatenate clinical + reduced visual for Strategy A
    X_train_a = np.hstack([X_clin_train, X_vis_train_a])
    X_test_a  = np.hstack([X_clin_test, X_vis_test_a])

    # ── Strategy B: Variance threshold ───────────────────────────────────
    # Fit full PCA to find how many components capture >= threshold
    max_components = min(X_vis_train_sc.shape[0], X_vis_train_sc.shape[1])
    pca_full = PCA(n_components=max_components, random_state=42)
    pca_full.fit(X_vis_train_sc)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    k_thresh = int(np.searchsorted(cum_var, PCA_VAR_THRESH) + 1)
    k_thresh = max(1, min(k_thresh, max_components))

    pca_b = PCA(n_components=k_thresh, random_state=42)
    X_vis_train_b = pca_b.fit_transform(X_vis_train_sc)
    X_vis_test_b  = pca_b.transform(X_vis_test_sc)
    var_expl_b = pca_b.explained_variance_ratio_.sum()

    # Concatenate clinical + reduced visual for Strategy B
    X_train_b = np.hstack([X_clin_train, X_vis_train_b])
    X_test_b  = np.hstack([X_clin_test, X_vis_test_b])

    print(f"      {fold_idx+1:4d}  {test_case:>8s}  "
          f"{PCA_FIXED_K:>10d}  {var_expl_a:>9.4f}  "
          f"{k_thresh:>10d}  {var_expl_b:>9.4f}  "
          f"{str(X_train_a.shape):>10s}  {str(X_test_a.shape):>10s}")

    summary_rows.append({
        'fold': fold_idx + 1,
        'test_case_code': test_case,
        'true_y': int(y[test_idx[0]]),
        'strategy_A_k': PCA_FIXED_K,
        'strategy_A_var_explained': round(var_expl_a, 4),
        'strategy_A_train_shape': str(X_train_a.shape),
        'strategy_A_test_shape': str(X_test_a.shape),
        'strategy_B_k': k_thresh,
        'strategy_B_var_explained': round(var_expl_b, 4),
        'strategy_B_train_shape': str(X_train_b.shape),
        'strategy_B_test_shape': str(X_test_b.shape),
        'cumulative_var_pc1': round(float(cum_var[0]), 4) if len(cum_var) > 0 else 0,
        'cumulative_var_pc2': round(float(cum_var[1]), 4) if len(cum_var) > 1 else 0,
        'cumulative_var_pc3': round(float(cum_var[2]), 4) if len(cum_var) > 2 else 0,
        'cumulative_var_pc5': round(float(cum_var[4]), 4) if len(cum_var) > 4 else 0,
        'cumulative_var_pc10': round(float(cum_var[9]), 4) if len(cum_var) > 9 else 0,
    })

print("      " + "-" * 90)

# ── 3. Diagnostics ───────────────────────────────────────────────────────────
print("\n[3/4] Diagnostics & Verification...")

df_summary = pd.DataFrame(summary_rows)

# Average variance explained
avg_var_a = df_summary['strategy_A_var_explained'].mean()
avg_var_b = df_summary['strategy_B_var_explained'].mean()
avg_k_b   = df_summary['strategy_B_k'].mean()

print(f"\n      Strategy A (Fixed k={PCA_FIXED_K}):")
print(f"        Avg cumulative variance explained : {avg_var_a:.4f}")
print(f"        Feature dims per fold (train)     : 7 clinical + {PCA_FIXED_K} PCA = {7+PCA_FIXED_K}")

print(f"\n      Strategy B (Variance >= {PCA_VAR_THRESH*100:.0f}%):")
print(f"        Avg components retained           : {avg_k_b:.1f}")
print(f"        Avg cumulative variance explained : {avg_var_b:.4f}")
print(f"        Feature dims per fold (train)     : 7 clinical + ~{int(avg_k_b)} PCA = ~{7+int(avg_k_b)}")

# Cumulative variance profile (averaged across folds)
print(f"\n      Avg Cumulative Variance Explained (across 17 folds):")
print(f"        PC1       : {df_summary['cumulative_var_pc1'].mean():.4f}")
print(f"        PC1–2     : {df_summary['cumulative_var_pc2'].mean():.4f}")
print(f"        PC1–3     : {df_summary['cumulative_var_pc3'].mean():.4f}")
print(f"        PC1–5     : {df_summary['cumulative_var_pc5'].mean():.4f}")
print(f"        PC1–10    : {df_summary['cumulative_var_pc10'].mean():.4f}")

# Zero-leakage verification
print(f"\n      Zero Data Leakage Verification:")
print(f"        ✔ StandardScaler fitted ONLY on train (16 patients) per fold")
print(f"        ✔ PCA fitted ONLY on train visual features per fold")
print(f"        ✔ Test patient projected using fold-specific transforms")
print(f"        ✔ No test statistics influence scaler means/stds or PCA loadings")

# ── 4. Save Summary ─────────────────────────────────────────────────────────
print(f"\n[4/4] Saving summary...")
df_summary.to_csv(SUMMARY_PATH, index=False)
print(f"      Saved: {SUMMARY_PATH}")

print("\n" + "=" * 70)
print("🎉 Task 3.3 COMPLETE – In-Fold PCA pipeline validated.")
print("=" * 70)
