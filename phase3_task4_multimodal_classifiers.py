"""
Task 4.3 – Multimodal Classification Modeling (LOPOCV)
=======================================================
Features:
  - Clinical/IEE (7 dims): LBC, MTB, WOS, TVF, MLE, TM, Age
  - Visual PCA (5 dims): In-fold PCA of 512-dim BiomedCLIP features
  - Total: 12-dim multimodal vector
Target : Subtype  (Complete=0, Incomplete=1)
CV     : Leave-One-Patient-Out (17 folds)
Models : Random Forest, SVM (RBF), Gradient Boosting
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
# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# pyrefly: ignore [missing-import]
from sklearn.svm import SVC
# pyrefly: ignore [missing-import]
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

print("=" * 70)
print("   Task 4.3 – Multimodal Classification Modeling (LOPOCV)")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("data", "processed")
DATA_PATH  = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.pkl")
PRED_PATH  = os.path.join(OUTPUT_DIR, "multimodal_classifier_predictions.csv")
COMP_PATH  = os.path.join(OUTPUT_DIR, "multimodal_model_comparison.csv")

CLINICAL_COLS = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM', 'Age']
LABEL_MAP     = {'Complete': 0, 'Incomplete': 1}
PCA_K         = 5
RANDOM_STATE  = 42

# Baseline from Task 2.3
BASELINE = {
    'model': 'L2-LogReg (Clinical Only)',
    'accuracy': 0.3529, 'sensitivity': 0.6667, 'specificity': 0.0000,
    'f1': 0.5217, 'roc_auc': 0.5000
}

# ── Model definitions ────────────────────────────────────────────────────────
MODELS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=50, max_depth=3, random_state=RANDOM_STATE
    ),
    'SVM_RBF': SVC(
        kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
    ),
}

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
print(f"      PCA components    : {PCA_K}")
print(f"      Final multimodal  : {len(CLINICAL_COLS) + PCA_K} dims")
print(f"      Target y          : Complete(0)={int((y==0).sum())}, Incomplete(1)={int((y==1).sum())}")
print(f"      Models            : {list(MODELS.keys())}")

# ── 2. LOPOCV Training ──────────────────────────────────────────────────────
print("\n[2/4] Running 17-fold LOPOCV with In-Fold PCA...")

logo = LeaveOneGroupOut()
all_predictions = []   # store per-fold, per-model predictions

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_vis, y, groups)):
    test_case = groups[test_idx[0]]
    y_train, y_test = y[train_idx], y[test_idx]

    # ── In-Fold Preprocessing ────────────────────────────────────────────
    # Clinical: scale on train
    scaler_clin = StandardScaler()
    X_clin_tr = scaler_clin.fit_transform(X_clin[train_idx])
    X_clin_te = scaler_clin.transform(X_clin[test_idx])

    # Visual: scale + PCA on train
    scaler_vis = StandardScaler()
    X_vis_tr_sc = scaler_vis.fit_transform(X_vis[train_idx])
    X_vis_te_sc = scaler_vis.transform(X_vis[test_idx])

    pca = PCA(n_components=PCA_K, random_state=RANDOM_STATE)
    X_vis_tr_pca = pca.fit_transform(X_vis_tr_sc)
    X_vis_te_pca = pca.transform(X_vis_te_sc)

    # Concatenate: 7 clinical + 5 PCA = 12 dims
    X_train = np.hstack([X_clin_tr, X_vis_tr_pca])
    X_test  = np.hstack([X_clin_te, X_vis_te_pca])

    # ── Train each model ─────────────────────────────────────────────────
    for model_name, model_template in MODELS.items():
        # Clone model for each fold (fresh state)
        from sklearn.base import clone  # pyrefly: ignore [missing-import]
        model = clone(model_template)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)[0]
        y_prob = model.predict_proba(X_test)[0, 1]  # P(Incomplete)

        all_predictions.append({
            'fold': fold_idx + 1,
            'test_case_code': test_case,
            'model': model_name,
            'true_y': int(y_test[0]),
            'true_label': 'Complete' if y_test[0] == 0 else 'Incomplete',
            'pred_y': int(y_pred),
            'pred_label': 'Complete' if y_pred == 0 else 'Incomplete',
            'prob_incomplete': round(float(y_prob), 4),
            'correct': y_pred == y_test[0],
        })

print("      ✔ All 17 folds × 3 models completed")

# ── 3. Aggregate Metrics ────────────────────────────────────────────────────
print("\n[3/4] Computing aggregate metrics per model...")

df_preds = pd.DataFrame(all_predictions)

model_metrics = []

for model_name in MODELS.keys():
    mask = df_preds['model'] == model_name
    y_true = df_preds.loc[mask, 'true_y'].values
    y_pred = df_preds.loc[mask, 'pred_y'].values
    y_prob = df_preds.loc[mask, 'prob_incomplete'].values

    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)   # Recall for Incomplete
    spec = recall_score(y_true, y_pred, pos_label=0)   # Recall for Complete
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    model_metrics.append({
        'model': model_name,
        'accuracy': round(acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'f1': round(f1, 4),
        'roc_auc': round(auc, 4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'correct': int((y_true == y_pred).sum()),
        'total': len(y_true),
    })

# Add baseline for comparison
model_metrics.append({
    'model': BASELINE['model'],
    'accuracy': BASELINE['accuracy'],
    'sensitivity': BASELINE['sensitivity'],
    'specificity': BASELINE['specificity'],
    'f1': BASELINE['f1'],
    'roc_auc': BASELINE['roc_auc'],
    'TP': 6, 'TN': 0, 'FP': 8, 'FN': 3,
    'correct': 6, 'total': 17,
})

df_metrics = pd.DataFrame(model_metrics)

# ── Print Results Table ──────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  MULTIMODAL CLASSIFICATION RESULTS – LOPOCV (n=17)")
print("=" * 90)

# Fold-by-fold detail for each model
for model_name in MODELS.keys():
    mask = df_preds['model'] == model_name
    model_preds = df_preds[mask]
    n_correct = int(model_preds['correct'].sum())
    print(f"\n  ── {model_name} ({n_correct}/17 correct) ──")
    print(f"  {'Fold':>4}  {'Case':>6}  {'True':>10}  {'Pred':>10}  {'P(Inc)':>7}  {'':>5}")
    for _, row in model_preds.iterrows():
        mark = '✔' if row['correct'] else '✘'
        print(f"  {row['fold']:4d}  {row['test_case_code']:>6s}  "
              f"{row['true_label']:>10s}  {row['pred_label']:>10s}  "
              f"{row['prob_incomplete']:>7.4f}  {mark:>5s}")

# Comparison table
print("\n" + "=" * 90)
print("  PERFORMANCE COMPARISON TABLE")
print("=" * 90)
print(f"\n  {'Model':<30s}  {'Acc':>6s}  {'Sens':>6s}  {'Spec':>6s}  "
      f"{'F1':>6s}  {'AUC':>6s}  {'Score':>5s}")
print("  " + "-" * 78)

for _, row in df_metrics.iterrows():
    is_baseline = 'Clinical Only' in row['model']
    marker = '  (baseline)' if is_baseline else ''
    print(f"  {row['model']:<30s}  {row['accuracy']:>6.4f}  {row['sensitivity']:>6.4f}  "
          f"{row['specificity']:>6.4f}  {row['f1']:>6.4f}  {row['roc_auc']:>6.4f}  "
          f"{row['correct']:>2d}/{row['total']}{marker}")

print("  " + "-" * 78)

# Highlight best multimodal model
best_idx = df_metrics[df_metrics['model'] != BASELINE['model']]['accuracy'].idxmax()
best = df_metrics.loc[best_idx]
print(f"\n  🏆 Best Multimodal Model: {best['model']}")
print(f"     Accuracy: {best['accuracy']:.4f} vs Baseline: {BASELINE['accuracy']:.4f} "
      f"(Δ = +{best['accuracy'] - BASELINE['accuracy']:.4f})")
print(f"     ROC-AUC:  {best['roc_auc']:.4f} vs Baseline: {BASELINE['roc_auc']:.4f} "
      f"(Δ = +{best['roc_auc'] - BASELINE['roc_auc']:.4f})")

# ── 4. Save Outputs ─────────────────────────────────────────────────────────
print(f"\n[4/4] Saving outputs...")
df_preds.to_csv(PRED_PATH, index=False)
df_metrics.to_csv(COMP_PATH, index=False)
print(f"      Predictions  : {PRED_PATH}")
print(f"      Comparison   : {COMP_PATH}")

print("\n" + "=" * 70)
print("🎉 Task 4.3 COMPLETE – Multimodal classifiers evaluated.")
print("=" * 70)
