"""
Task 2.3 – L2-Regularized Logistic Regression Baseline (LOPOCV)
================================================================
Features : 7 clinical/IEE predictors  ['LBC','MTB','WOS','TVF','MLE','TM','Age']
Target   : Subtype  (Complete=0, Incomplete=1)
CV       : Leave-One-Patient-Out (17 folds)
Scaling  : StandardScaler fitted on Train only per fold
Tuning   : C in {0.01, 0.1, 1.0, 10.0} via internal CV on Train
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
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedKFold
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

print("=" * 70)
print("   Task 2.3 – L2-Regularized Logistic Regression Baseline")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join("data", "processed")
DATA_PATH     = os.path.join(OUTPUT_DIR, "multimodal_dataset_n17.pkl")
PRED_CSV_PATH = os.path.join(OUTPUT_DIR, "baseline_lr_predictions.csv")

CLINICAL_COLS = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM', 'Age']
LABEL_MAP     = {'Complete': 0, 'Incomplete': 1}
C_GRID        = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE  = 42

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading multimodal dataset...")
df = pd.read_pickle(DATA_PATH)
print(f"      Full dataset shape: {df.shape}")

# Verify all clinical columns exist
missing = [c for c in CLINICAL_COLS if c not in df.columns]
if missing:
    print(f"      [ERROR] Missing columns: {missing}")
    sys.exit(1)

# Extract features, target, and groups
X = df[CLINICAL_COLS].values.astype(np.float64)
y = df['Subtype'].map(LABEL_MAP).values
groups = df['Case Code'].astype(str).str.replace(r'\.0$', '', regex=True).values

print(f"      Feature matrix X : {X.shape}  (columns: {CLINICAL_COLS})")
print(f"      Target vector y  : {y.shape}")
print(f"      Groups           : {len(np.unique(groups))} unique case codes")
print(f"      Class balance    : Complete(0)={int((y==0).sum())}, Incomplete(1)={int((y==1).sum())}")

# Leakage check: ensure NO target-derived columns in X
leakage_cols = ['Subtype', 'Subtype_Binary', 'Pathology_Subtype', 'y']
for lc in leakage_cols:
    assert lc not in CLINICAL_COLS, f"TARGET LEAKAGE: '{lc}' found in feature set!"
print("      ✔ No target leakage in feature matrix")

# ── 2. LOPOCV Setup ─────────────────────────────────────────────────────────
print("\n[2/5] Setting up LOPOCV (17 folds)...")
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, groups)
print(f"      Total folds: {n_splits}")

# ── 3. Train & Evaluate ─────────────────────────────────────────────────────
print("\n[3/5] Training L2-Regularized Logistic Regression per fold...")
print(f"      Hyperparameter grid: C = {C_GRID}")
print(f"      Inner CV: StratifiedKFold(n_splits=3) on 16 training patients")

print("\n      " + "-" * 72)
print(f"      {'Fold':>4}  {'TestCase':>8}  {'TrueLabel':>9}  "
      f"{'PredLabel':>9}  {'Prob(Inc)':>9}  {'BestC':>6}  {'Correct':>7}")
print("      " + "-" * 72)

fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_case = groups[test_idx[0]]
    true_label = 'Complete' if y_test[0] == 0 else 'Incomplete'

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Inner CV for hyperparameter tuning
    # With 16 training patients, use StratifiedKFold(3) for inner CV
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    lr = LogisticRegression(penalty='l2', solver='lbfgs', random_state=RANDOM_STATE,
                            max_iter=1000)
    grid = GridSearchCV(lr, param_grid={'C': C_GRID}, cv=inner_cv,
                        scoring='accuracy', refit=True)
    grid.fit(X_train_sc, y_train)

    best_C = grid.best_params_['C']
    best_model = grid.best_estimator_

    # Predict on held-out test patient
    y_pred = best_model.predict(X_test_sc)[0]
    y_prob = best_model.predict_proba(X_test_sc)[0, 1]   # P(Incomplete)
    pred_label = 'Complete' if y_pred == 0 else 'Incomplete'
    correct = '✔' if y_pred == y_test[0] else '✘'

    print(f"      {fold_idx+1:4d}  {test_case:>8s}  {true_label:>9s}  "
          f"{pred_label:>9s}  {y_prob:>9.4f}  {best_C:>6.2f}  {correct:>7s}")

    fold_results.append({
        'fold': fold_idx + 1,
        'test_case_code': test_case,
        'true_label': true_label,
        'true_y': int(y_test[0]),
        'pred_label': pred_label,
        'pred_y': int(y_pred),
        'prob_incomplete': float(y_prob),
        'best_C': best_C,
        'correct': y_pred == y_test[0],
    })

print("      " + "-" * 72)

# ── 4. Aggregate Metrics ────────────────────────────────────────────────────
print("\n[4/5] Computing aggregate metrics across 17 folds...")

y_true_all = np.array([r['true_y'] for r in fold_results])
y_pred_all = np.array([r['pred_y'] for r in fold_results])
y_prob_all = np.array([r['prob_incomplete'] for r in fold_results])

accuracy    = accuracy_score(y_true_all, y_pred_all)
sensitivity = recall_score(y_true_all, y_pred_all, pos_label=1)  # Recall for Incomplete
specificity = recall_score(y_true_all, y_pred_all, pos_label=0)  # Recall for Complete
f1          = f1_score(y_true_all, y_pred_all)
roc_auc     = roc_auc_score(y_true_all, y_prob_all)

tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()

print(f"""
      ┌────────────────────────────────────────────────┐
      │  L2-Logistic Regression (Clinical/IEE Only)    │
      │  Leave-One-Patient-Out CV  (n=17)              │
      ├────────────────────────────────────────────────┤
      │  Accuracy     : {accuracy:.4f}  ({int(accuracy*17)}/17)            │
      │  Sensitivity  : {sensitivity:.4f}  (Recall for Incomplete)  │
      │  Specificity  : {specificity:.4f}  (Recall for Complete)    │
      │  F1-Score     : {f1:.4f}                            │
      │  ROC-AUC      : {roc_auc:.4f}                            │
      ├────────────────────────────────────────────────┤
      │  Confusion Matrix:                             │
      │            Pred Complete  Pred Incomplete       │
      │  Complete        {tn:2d}             {fp:2d}             │
      │  Incomplete      {fn:2d}             {tp:2d}             │
      └────────────────────────────────────────────────┘
""")

# ── 5. Save Predictions ─────────────────────────────────────────────────────
print("[5/5] Saving out-of-fold predictions...")
df_preds = pd.DataFrame(fold_results)
df_preds.to_csv(PRED_CSV_PATH, index=False)
print(f"      Saved: {PRED_CSV_PATH}")

print("\n" + "=" * 70)
print("🎉 Task 2.3 COMPLETE – Baseline LR evaluated under LOPOCV.")
print("=" * 70)
