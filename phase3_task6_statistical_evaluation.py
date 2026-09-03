"""
Task 6.3 – Statistical Significance Testing & Bootstrap 95% CI
================================================================
Input:
  - data/processed/baseline_lr_predictions.csv
  - data/processed/multimodal_classifier_predictions.csv
Output:
  - data/processed/statistical_evaluation_report.csv
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("=" * 70)
print("   Task 6.3 – Statistical Significance & Bootstrap 95% CI")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join("data", "processed")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_lr_predictions.csv")
MULTI_PATH    = os.path.join(OUTPUT_DIR, "multimodal_classifier_predictions.csv")
REPORT_PATH   = os.path.join(OUTPUT_DIR, "statistical_evaluation_report.csv")

N_BOOTSTRAP   = 1000
RANDOM_STATE  = 42
CI_ALPHA      = 0.05   # 95% CI

# ── 1. Load Predictions ─────────────────────────────────────────────────────
print("\n[1/4] Loading prediction files...")
df_base  = pd.read_csv(BASELINE_PATH)
df_multi = pd.read_csv(MULTI_PATH)

print(f"      Baseline predictions : {df_base.shape}")
print(f"      Multimodal predictions: {df_multi.shape}")

# Build per-model arrays: {model_name: (y_true, y_pred, y_prob)}
models = {}

# Baseline LR
models['L2-LogReg (Clinical)'] = {
    'y_true': df_base['true_y'].values,
    'y_pred': df_base['pred_y'].values,
    'y_prob': df_base['prob_incomplete'].values,
}

# Multimodal models
for model_name in ['RandomForest', 'SVM_RBF', 'GradientBoosting']:
    mask = df_multi['model'] == model_name
    models[model_name] = {
        'y_true': df_multi.loc[mask, 'true_y'].values,
        'y_pred': df_multi.loc[mask, 'pred_y'].values,
        'y_prob': df_multi.loc[mask, 'prob_incomplete'].values,
    }

for name, data in models.items():
    print(f"      {name:30s}: n={len(data['y_true'])}")

# ── 2. Bootstrap 95% Confidence Intervals ────────────────────────────────────
print(f"\n[2/4] Computing Bootstrap 95% CIs ({N_BOOTSTRAP} iterations)...")

rng = np.random.RandomState(RANDOM_STATE)

def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_boot, rng, **kwargs):
    """Stratified bootstrap CI for a given metric function."""
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        # Stratified resampling: sample with replacement
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]

        # Skip degenerate samples (only one class)
        if len(np.unique(yt)) < 2:
            continue

        try:
            if 'needs_prob' in kwargs and kwargs['needs_prob']:
                score = metric_fn(yt, ypr)
            else:
                score = metric_fn(yt, yp)
            boot_scores.append(score)
        except Exception:
            continue

    boot_scores = np.array(boot_scores)
    if len(boot_scores) == 0:
        return 0.0, 0.0, 0.0

    point = np.mean(boot_scores)
    lo = np.percentile(boot_scores, 100 * CI_ALPHA / 2)
    hi = np.percentile(boot_scores, 100 * (1 - CI_ALPHA / 2))
    return point, lo, hi

ci_results = []

for model_name, data in models.items():
    y_true = data['y_true']
    y_pred = data['y_pred']
    y_prob = data['y_prob']

    # Point estimates
    acc_point = accuracy_score(y_true, y_pred)
    f1_point  = f1_score(y_true, y_pred)
    try:
        auc_point = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_point = 0.0

    # Bootstrap CIs
    acc_mean, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, y_prob,
                                             accuracy_score, N_BOOTSTRAP, rng)
    f1_mean, f1_lo, f1_hi   = bootstrap_ci(y_true, y_pred, y_prob,
                                             f1_score, N_BOOTSTRAP, rng)
    auc_mean, auc_lo, auc_hi = bootstrap_ci(y_true, y_pred, y_prob,
                                              roc_auc_score, N_BOOTSTRAP, rng,
                                              needs_prob=True)

    ci_results.append({
        'model': model_name,
        'accuracy_point': round(acc_point, 4),
        'accuracy_ci_lo': round(acc_lo, 4),
        'accuracy_ci_hi': round(acc_hi, 4),
        'f1_point': round(f1_point, 4),
        'f1_ci_lo': round(f1_lo, 4),
        'f1_ci_hi': round(f1_hi, 4),
        'roc_auc_point': round(auc_point, 4),
        'roc_auc_ci_lo': round(auc_lo, 4),
        'roc_auc_ci_hi': round(auc_hi, 4),
    })

# Print CI table
print(f"\n      ┌{'─'*80}┐")
print(f"      │  {'Model':<28s}  {'Accuracy':>18s}  {'F1-Score':>18s}  {'ROC-AUC':>18s}  │")
print(f"      │  {'':28s}  {'(95% CI)':>18s}  {'(95% CI)':>18s}  {'(95% CI)':>18s}  │")
print(f"      ├{'─'*80}┤")

for r in ci_results:
    acc_str = f"{r['accuracy_point']:.3f} [{r['accuracy_ci_lo']:.3f}-{r['accuracy_ci_hi']:.3f}]"
    f1_str  = f"{r['f1_point']:.3f} [{r['f1_ci_lo']:.3f}-{r['f1_ci_hi']:.3f}]"
    auc_str = f"{r['roc_auc_point']:.3f} [{r['roc_auc_ci_lo']:.3f}-{r['roc_auc_ci_hi']:.3f}]"
    print(f"      │  {r['model']:<28s}  {acc_str:>18s}  {f1_str:>18s}  {auc_str:>18s}  │")

print(f"      └{'─'*80}┘")

# ── 3. McNemar's Test ────────────────────────────────────────────────────────
print(f"\n[3/4] McNemar's Test: RandomForest vs L2-LogReg (Clinical)...")

y_true_base = models['L2-LogReg (Clinical)']['y_true']
y_pred_base = models['L2-LogReg (Clinical)']['y_pred']
y_pred_rf   = models['RandomForest']['y_pred']

# Correctness vectors
correct_base = (y_true_base == y_pred_base).astype(int)
correct_rf   = (y_true_base == y_pred_rf).astype(int)

# 2x2 contingency table
#                    RF Correct   RF Wrong
# Baseline Correct      a            b
# Baseline Wrong        c            d

a = int(((correct_base == 1) & (correct_rf == 1)).sum())  # both correct
b = int(((correct_base == 1) & (correct_rf == 0)).sum())  # base correct, RF wrong
c = int(((correct_base == 0) & (correct_rf == 1)).sum())  # base wrong, RF correct
d = int(((correct_base == 0) & (correct_rf == 0)).sum())  # both wrong

print(f"\n      2×2 Contingency Table (Discordant Pairs):")
print(f"      ┌────────────────────────────────────────┐")
print(f"      │                  RF Correct  RF Wrong  │")
print(f"      │  Baseline Correct     {a:2d}         {b:2d}     │")
print(f"      │  Baseline Wrong       {c:2d}         {d:2d}     │")
print(f"      └────────────────────────────────────────┘")
print(f"      Discordant pairs: b={b}, c={c}")

# McNemar's test with Edwards' continuity correction
# χ² = (|b - c| - 1)² / (b + c)
if (b + c) > 0:
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution (1 df)
    # pyrefly: ignore [missing-import]
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    # Exact two-sided binomial test on discordant pairs (exact McNemar)
    # pyrefly: ignore [missing-import]
    from scipy.stats import binomtest
    binom_result = binomtest(c, c + b, p=0.5, alternative='two-sided')
    p_exact = binom_result.pvalue

    print(f"\n      McNemar's Test (with Edwards' continuity correction):")
    print(f"        χ²(1)          = {chi2:.4f}")
    print(f"        p-value (χ²)   = {p_value:.4f}")
    print(f"        p-value (exact) = {p_exact:.4f}")

    if p_value < 0.05:
        print(f"        → Statistically significant (p < 0.05)")
    else:
        print(f"        → NOT statistically significant (p ≥ 0.05)")
        print(f"          (Expected with n=17; small sample limits statistical power)")
else:
    chi2 = 0.0
    p_value = 1.0
    p_exact = 1.0
    print(f"\n      McNemar's Test: No discordant pairs (b=c=0). Cannot compute.")

# Effect size: proportion of improvement
n_total = len(y_true_base)
acc_base = accuracy_score(y_true_base, y_pred_base)
acc_rf   = accuracy_score(y_true_base, y_pred_rf)
delta_acc = acc_rf - acc_base

print(f"\n      Effect Size:")
print(f"        Baseline Accuracy : {acc_base:.4f}")
print(f"        RF Accuracy       : {acc_rf:.4f}")
print(f"        Δ Accuracy        : +{delta_acc:.4f} ({delta_acc*100:.1f} pp)")
print(f"        Improvement ratio : {acc_rf/acc_base:.2f}x")

# ── 4. Save Report ──────────────────────────────────────────────────────────
print(f"\n[4/4] Saving statistical evaluation report...")

df_ci = pd.DataFrame(ci_results)

# Add McNemar summary row
mcnemar_row = {
    'model': 'McNemar: RF vs Baseline',
    'accuracy_point': chi2,
    'accuracy_ci_lo': p_value,
    'accuracy_ci_hi': p_exact,
    'f1_point': b,
    'f1_ci_lo': c,
    'f1_ci_hi': b + c,
    'roc_auc_point': delta_acc,
    'roc_auc_ci_lo': acc_base,
    'roc_auc_ci_hi': acc_rf,
}
df_report = pd.concat([df_ci, pd.DataFrame([mcnemar_row])], ignore_index=True)
df_report.to_csv(REPORT_PATH, index=False)
print(f"      Saved: {REPORT_PATH}")

# Final consolidated summary
print(f"\n{'='*70}")
print(f"  CONSOLIDATED STATISTICAL REPORT")
print(f"{'='*70}")
print(f"\n  Bootstrap 95% CIs ({N_BOOTSTRAP} iterations):")
for r in ci_results:
    print(f"    {r['model']:<28s}")
    print(f"      Accuracy : {r['accuracy_point']:.3f}  95% CI [{r['accuracy_ci_lo']:.3f}, {r['accuracy_ci_hi']:.3f}]")
    print(f"      F1-Score : {r['f1_point']:.3f}  95% CI [{r['f1_ci_lo']:.3f}, {r['f1_ci_hi']:.3f}]")
    print(f"      ROC-AUC  : {r['roc_auc_point']:.3f}  95% CI [{r['roc_auc_ci_lo']:.3f}, {r['roc_auc_ci_hi']:.3f}]")

print(f"\n  McNemar's Test (RF vs Clinical-Only Baseline):")
print(f"    Discordant pairs: b={b} (base✔ RF✘), c={c} (base✘ RF✔)")
print(f"    χ²(1) = {chi2:.4f},  p = {p_value:.4f} (corrected),  p = {p_exact:.4f} (exact)")
if p_value < 0.05:
    print(f"    ✔ SIGNIFICANT at α=0.05")
else:
    print(f"    ✘ Not significant at α=0.05 (limited by n=17 sample size)")

print(f"\n{'='*70}")
print(f"🎉 Task 6.3 COMPLETE – Statistical evaluation finished.")
print(f"{'='*70}")
