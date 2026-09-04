"""
verify_all_phases.py – Quick Visual Verification of All Pipeline Outputs
=========================================================================
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
# pyrefly: ignore [missing-import]
from sklearn.metrics import confusion_matrix

P = os.path.join  # shorthand

print()
print("╔" + "═"*68 + "╗")
print("║" + "  GIMENDO v2 – Full Pipeline Verification Report".center(68) + "║")
print("╚" + "═"*68 + "╝")

# ── 1. File Inventory ────────────────────────────────────────────────────────
print("\n┌─ 1. FILE INVENTORY " + "─"*49 + "┐")

checks = [
    ("data/filtered_frames",                           "dir",  "Filtered Frames"),
    ("data/processed/cleaned_metadata_n17.xlsx",       "file", "Cleaned Metadata"),
    ("data/processed/frame_embeddings_1976x512.npy",   "file", "Frame Embeddings"),
    ("data/processed/frame_metadata.csv",              "file", "Frame Metadata"),
    ("data/processed/patient_visual_features_n17.npy", "file", "Patient Visual Features"),
    ("data/processed/patient_visual_features_n17.csv", "file", "Patient Visual CSV"),
    ("data/processed/multimodal_dataset_n17.csv",      "file", "Multimodal Dataset CSV"),
    ("data/processed/multimodal_dataset_n17.pkl",      "file", "Multimodal Dataset PKL"),
    ("data/processed/lopocv_splits_n17.pkl",           "file", "LOPOCV Splits"),
    ("data/processed/baseline_lr_predictions.csv",     "file", "Baseline LR Predictions"),
    ("data/processed/multimodal_classifier_predictions.csv", "file", "Multimodal Predictions"),
    ("data/processed/multimodal_model_comparison.csv", "file", "Model Comparison"),
    ("data/processed/infold_pca_summary.csv",          "file", "In-Fold PCA Summary"),
    ("data/processed/olgim_subcohort_analysis.csv",    "file", "OLGIM Sub-cohort"),
    ("data/processed/statistical_evaluation_report.csv","file","Statistical Report"),
    ("data/processed/dinov2_frame_embeddings_2016x768.npy", "file", "DINOv2 Frame Embeds"),
    ("data/processed/dinov2_patient_features_17x768.npy", "file", "DINOv2 Patient Feats"),
    ("data/processed/multimodal_biomedclip_dinov2_n17.csv", "file", "DINOv2 Fused Dataset"),
    ("data/processed/dinov2_final_statistical_report.csv", "file", "DINOv2 Stats Report"),
    ("figures/multimodal_roc_comparison_dinov2.png", "file", "DINOv2 ROC Plot"),
    ("figures/model_accuracy_jump_comparison.png", "file", "DINOv2 Jump Plot"),
]

print(f"│  {'Asset':<28s}  {'Status':>6s}  {'Size':>10s}  │")
print("│  " + "─"*50 + "  │")

for path, kind, label in checks:
    if kind == "dir":
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            print(f"│  {label:<28s}  {'  ✔':>6s}  {n:>7d} img  │")
        else:
            print(f"│  {label:<28s}  {'  ✘':>6s}  {'MISSING':>10s}  │")
    else:
        if os.path.isfile(path):
            sz = os.path.getsize(path)
            if sz > 1_000_000:
                sz_str = f"{sz/1_000_000:.1f} MB"
            else:
                sz_str = f"{sz/1_000:.1f} KB"
            print(f"│  {label:<28s}  {'  ✔':>6s}  {sz_str:>10s}  │")
        else:
            print(f"│  {label:<28s}  {'  ✘':>6s}  {'MISSING':>10s}  │")

print("└" + "─"*68 + "┘")

# ── 2. Key Data Shapes ──────────────────────────────────────────────────────
print("\n┌─ 2. DATA SHAPES " + "─"*51 + "┐")

frames_dir = "data/filtered_frames"
n_frames = len([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
frame_check = "✔" if n_frames == 2016 else "⚠"

df_mm = pd.read_csv(P("data","processed","multimodal_dataset_n17.csv"))
shape_check = "✔" if df_mm.shape == (17, 533) else "⚠"

emb = np.load(P("data","processed","frame_embeddings_1976x512.npy"))
emb_check = "✔" if emb.shape == (2016, 512) else "⚠"

pvf = np.load(P("data","processed","patient_visual_features_n17.npy"))
pvf_check = "✔" if pvf.shape == (17, 512) else "⚠"

dinov2_emb = np.load(P("data","processed","dinov2_frame_embeddings_2016x768.npy")) if os.path.exists(P("data","processed","dinov2_frame_embeddings_2016x768.npy")) else np.zeros((0,0))
dinov2_emb_check = "✔" if dinov2_emb.shape == (2016, 768) else "⚠"

dinov2_pvf = np.load(P("data","processed","dinov2_patient_features_17x768.npy")) if os.path.exists(P("data","processed","dinov2_patient_features_17x768.npy")) else np.zeros((0,0))
dinov2_pvf_check = "✔" if dinov2_pvf.shape == (17, 768) else "⚠"

dinov2_mm = pd.read_csv(P("data","processed","multimodal_biomedclip_dinov2_n17.csv")) if os.path.exists(P("data","processed","multimodal_biomedclip_dinov2_n17.csv")) else pd.DataFrame()
dinov2_shape_check = "✔" if dinov2_mm.shape == (17, 1301) else "⚠"

print(f"│  {'Item':<35s}  {'Actual':>14s}  {'Expected':>14s}  {'':>2s} │")
print("│  " + "─"*60 + "  │")
print(f"│  {'Filtered Frames':<35s}  {n_frames:>14d}  {'2,016':>14s}  {frame_check:>2s} │")
print(f"│  {'Frame Embeddings':<35s}  {str(emb.shape):>14s}  {'(2016, 512)':>14s}  {emb_check:>2s} │")
print(f"│  {'Patient Visual Features':<35s}  {str(pvf.shape):>14s}  {'(17, 512)':>14s}  {pvf_check:>2s} │")
print(f"│  {'Multimodal Dataset':<35s}  {str(df_mm.shape):>14s}  {'(17, 533)':>14s}  {shape_check:>2s} │")
print(f"│  {'DINOv2 Frame Embeds':<35s}  {str(dinov2_emb.shape):>14s}  {'(2016, 768)':>14s}  {dinov2_emb_check:>2s} │")
print(f"│  {'DINOv2 Patient Features':<35s}  {str(dinov2_pvf.shape):>14s}  {'(17, 768)':>14s}  {dinov2_pvf_check:>2s} │")
print(f"│  {'DINOv2 Multimodal Dataset':<35s}  {str(dinov2_mm.shape):>14s}  {'(17, 1301)':>14s}  {dinov2_shape_check:>2s} │")
print("└" + "─"*68 + "┘")

# ── 3. Random Forest Confusion Matrix ───────────────────────────────────────
print("\n┌─ 3. BEST MODEL CONFUSION MATRIX (Random Forest) " + "─"*18 + "┐")

df_preds = pd.read_csv(P("data","processed","multimodal_classifier_predictions.csv"))
rf = df_preds[df_preds['model'] == 'RandomForest']
y_true = rf['true_y'].values
y_pred = rf['pred_y'].values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

acc = (tp+tn)/(tp+tn+fp+fn)
sens = tp/(tp+fn) if (tp+fn) > 0 else 0
spec = tn/(tn+fp) if (tn+fp) > 0 else 0

print("│                                                                    │")
print("│              Predicted Complete    Predicted Incomplete             │")
print(f"│   Complete          {tn:2d}                   {fp:2d}                    │")
print(f"│   Incomplete        {fn:2d}                   {tp:2d}                    │")
print("│                                                                    │")
print(f"│   Accuracy    = {acc:.4f}  ({int(acc*17)}/17)                              │")
print(f"│   Sensitivity = {sens:.4f}  (Recall for Incomplete)                  │")
print(f"│   Specificity = {spec:.4f}  (Recall for Complete)                    │")
print("└" + "─"*68 + "┘")

# ── 4. Statistical Summary ──────────────────────────────────────────────────
print("\n┌─ 4. STATISTICAL SIGNIFICANCE SUMMARY " + "─"*30 + "┐")

df_stats = pd.read_csv(P("data","processed","statistical_evaluation_report.csv"))

# Extract model CIs
for _, row in df_stats.iterrows():
    model = row['model']
    if 'McNemar' in str(model):
        chi2_val = row['accuracy_point']
        p_corr   = row['accuracy_ci_lo']
        p_exact  = row['accuracy_ci_hi']
        b_val    = int(row['f1_point'])
        c_val    = int(row['f1_ci_lo'])
    else:
        acc_str = f"{row['accuracy_point']:.3f} [{row['accuracy_ci_lo']:.3f}, {row['accuracy_ci_hi']:.3f}]"
        f1_str  = f"{row['f1_point']:.3f} [{row['f1_ci_lo']:.3f}, {row['f1_ci_hi']:.3f}]"
        auc_str = f"{row['roc_auc_point']:.3f} [{row['roc_auc_ci_lo']:.3f}, {row['roc_auc_ci_hi']:.3f}]"
        print(f"│  {model:<26s}                                          │")
        print(f"│    Accuracy  : {acc_str:<52s} │")
        print(f"│    F1-Score  : {f1_str:<52s} │")
        print(f"│    ROC-AUC   : {auc_str:<52s} │")
        print("│                                                                    │")

print("│  " + "─"*64 + "  │")
print("│  McNemar's Test (RF vs Clinical Baseline):                         │")
print(f"│    Discordant pairs  : b={b_val}, c={c_val}                                   │")
print(f"│    χ²(1)             : {chi2_val:.4f}                                         │")
print(f"│    p-value (corrected): {p_corr:.4f}                                         │")
print(f"│    p-value (exact)   : {p_exact:.4f}                                         │")
sig = "✔ Significant" if p_corr < 0.05 else "✘ Not significant (p≥0.05)"
print(f"│    Result            : {sig:<44s} │")
print("└" + "─"*68 + "┘")

# ── 5. New DINOv2 Statistical Summary ─────────────────────────────────────────
print("\n┌─ 5. MULTI-VISION DINOV2 STATISTICAL SUMMARY " + "─"*22 + "┐")

dinov2_stats = pd.read_csv(P("data","processed","dinov2_final_statistical_report.csv"))

for _, row in dinov2_stats.iterrows():
    model = row['Model']
    acc_str = f"{row['Accuracy']:.3f} [{row['Accuracy_CI_lower']:.3f}, {row['Accuracy_CI_upper']:.3f}]"
    f1_str  = f"{row['F1_Score']:.3f} [{row['F1_CI_lower']:.3f}, {row['F1_CI_upper']:.3f}]"
    auc_str = f"{row['ROC_AUC']:.3f} [{row['ROC_AUC_CI_lower']:.3f}, {row['ROC_AUC_CI_upper']:.3f}]"
    p_val   = row['McNemar_p_value_vs_Baseline']
    print(f"│  {model:<26s}                                          │")
    print(f"│    Accuracy  : {acc_str:<52s} │")
    print(f"│    F1-Score  : {f1_str:<52s} │")
    print(f"│    ROC-AUC   : {auc_str:<52s} │")
    print(f"│    McNemar vs Baseline : {p_val:<41.4f} │")
    print("│                                                                    │")

print("└" + "─"*68 + "┘")

# ── Final Verdict ────────────────────────────────────────────────────────────
all_ok = (n_frames == 2016 and df_mm.shape == (17, 533)
          and emb.shape == (2016, 512) and pvf.shape == (17, 512)
          and dinov2_emb.shape == (2016, 768) and dinov2_pvf.shape == (17, 768)
          and dinov2_mm.shape == (17, 1301))

print()
if all_ok:
    print("╔" + "═"*68 + "╗")
    print("║" + "  ✔ ALL PIPELINE PHASES VERIFIED SUCCESSFULLY".center(68) + "║")
    print("╚" + "═"*68 + "╝")
else:
    print("╔" + "═"*68 + "╗")
    print("║" + "  ⚠ SOME CHECKS FAILED — REVIEW ABOVE".center(68) + "║")
    print("╚" + "═"*68 + "╝")
