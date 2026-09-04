"""
Task 2.4 – Publication-Quality SHAP Visualizations (300 DPI)
============================================================
Input:
  - data/processed/shap_values_cache.pkl
  - data/processed/shap_feature_importance.csv
Output:
  - figures/shap_summary_dot_plot.png
  - figures/shap_bar_importance.png
  - figures/baseline_vs_multimodal_shap_shift.png
"""

import os
import sys
import pickle
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
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns
# pyrefly: ignore [missing-import]
import shap

print("=" * 70)
print("   Task 2.4 – SHAP Visualizations (Publication Quality)")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "processed")
FIG_DIR  = "figures"
PKL_PATH = os.path.join(DATA_DIR, "shap_values_cache.pkl")
CSV_PATH = os.path.join(DATA_DIR, "shap_feature_importance.csv")

os.makedirs(FIG_DIR, exist_ok=True)

PLOT_SHAP_DOT  = os.path.join(FIG_DIR, "shap_summary_dot_plot.png")
PLOT_SHAP_BAR  = os.path.join(FIG_DIR, "shap_bar_importance.png")
PLOT_SHAP_COMP = os.path.join(FIG_DIR, "baseline_vs_multimodal_shap_shift.png")

# Aesthetic settings for publication
plt.style.use('default')
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading SHAP cache and feature importance...")
with open(PKL_PATH, 'rb') as f:
    cache = pickle.load(f)

df_imp = pd.read_csv(CSV_PATH)

rf_shap_vals = cache['rf']['shap_values']
rf_features  = cache['rf']['features']
rf_feat_names = cache['rf']['feature_names']

# ── 2. SHAP Summary Dot Plot (Beeswarm) ──────────────────────────────────────
print("\n[2/4] Generating SHAP Summary Dot Plot...")

# We use shap.summary_plot which by default creates a beeswarm plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    rf_shap_vals, 
    features=rf_features, 
    feature_names=rf_feat_names,
    show=False,
    plot_type="dot",
    cmap=plt.get_cmap("coolwarm")
)
plt.title("SHAP Summary: Multimodal Feature Impact (RF Model)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(PLOT_SHAP_DOT, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✔ Saved: {PLOT_SHAP_DOT}")

# ── 3. SHAP Bar Importance Plot (Color-coded) ────────────────────────────────
print("\n[3/4] Generating Ranked SHAP Bar Plot...")

# Get RF importance data
df_rf_imp = df_imp[df_imp['Model'].str.contains('RandomForest')].copy()
df_rf_imp = df_rf_imp.sort_values('Mean_Abs_SHAP', ascending=True) # Ascending for horizontal bar plot

features = df_rf_imp['Feature'].values
shaps = df_rf_imp['Mean_Abs_SHAP'].values

# Color mapping: Visual PCs = blue, Clinical = orange
colors = ['#4c72b0' if 'Visual' in f else '#dd8452' for f in features]

plt.figure(figsize=(10, 8))
bars = plt.barh(features, shaps, color=colors, edgecolor='none')
plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=12)
plt.title("Feature Importance: Deep Visual vs. Clinical (RF Model)", fontsize=14, pad=15)

# Custom legend
# pyrefly: ignore [missing-import]
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4c72b0', label='Deep Visual Features (PCA)'),
    Patch(facecolor='#dd8452', label='Clinical / IEE Features')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(PLOT_SHAP_BAR, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✔ Saved: {PLOT_SHAP_BAR}")

# ── 4. Baseline vs. Multimodal SHAP Shift ────────────────────────────────────
print("\n[4/4] Generating Comparative SHAP Shift Plot (Clinical Features)...")

df_lr_imp = df_imp[df_imp['Model'].str.contains('L2-LogReg')].copy()

# We want to compare the SHAP of clinical features in LR vs RF.
# To make them somewhat comparable visually, we can normalize them to sum to 100% within each model
sum_lr = df_lr_imp['Mean_Abs_SHAP'].sum()
sum_rf_clin = df_rf_imp[~df_rf_imp['Feature'].str.contains('Visual')]['Mean_Abs_SHAP'].sum()

df_lr_imp['Normalized_SHAP'] = (df_lr_imp['Mean_Abs_SHAP'] / sum_lr) * 100
df_rf_clin = df_rf_imp[~df_rf_imp['Feature'].str.contains('Visual')].copy()
df_rf_clin['Normalized_SHAP'] = (df_rf_clin['Mean_Abs_SHAP'] / sum_rf_clin) * 100

# Merge
df_shift = pd.merge(
    df_lr_imp[['Feature', 'Normalized_SHAP']],
    df_rf_clin[['Feature', 'Normalized_SHAP']],
    on='Feature', suffixes=('_Baseline', '_Multimodal')
)
df_shift = df_shift.sort_values('Normalized_SHAP_Baseline', ascending=True)

x = np.arange(len(df_shift))
width = 0.35

plt.figure(figsize=(10, 7))
plt.barh(x - width/2, df_shift['Normalized_SHAP_Baseline'], width, label='Clinical-Only (Baseline)', color='#8c8c8c')
plt.barh(x + width/2, df_shift['Normalized_SHAP_Multimodal'], width, label='Multimodal (RF)', color='#dd8452')

plt.yticks(x, df_shift['Feature'])
plt.xlabel("Relative Clinical Importance (%)", fontsize=12)
plt.title("Attribution Shift of Clinical Features: Baseline vs. Multimodal", fontsize=14, pad=15)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig(PLOT_SHAP_COMP, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✔ Saved: {PLOT_SHAP_COMP}")

print("\n" + "=" * 70)
print("🎉 Task 2.4 COMPLETE – High-resolution SHAP figures generated.")
print("=" * 70)
