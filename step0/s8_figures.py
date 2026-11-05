"""Sub-step 8 - traceable replacements for figure2_comprehensive_performance_panel.png and
patient_level_correction_analysis.csv (neither had a generating script; their numbers did not match other files).

Every number is read from step-0 CSVs:
  s5_lpo_ci/ci_table.csv, s4_permutation/{null_distribution,permutation_summary}.csv,
  s3_v1_configs/{per_patient_probs,fold_thresholds}.csv, s2_baselines/per_patient_scores.csv,
  s6_confounders/patient_factors.csv
Outputs: results/step0/figures/step0_performance_panel.png, results/step0/patient_level_table.csv
No clinical-impact wording is generated.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RES, load_cohort, out_dir

FIG = out_dir("figures")
CI = pd.read_csv(RES / "s5_lpo_ci" / "ci_table.csv")
null = pd.read_csv(RES / "s4_permutation" / "null_distribution.csv")
perm = pd.read_csv(RES / "s4_permutation" / "permutation_summary.csv")

# post-hoc row: nested BiomedCLIP-LR on FOV-cropped embeddings (sub-step 7d), drawn in a different colour
from s5_lpo_bootstrap import boot_ci, clopper_pearson
pc = pd.read_csv(RES / "s7_fov" / "per_patient_cropped.csv")
pc = pc[(pc.variant == "cropped_v1frames") & (pc.model == "b_biomedclip/LR")].sort_values("Case Code")
chk = pd.read_csv(RES / "s7_fov" / "cropped_biomedclip_lr_checks.csv").set_index("check")
yc, sc, prc = pc.y.values, pc.score.values, pc.pred.values
(alo, ahi), _, _ = boot_ci(yc, sc, prc)
k = int((yc == prc).sum())
lo, hi = clopper_pearson(k, len(yc))
from sklearn.metrics import roc_auc_score
CI = pd.concat([CI, pd.DataFrame([{"model": "post hoc: FOV-cropped b_biomedclip/LR", "n_correct": k, "accuracy": k / len(yc),
                                   "acc_clopper_pearson_lo": lo, "acc_clopper_pearson_hi": hi,
                                   "auc_lopocv": roc_auc_score(yc, sc), "auc_boot_lo": alo, "auc_boot_hi": ahi,
                                   "lpo_auc": chk.loc["leave-pair-out AUC", "value"]}])], ignore_index=True)
CI = CI.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(1, 3, figsize=(20, 7.5), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
yy = np.arange(len(CI))
is_v1 = CI.model.str.startswith("v1_")
col = np.where(is_v1, "#c0392b", np.where(CI.model.str.startswith("post hoc"), "#2e86c1", "#34495e"))
ax[0].hlines(yy, CI.acc_clopper_pearson_lo, CI.acc_clopper_pearson_hi, color=col, lw=2)
ax[0].scatter(CI.accuracy, yy, color=col, zorder=3)
ax[0].axvline(0.5, ls=":", c="gray")
ax[0].set_yticks(yy, [m.replace("v1_champion_SVC8", "v1 champion SVC-8PC\n(best of 12, chosen post hoc)") for m in CI.model], fontsize=8)
ax[0].set_xlabel("LOPOCV accuracy with Clopper-Pearson 95% CI (n = 17)")
ax[0].set_xlim(0, 1)
ax[0].set_title("A  Accuracy (step-0 models: fully nested)")
ax[1].hlines(yy, CI.auc_boot_lo, CI.auc_boot_hi, color=col, lw=2)
ax[1].scatter(CI.auc_lopocv, yy, color=col, zorder=3, label="pooled LOPOCV AUC (bootstrap CI)")
ax[1].scatter(CI.lpo_auc, yy, marker="D", facecolor="white", edgecolor=col, zorder=4, label="Leave-Pair-Out AUC")
ax[1].axvline(0.5, ls=":", c="gray")
ax[1].set_yticks(yy, [""] * len(yy))
ax[1].set_xlim(0, 1)
ax[1].set_xlabel("AUC")
ax[1].legend(loc="lower left", fontsize=8)
ax[1].set_title("B  AUC: pooled LOPOCV vs Leave-Pair-Out")
obs = perm.set_index("statistic")
o = obs.loc["max accuracy over 12 configs (selection-aware)"]
ax[2].hist(null.max_acc_12, bins=np.arange(0, 18) / 17 - 1 / 34, color="#95a5a6", edgecolor="white")
ax[2].axvline(o.observed, color="#c0392b", lw=2, label=f"observed 14/17\nselection-aware p = {o.p_value:.3f}")
ax[2].set_xlabel("max accuracy over the 12 v1 configs")
ax[2].set_ylabel(f"count ({len(null)} label permutations)")
ax[2].legend(loc="upper left", fontsize=9)
ax[2].set_title("C  Whole-pipeline permutation null")
fig.suptitle("Goal 3 (Complete vs Incomplete), n = 17 - feasibility pilot, no clinical claim", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "step0_performance_panel.png", dpi=200)

# patient-level table
df, y, b = load_cohort()
cases = df["Case Code"].values
fac = pd.read_csv(RES / "s6_confounders" / "patient_factors.csv").set_index("Case Code").loc[cases]
v1p = pd.read_csv(RES / "s3_v1_configs" / "per_patient_probs.csv")
v1t = pd.read_csv(RES / "s3_v1_configs" / "fold_thresholds.csv")
sv = v1p[(v1p.model == "SVC") & (v1p.PCs == 8)].set_index("Case Code").loc[cases]
th = v1t[(v1t.model == "SVC") & (v1t.PCs == 8)].set_index("Case Code").loc[cases].youden_threshold
s2 = pd.read_csv(RES / "s2_baselines" / "per_patient_scores.csv")
T = pd.DataFrame({"Case Code": cases, "histology_label": fac.label.values,
                  "mixed_histology": np.isin(cases, [2602, 2611]),
                  "visual_source_v1": fac.source.values, "file_type": fac.file_type.values,
                  "processor": fac.processor.values, "OSD_text_panel": fac.OSD_text_panel.values,
                  "v1_champion_prob": sv.prob_incomplete.values.round(4), "v1_champion_threshold": th.values,
                  "v1_champion_correct": ((sv.prob_incomplete.values >= th.values).astype(int) == y)})
for fs, m in (("a_clinical", "LR"), ("a_clinical", "RF"), ("b_biomedclip", "LR"), ("b_biomedclip", "RF"),
              ("e_visual+clinical", "LR")):
    g = s2[(s2.feature_set == fs) & (s2.model == m)].set_index("Case Code").loc[cases]
    T[f"{fs}/{m}_score"] = g.score.values.round(4)
    T[f"{fs}/{m}_correct"] = (g.pred.values == y)
T.to_csv(RES / "patient_level_table.csv", index=False)
print("saved", FIG / "step0_performance_panel.png", RES / "patient_level_table.csv")
