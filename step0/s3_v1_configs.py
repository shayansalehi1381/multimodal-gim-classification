"""Sub-step 3 - all 12 v1 configurations side by side (3 models x PCs{5,8} x threshold{0.5, Youden}).
Uses the exact v1 protocol (common.v1_run; fallback removed), checked against the tracked
data/processed/dinov2_multimodal_evaluation_results.csv.
Outputs: results/step0/s3_v1_configs/{configs_table.csv, fold_thresholds.csv, per_patient_probs.csv}
"""
import warnings

import numpy as np
import pandas as pd

from common import PROC, V1_CONFIGS, load_cohort, log, metrics, out_dir, v1_run

warnings.filterwarnings("ignore")
OUT = out_dir("s3_v1_configs")

df, y, b = load_cohort()
Xc, Xv = b["clinical"], np.hstack([b["biomedclip"], b["dinov2"]])
rows, thr_rows, prob_rows = [], [], []
for model, k in V1_CONFIGS:
    prob, thr = v1_run(Xc, Xv, y, model, k)
    for ttype, pred in (("default_0.5", (prob >= 0.5).astype(int)), ("Youden_in_fold", (prob >= thr).astype(int))):
        m = metrics(y, prob, pred)
        rows.append({"model": model, "PCs": k, "threshold": ttype, **m,
                     "thr_min": 0.5 if ttype == "default_0.5" else thr.min(),
                     "thr_median": 0.5 if ttype == "default_0.5" else np.median(thr),
                     "thr_max": 0.5 if ttype == "default_0.5" else thr.max(),
                     "n_folds_thr_0.01": 0 if ttype == "default_0.5" else int((thr <= 0.0101).sum())})
    for cc, t, p in zip(df["Case Code"], thr, prob):
        thr_rows.append({"model": model, "PCs": k, "Case Code": cc, "youden_threshold": t})
        prob_rows.append({"model": model, "PCs": k, "Case Code": cc, "y": int(y[list(df["Case Code"]).index(cc)]),
                          "prob_incomplete": p})
T = pd.DataFrame(rows)
T.to_csv(OUT / "configs_table.csv", index=False)
pd.DataFrame(thr_rows).to_csv(OUT / "fold_thresholds.csv", index=False)
pd.DataFrame(prob_rows).to_csv(OUT / "per_patient_probs.csv", index=False)

# consistency with tracked v1 output
v1 = pd.read_csv(PROC / "dinov2_multimodal_evaluation_results.csv")
mine = T.assign(Threshold_Type=T.threshold.map({"default_0.5": "Standard_0.5", "Youden_in_fold": "Tuned_Youden"}))
mrg = v1.merge(mine, left_on=["Model", "PCs", "Threshold_Type"], right_on=["model", "PCs", "Threshold_Type"])
same = np.allclose(mrg.Accuracy, mrg.accuracy) and np.allclose(mrg.ROC_AUC, mrg.auc)

acc = T.accuracy.values
txt = f"""
All 12 v1 configurations rerun with the exact v1 protocol (fallback removed). They match the tracked
dinov2_multimodal_evaluation_results.csv: {same}.
Accuracy over the 12 configs: min {acc.min():.3f}, median {np.median(acc):.3f}, mean {acc.mean():.3f}, max {acc.max():.3f}.
Only 1 of 12 reaches 0.824 (SVC, 8 PCs, Youden). The second best is 0.588. The same SVC-8PC at the default 0.5 cut
gets {T[(T.model=='SVC')&(T.PCs==8)&(T.threshold=='default_0.5')].accuracy.iloc[0]:.3f}.
Balanced accuracy: median {np.median(T.balanced_accuracy):.3f}, max {T.balanced_accuracy.max():.3f}. AUC does not depend on the threshold:
RF {T[T.model=='RF'].auc.unique().round(3).tolist()}, SVC {T[T.model=='SVC'].auc.unique().round(3).tolist()},
GB {T[T.model=='GB'].auc.unique().round(3).tolist()} (5 vs 8 PCs).
Per-fold Youden thresholds vary widely (fold_thresholds.csv). RF/GB fall to the degenerate cut 0.01 (predict all
Incomplete) in {int(T.n_folds_thr_0_01.sum()) if 'n_folds_thr_0_01' in T else int(T['n_folds_thr_0.01'].sum())} fold-config cases.
Selection bias: the reported 82.4% is the maximum of 12 correlated estimates on the same 17 patients. The model, PCs
and threshold rule were chosen after looking at LOPOCV results, so 82.4% is an optimistic estimate. Sub-step 4 measures
how often a max-of-12 this high appears under permuted labels.
Files: results/step0/s3_v1_configs/{{configs_table, fold_thresholds, per_patient_probs}}.csv
"""
log("Sub-step 3 - All 12 v1 configurations", txt)
print(txt)
print(T[["model", "PCs", "threshold", "n_correct", "accuracy", "balanced_accuracy", "auc", "thr_min", "thr_max"]].round(3).to_string(index=False))
