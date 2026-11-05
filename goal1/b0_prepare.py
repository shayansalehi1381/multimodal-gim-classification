"""Goal 1, Step B preparation - no model evaluation here, no diagnosis label used to set the filter.

1. Modality filter: colour LR for WLI vs non-WLI (visual labels only, chromo excluded). The threshold is Youden's J on
   out-of-fold predictions against the visual modality labels; it never sees diagnosis or model performance.
2. Exact within-patient duplicates removed (same SHA-256 inside one patient) BEFORE capping.
3. Capped list drawn ONCE (default_rng(42), at most 10 images per patient), saved to results/goal1/capped_list.csv.

Datasets:
  A2-NBI : 2018-2021, non-WLI images, patients with >= 3 images left. IM&CG vs LGIN+HGIN+EGC.
  S1     : 2018-2021, no modality filter. IM&CG vs LGIN+HGIN+EGC.
  S2     : S1 without LGIN (same drawn images). IM&CG vs HGIN+EGC.
  S3     : A2-NBI without LGIN (same drawn images). IM&CG vs HGIN+EGC. Added before any evaluation because review of the
           primary images showed residual dye/forceps/white-light concentrated in LGIN.
Outputs: results/goal1/{modality_filter.json, capped_list.csv, dataset_counts.csv}
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = "../results/goal1/"
D = "D:/gim_outputs/goal1/"
COL = ["mean_r", "mean_g", "mean_b", "r_over_g", "g_over_b", "r_over_b", "sat", "val", "std_gray"]
MIN_IMAGES_FILTERED, CAP = 3, 10

df = pd.read_csv(D + "image_stats.csv")
vis = pd.read_csv(OUT + "visual_labels.csv")
vis = vis[vis.visual.isin(["WLI", "NBI_nonmag", "ME_NBI"])]
X, y = vis[COL].values, (vis.visual == "WLI").astype(int).values
make = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
oof = cross_val_predict(make(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
cuts = np.unique(oof)
cuts = (cuts[:-1] + cuts[1:]) / 2
j = np.array([(oof[y == 1] >= c).mean() + (oof[y == 0] < c).mean() - 1 for c in cuts])
best = np.flatnonzero(np.isclose(j, j.max()))
thr = float(cuts[best[len(best) // 2]])
model = make().fit(X, y)
df["p_wli"] = model.predict_proba(df[COL].values)[:, 1]
filt = {"features": COL, "n_labelled": int(len(y)), "n_wli": int(y.sum()), "oof_auc": float(roc_auc_score(y, oof)),
        "threshold": thr, "youden_j": float(j.max()),
        "sensitivity_wli": float((oof[y == 1] >= thr).mean()), "specificity_wli": float((oof[y == 0] < thr).mean())}
json.dump(filt, open(OUT + "modality_filter.json", "w"), indent=1)
print(filt)

n_dup_all = int(df.duplicated(["patient", "sha256"]).sum())
a = df[df.year >= 2018].sort_values(["patient", "file"])
n_before = len(a)
a = a.drop_duplicates(["patient", "sha256"])
n_dup_window = n_before - len(a)
print("exact within-patient duplicates: all years", n_dup_all, "| 2018-2021", n_dup_window)


def draw(sub, rng):
    rows = []
    for pid, g in sub.groupby("patient", sort=True):
        take = rng.choice(len(g), min(CAP, len(g)), replace=False)
        rows.append(g.iloc[np.sort(take)])
    return pd.concat(rows)


rng = np.random.default_rng(SEED)
f = a[a.p_wli < thr]
cnt = f.groupby("patient").size()
f = f[f.patient.map(cnt) >= MIN_IMAGES_FILTERED]
primary = draw(f, rng).assign(dataset="A2-NBI")
s1 = draw(a, rng).assign(dataset="S1")
s2 = s1[s1.label != "LGIN"].assign(dataset="S2")
s3 = primary[primary.label != "LGIN"].assign(dataset="S3")
cl = pd.concat([primary, s1, s2, s3])[["dataset", "label", "patient", "year", "file", "path", "sha256", "p_wli"]]
cl["y"] = (cl.label == "IM&CG").astype(int)         # positive class = IM&CG
cl.to_csv(OUT + "capped_list.csv", index=False)

rows = []
for name, g in cl.groupby("dataset"):
    for lab, h in g.groupby("label"):
        rows.append({"dataset": name, "label": lab, "patients": h.patient.nunique(), "images": len(h),
                     "img_per_patient_min": h.groupby("patient").size().min(), "img_per_patient_median": h.groupby("patient").size().median()})
c = pd.DataFrame(rows)
c.to_csv(OUT + "dataset_counts.csv", index=False)
print(c.to_string(index=False))
print(cl.groupby("dataset").agg(patients=("patient", "nunique"), images=("file", "size"), pos=("y", "sum")))
print("unique images across datasets:", cl.path.nunique())
assert cl.groupby(["dataset", "patient"]).y.nunique().max() == 1
assert cl.groupby(["dataset", "patient"]).sha256.apply(lambda s: s.duplicated().sum()).sum() == 0
