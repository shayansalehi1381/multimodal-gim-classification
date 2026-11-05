"""Goal 1, Step A2 - per-class inspection, modality mix and shortcut baselines. No image model is trained.

Inputs: D:\\gim_outputs\\goal1\\image_stats.csv + sheets/sheet_index.csv (a1_image_stats.py), metadata.csv.
Seeds: SEED=42 (cap, CV), bootstrap default_rng(12345).
Visual modality labels below were assigned by Claude from 200-px contact sheets. They are NOT expert labels and NOT
blind (sheets were per class). Default in each list = ME-NBI, except LGIN where the default is WLI.
Categories: WLI, NBI_nonmag, ME_NBI, chromo (dye visible).
Outputs: results/goal1/*.csv
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact  # noqa: F401
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
OUT = "../results/goal1/"
D = "D:/gim_outputs/goal1/"
META = r"D:\dataset_gim\EndoWLI-NBI\EndoWLI-NBI\metadata.csv"

VIS = {
    "LGIN": dict(default="WLI", ME_NBI=[52, 54], chromo=[3, 39, 46, 51, 72, 93],
                 NBI_nonmag=[2, 7, 9, 10, 17, 43, 48, 53, 58, 61, 64, 65, 75, 85, 86, 87]),
    "IM&CG": dict(default="ME_NBI", WLI=[16, 41, 87], chromo=[65, 73, 85],
                  NBI_nonmag=[3, 4, 6, 7, 9, 11, 25, 31, 37, 39, 53, 78, 80, 86, 95]),
    "EGC": dict(default="ME_NBI", chromo=[15, 24, 27, 40], NBI_nonmag=[6]),
    "HGIN": dict(default="ME_NBI", WLI=[17, 32], chromo=[12, 27], NBI_nonmag=[91]),
}
COL = ["mean_r", "mean_g", "mean_b", "r_over_g", "g_over_b", "r_over_b", "sat", "val", "std_gray"]
COL_ME = COL + ["hf_ratio", "lap_var", "dark_frac", "glare_frac"]

df = pd.read_csv(D + "image_stats.csv")
meta = pd.read_csv(META).rename(columns={"ID": "patient", "Label": "meta_label"})
sheet = pd.read_csv(D + "sheets/sheet_index.csv")

# ── 1. per-class description ────────────────────────────────────────────────
n_img = df.groupby(["label", "patient"]).size().rename("n").reset_index()
s = n_img.groupby("label").n.agg(patients="size", images="sum", img_min="min", img_median="median", img_mean="mean", img_max="max")
g = df.groupby("label")
s = s.join(pd.DataFrame({
    "width_median": g.width.median(), "height_median": g.height.median(), "aspect_median": g.aspect.median(),
    "aspect_p05": g.aspect.quantile(.05), "aspect_p95": g.aspect.quantile(.95), "pct_width_1158": g.width.apply(lambda x: (x == 1158).mean() * 100),
    "bytes_median_kb": g.bytes.median() / 1024, "bytes_p05_kb": g.bytes.quantile(.05) / 1024, "bytes_p95_kb": g.bytes.quantile(.95) / 1024,
    "bytes_per_pixel_median": g.bytes_per_pixel.median(), "n_jpeg_tables": g.jpeg_tables.nunique(), "corner_dark_median": g.corner_dark.median()}))
s.round(3).to_csv(OUT + "class_summary.csv")

first = df.groupby("patient").agg(label=("label", "first"), year=("year", "first"), n=("file", "size")).reset_index()
pt = first.merge(meta, on="patient", how="left")
assert (pt.label == pt.meta_label).all() and pt.Sex.notna().all()
pt["date_year"] = pd.to_numeric(pt.Date.astype(str).str[:4], errors="coerce")
pt["month_idx"] = pt.date_year * 12 + pd.to_numeric(pt.Date.astype(str).str[5:7], errors="coerce")
pd.crosstab(pt.label, pt.year).to_csv(OUT + "year_by_class.csv")
pd.crosstab(df.label, df.year).to_csv(OUT + "year_by_class_images.csv")
dev = pt.Device.fillna("missing")
pd.crosstab(dev, pt.label, margins=True).to_csv(OUT + "device_by_class.csv")
a2_pts = pt[pt.year >= 2018]
pd.crosstab(dev[a2_pts.index], a2_pts.label, margins=True).to_csv(OUT + "device_by_class_2018_2021.csv")
year_mismatch = int((pt.date_year.notna() & (pt.date_year != pt.year)).sum())

# exact / perceptual duplicates across patients
dup_rows = []
for key in ("sha256", "dhash"):
    for k, grp in df.groupby(key):
        if grp.patient.nunique() > 1:
            dup_rows.append({"kind": key, "n_images": len(grp), "n_patients": grp.patient.nunique(),
                             "labels": "/".join(sorted(grp.label.unique())), "patients": "/".join(sorted(grp.patient.unique())[:6])})
dups = pd.DataFrame(dup_rows, columns=["kind", "n_images", "n_patients", "labels", "patients"])
dups.to_csv(OUT + "duplicates_cross_patient.csv", index=False)
within_exact = int(df.groupby("patient").sha256.apply(lambda x: x.duplicated().sum()).sum())
pd.crosstab(df.jpeg_tables, df.label).sort_values(by=["IM&CG"], ascending=False).head(15).to_csv(OUT + "jpeg_tables_by_class.csv")

# ── 2. visual labels and automatic colour check ─────────────────────────────
labs = []
for r in sheet.itertuples():
    v = VIS[r.label]
    lab = next((c for c in ("WLI", "NBI_nonmag", "ME_NBI", "chromo") if c != "default" and r.tile in v.get(c, [])), v["default"])
    labs.append(lab)
sheet["visual"] = labs
sheet = sheet.merge(df[["path"] + sorted(set(COL_ME))], on="path")
sheet.drop(columns="path").to_csv(OUT + "visual_labels.csv", index=False)
vt = pd.crosstab(sheet.label, sheet.visual)
vt.to_csv(OUT + "visual_modality_by_class.csv")

Xw, yw = sheet[COL].values, (sheet.visual == "WLI").astype(int).values
keep = sheet.visual.isin(["WLI", "NBI_nonmag", "ME_NBI"]).values
Xm, ym = sheet[COL_ME].values, (sheet.visual == "ME_NBI").astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)   # every sampled image is from a different patient
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
p_w = cross_val_predict(lr(), Xw[keep], yw[keep], cv=cv, method="predict_proba")[:, 1]
p_m = cross_val_predict(lr(), Xm, ym, cv=cv, method="predict_proba")[:, 1]
sub = sheet[keep]
rows = [("WLI vs NBI, colour LR (out-of-fold)", roc_auc_score(yw[keep], p_w), accuracy_score(yw[keep], p_w >= .5), balanced_accuracy_score(yw[keep], p_w >= .5)),
        ("WLI vs NBI, r_over_g alone", roc_auc_score(yw[keep], sub.r_over_g), np.nan, np.nan),
        ("ME_NBI vs all others, colour+texture LR (out-of-fold)", roc_auc_score(ym, p_m), accuracy_score(ym, p_m >= .5), balanced_accuracy_score(ym, p_m >= .5))]
nbi = sub.visual.isin(["NBI_nonmag", "ME_NBI"]).values
rows.append(("ME_NBI vs NBI_nonmag, hf_ratio alone", roc_auc_score((sub.visual == "ME_NBI")[nbi], sub.hf_ratio[nbi]), np.nan, np.nan))
rows.append(("ME_NBI vs NBI_nonmag, lap_var alone", roc_auc_score((sub.visual == "ME_NBI")[nbi], sub.lap_var[nbi]), np.nan, np.nan))
pd.DataFrame(rows, columns=["check", "auc", "accuracy", "balanced_accuracy"]).round(3).to_csv(OUT + "colour_check_vs_visual.csv", index=False)
# agreement inside each class (is the proxy just a class detector?)
ag = pd.DataFrame({"label": sheet.label, "visual_ME": ym, "pred_ME": (p_m >= .5).astype(int)})
ag.groupby("label").apply(lambda x: pd.Series({"visual_ME_pct": x.visual_ME.mean() * 100, "pred_ME_pct": x.pred_ME.mean() * 100, "agreement_pct": (x.visual_ME == x.pred_ME).mean() * 100})).round(1).to_csv(OUT + "colour_check_within_class.csv")

m_w, m_m = lr().fit(Xw[keep], yw[keep]), lr().fit(Xm, ym)
df["p_wli"] = m_w.predict_proba(df[COL].values)[:, 1]
df["p_me"] = m_m.predict_proba(df[COL_ME].values)[:, 1]
df["pred_me"] = df.p_me >= .5
df["pred_wli"] = df.p_wli >= .5
auto = df.groupby("label").agg(images=("file", "size"), pred_WLI_pct=("pred_wli", lambda x: x.mean() * 100), pred_ME_NBI_pct=("pred_me", lambda x: x.mean() * 100))
pm = df.groupby(["label", "patient"]).pred_me.agg(["mean", "sum"]).reset_index()
auto = auto.join(pm.groupby("label").agg(patient_median_ME_frac=("mean", "median"), patients_ge3_ME=("sum", lambda x: (x >= 3).sum()),
                                         patients_ge1_ME=("sum", lambda x: (x >= 1).sum())))
y18 = pm.merge(first[["patient", "year"]], on="patient")
auto["patients_ge3_ME_2018_2021"] = y18[(y18.year >= 2018) & (y18["sum"] >= 3)].groupby("label").size()
auto.round(2).to_csv(OUT + "modality_auto_by_class.csv")
df[["label", "patient", "file", "p_wli", "p_me"]].to_csv(D + "image_modality_proxy.csv", index=False)

# ── 3. shortcut baselines (one row per patient; no image model) ─────────────
capped = df.sample(frac=1, random_state=SEED).groupby("patient").head(10)


def patient_features(sample, pts):
    a = sample.groupby("patient")
    f = pd.DataFrame(index=pts.patient)
    agg = lambda c, fn: a[c].agg(fn).reindex(f.index).fillna(0).values
    G = {}
    G["resolution_aspect"] = np.c_[agg("width", "mean"), agg("height", "mean"), agg("aspect", "mean"), agg("aspect", "std"),
                                   a.width.apply(lambda x: (x == 1158).mean()).reindex(f.index).fillna(0).values]
    G["file_size"] = np.c_[agg("bytes", "mean"), agg("bytes", "std"), agg("bytes_per_pixel", "mean")]
    G["images_per_patient_capped"] = np.log1p(a.size().reindex(f.index).fillna(0).values)[:, None]
    G["colour_modality_proxy"] = np.c_[agg("r_over_g", "mean"), agg("r_over_g", "std"), agg("g_over_b", "mean"), agg("sat", "mean"),
                                       agg("val", "mean"), agg("std_gray", "mean"), agg("hf_ratio", "mean"), agg("lap_var", "mean")]
    G["border_glare"] = np.c_[agg("corner_dark", "mean"), agg("dark_frac", "mean"), agg("glare_frac", "mean")]
    top = df.jpeg_tables.value_counts().index[:8]
    G["jpeg_tables"] = np.c_[[sample[sample.jpeg_tables == t].groupby("patient").size().reindex(f.index).fillna(0).values / np.maximum(a.size().reindex(f.index).fillna(1).values, 1) for t in top]].T
    G["images_per_patient_uncapped"] = np.log1p(pts.n.values)[:, None]
    G["year"] = pd.get_dummies(pts.year).values.astype(float)
    G["year_month"] = np.c_[pd.get_dummies(pts.year).values.astype(float), pts.month_idx.fillna(pts.month_idx.median()).values]
    G["device_metadata"] = pd.get_dummies(pts.Device.fillna("missing").str.split(",").str[0]).values.astype(float)
    age = pts.Age.str[:2].astype(float).values
    G["sex_age_metadata"] = np.c_[(pts.Sex == "Male").values.astype(float), age]
    G["ALL_image_visible"] = np.hstack([G[k] for k in ("resolution_aspect", "file_size", "colour_modality_proxy", "border_glare", "jpeg_tables")])
    G["image_visible_no_colour"] = np.hstack([G[k] for k in ("resolution_aspect", "file_size", "border_glare", "jpeg_tables")])
    return G


def boot(y, P, multi, B=1000, seed=12345):
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(y == c) for c in np.unique(y)]
    A, Ba = [], []
    for _ in range(B):
        i = np.concatenate([rng.choice(c, len(c)) for c in idx])
        if multi:
            A.append(roc_auc_score(y[i], P[i], multi_class="ovr", average="macro"))
            Ba.append(balanced_accuracy_score(y[i], P[i].argmax(1)))
        else:
            A.append(roc_auc_score(y[i], P[i, 1]))
            Ba.append(balanced_accuracy_score(y[i], P[i, 1] >= .5))
    return np.percentile(A, [2.5, 97.5]), np.percentile(Ba, [2.5, 97.5])


def run(X, y, model):
    """Patient-level 5-fold stratified CV repeated 5x (each row = one patient, so groups cannot cross folds).
    Scaler and model fit inside the training folds. Returns averaged out-of-fold probabilities."""
    n_c = len(np.unique(y))
    P = np.zeros((len(y), n_c))
    for rep in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(X, y):
            if model == "LR":
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
            else:
                m = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced", random_state=SEED + rep, n_jobs=-1)
            P[te] += m.fit(X[tr], y[tr]).predict_proba(X[te]) / 5
    return P


tasks = {}
p18 = pt[pt.year >= 2018].reset_index(drop=True)
tasks["A2 IM&CG vs LGIN+HGIN+EGC, 2018-2021"] = (p18, (p18.label == "IM&CG").astype(int).values, capped, False)
pall = pt.reset_index(drop=True)
tasks["A2 all years (only to show the year effect)"] = (pall, (pall.label == "IM&CG").astype(int).values, capped, False)
p18n = p18[p18.label != "LGIN"].reset_index(drop=True)
tasks["A2 without LGIN, 2018-2021"] = (p18n, (p18n.label == "IM&CG").astype(int).values, capped, False)
me_img = df[df.pred_me]
me_cnt = me_img.groupby("patient").size()
pme = p18[p18.patient.map(me_cnt).fillna(0) >= 3].reset_index(drop=True).copy()
pme["n"] = pme.patient.map(me_cnt)
tasks["A2 predicted ME-NBI images only, patients with >=3, 2018-2021"] = (
    pme, (pme.label == "IM&CG").astype(int).values, me_img.sample(frac=1, random_state=SEED).groupby("patient").head(10), False)
lab4 = {"EGC": 0, "HGIN": 1, "LGIN": 2, "IM&CG": 3}
tasks["B 4-class, all years"] = (pall, pall.label.map(lab4).values, capped, True)
tasks["B 4-class, 2018-2021"] = (p18, p18.label.map(lab4).values, capped, True)

res = []
for tname, (P_, y, sample, multi) in tasks.items():
    G = patient_features(sample, P_)
    for gname, X in G.items():
        if X.shape[1] == 0 or (gname == "year" and P_.year.nunique() < 2):
            continue
        for model in ("LR", "RF"):
            P = run(X, y, model)
            if multi:
                auc, bac = roc_auc_score(y, P, multi_class="ovr", average="macro"), balanced_accuracy_score(y, P.argmax(1))
            else:
                auc, bac = roc_auc_score(y, P[:, 1]), balanced_accuracy_score(y, P[:, 1] >= .5)
            (al, ah), (bl, bh) = boot(y, P, multi)
            res.append({"task": tname, "shortcut": gname, "model": model, "n_patients": len(y),
                        "class_counts": "/".join(str(int((y == c).sum())) for c in np.unique(y)), "auc": auc, "auc_lo": al, "auc_hi": ah,
                        "bal_acc": bac, "bal_acc_lo": bl, "bal_acc_hi": bh})
    print("done", tname, flush=True)
R = pd.DataFrame(res).round(3)
R.to_csv(OUT + "shortcut_baselines.csv", index=False)

print(s.round(2).T.to_string())
print(vt)
print(pd.read_csv(OUT + "colour_check_vs_visual.csv").to_string(index=False))
print(pd.read_csv(OUT + "colour_check_within_class.csv").to_string(index=False))
print(auto.round(2).to_string())
print("Date-year vs folder-year mismatches:", year_mismatch, " date missing:", int(pt.date_year.isna().sum()))
print("cross-patient dup groups:", dups.groupby("kind").size().to_dict(), " within-patient exact dups:", within_exact)
print(pd.crosstab(dev, pt.label, margins=True).to_string())
print(pd.crosstab(dev[a2_pts.index], a2_pts.label, margins=True).to_string())
print(R.sort_values(["task", "auc"], ascending=[True, False]).groupby("task").head(6)[["task", "shortcut", "model", "class_counts", "auc", "auc_lo", "auc_hi", "bal_acc"]].to_string(index=False))
