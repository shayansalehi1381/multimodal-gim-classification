"""Sub-step 6 - confounder analysis.

Patient-level factors (cohort n=17, from s0 patient_inventory): visual source used by v1 (stills vs video frames),
file type (jpg stills / png stills / video frames), processor token in filename (X1 vs 190), and
on-screen display (OSD text panel). In the cohort: present in png stills and video screen-captures, absent in jpg stills.
Outside the cohort, controls 2623/2624 have no panel.
 (1) contingency tables label x factor (+ Fisher exact where 2x2)
 (2) stratified performance: OOF predictions of the v1 champion and the 15 nested step-0 models restricted to
     each subgroup (acc / balanced acc / AUC where both classes exist)
 (3) re-training inside subgroups (whole pipeline re-run on the subgroup only): image-only (n=12),
     X1-only (n=15, drops 2614/2615), no-OSD jpg-stills only (n=9), and "drop 190 + png" (n=14)
 (4) shortcut probe: can a linear model read acquisition factors from the embeddings?
     image level, grouped leave-one-patient-out (all 24 patients), L2-LR on in-fold scaled + PCA(16) embeddings,
     raw (v1-style) vs FOV-cropped embeddings: still-vs-video-frame, png-vs-jpg (stills), 190-vs-X1 (stills),
     OSD-vs-none. Patient level (cohort 17, v1 pooled features): nested LR predicting source / OSD.
Outputs: results/step0/s6_confounders/*.csv
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import (FEATURE_SETS, PROC, RES, ROOT, feature_blocks, load_cohort, log, metrics, out_dir,
                    run_nested_lopo, v1_run)

warnings.filterwarnings("ignore")
OUT = out_dir("s6_confounders")


def safe_metrics(y, s, p):
    r = {"n": len(y), "n_incomplete": int(y.sum()), "n_correct": int((y == p).sum()), "accuracy": accuracy_score(y, p)}
    if len(np.unique(y)) == 2:
        r["balanced_accuracy"] = balanced_accuracy_score(y, p)
        r["auc"] = roc_auc_score(y, s)
    return r


if __name__ == "__main__":
    df, y, b = load_cohort()
    cases = df["Case Code"].values
    pinv = pd.read_csv(RES / "s0_integrity" / "patient_inventory.csv").set_index("Case Code")
    fac = pd.DataFrame(index=cases)
    fac["label"] = np.where(y == 1, "Incomplete", "Complete")
    fac["source"] = pinv.loc[cases, "v1_visual_source"].values
    fac["file_type"] = np.where(fac.source == "video frames only", "video frames",
                                np.where(pinv.loc[cases, "img_ext"].values == ".png", "png stills", "jpg stills"))
    fac["processor"] = pinv.loc[cases, "processor_filename"].values
    fac["OSD_text_panel"] = np.where(fac.file_type == "jpg stills", "absent", "present")
    fac.index.name = "Case Code"
    fac.to_csv(OUT / "patient_factors.csv")

    # (1) contingency
    ct_rows = []
    for col in ("source", "file_type", "processor", "OSD_text_panel"):
        ct = pd.crosstab(fac[col], fac["label"])
        p = fisher_exact(ct.values)[1] if ct.shape == (2, 2) else np.nan
        for lvl, r in ct.iterrows():
            ct_rows.append({"factor": col, "level": lvl, "Complete": r.get("Complete", 0),
                            "Incomplete": r.get("Incomplete", 0), "fisher_p_2x2": p})
    CT = pd.DataFrame(ct_rows)
    CT.to_csv(OUT / "contingency_tables.csv", index=False)

    # (2) stratified OOF performance
    v1p = pd.read_csv(RES / "s3_v1_configs" / "per_patient_probs.csv")
    v1t = pd.read_csv(RES / "s3_v1_configs" / "fold_thresholds.csv")
    sv = v1p[(v1p.model == "SVC") & (v1p.PCs == 8)].set_index("Case Code").loc[cases]
    th = v1t[(v1t.model == "SVC") & (v1t.PCs == 8)].set_index("Case Code").loc[cases].youden_threshold.values
    preds = {"v1_champion_SVC8": (sv.prob_incomplete.values, (sv.prob_incomplete.values >= th).astype(int))}
    s2 = pd.read_csv(RES / "s2_baselines" / "per_patient_scores.csv")
    for (fs, m), g in s2.groupby(["feature_set", "model"], sort=False):
        g = g.set_index("Case Code").loc[cases]
        preds[f"{fs}/{m}"] = (g.score.values, g.pred.values)
    strat = []
    for name, (s, p) in preds.items():
        strat.append({"model": name, "subgroup": "all", **safe_metrics(y, s, p)})
        for col in ("source", "file_type", "processor", "OSD_text_panel"):
            for lvl in fac[col].unique():
                msk = (fac[col] == lvl).values
                strat.append({"model": name, "subgroup": f"{col}={lvl}", **safe_metrics(y[msk], s[msk], p[msk])})
    ST = pd.DataFrame(strat)
    ST.to_csv(OUT / "stratified_performance.csv", index=False)

    # (3) re-training inside subgroups
    subsets = {
        "image_only_n12": (fac.source == "stills only").values,
        "X1_only_n15": (fac.processor == "X1").values,
        "jpg_stills_noOSD_n9": (fac.file_type == "jpg stills").values,
        "drop_190_and_png_n14": ~fac.index.isin([2613, 2614, 2615]),
    }
    Xc_all, Xv_all = b["clinical"], np.hstack([b["biomedclip"], b["dinov2"]])
    sub_rows = []
    for sname, msk in subsets.items():
        ys = y[msk]
        try:
            prob, thr = v1_run(Xc_all[msk], Xv_all[msk], ys, "SVC", 8)
            sub_rows.append({"subset": sname, "model": "v1_champion_SVC8 (v1 protocol)", **safe_metrics(ys, prob, (prob >= thr).astype(int))})
        except ValueError as ex:   # StratifiedKFold(4) impossible with <4 per class in a training fold
            sub_rows.append({"subset": sname, "model": "v1_champion_SVC8 (v1 protocol)", "n": len(ys),
                             "n_incomplete": int(ys.sum()), "note": f"not estimable: {str(ex)[:80]}"})
        for fs in ("a_clinical", "b_biomedclip", "c_dinov2", "e_visual+clinical"):
            Xc, Xv = feature_blocks(b, fs)
            for m in ("LR", "SVC"):
                r = run_nested_lopo(Xc[msk] if Xc is not None else None, Xv[msk] if Xv is not None else None, ys, m)
                sub_rows.append({"subset": sname, "model": f"{fs}/{m} (nested)", **safe_metrics(ys, r.score.values, r.pred.values)})
        print("subset done:", sname, flush=True)
    SUB = pd.DataFrame(sub_rows)
    SUB.to_csv(OUT / "subgroup_retrained.csv", index=False)

    # (4a) image-level shortcut probe (raw vs FOV-cropped)
    idx_c = pd.read_csv(ROOT / "step0" / "embeddings_cropped" / "index.csv")
    idx_r = pd.read_csv(ROOT / "step0" / "embeddings_raw" / "index_stills_raw.csv")
    emb = {("biomedclip", "cropped"): np.load(ROOT / "step0" / "embeddings_cropped" / "biomedclip_cropped.npy"),
           ("dinov2", "cropped"): np.load(ROOT / "step0" / "embeddings_cropped" / "dinov2_cropped.npy")}
    # raw: stills recomputed uncropped; video frames = v1 frame embeddings (uncropped)
    fm = pd.read_csv(PROC / "frame_metadata.csv")
    dm = pd.read_csv(PROC / "dinov2_frame_metadata.csv")
    v1b, v1d = np.load(PROC / "frame_embeddings_1976x512.npy"), np.load(PROC / "dinov2_frame_embeddings_2016x768.npy")
    rawb, rawd = np.load(ROOT / "step0" / "embeddings_raw" / "biomedclip_stills_raw.npy"), np.load(ROOT / "step0" / "embeddings_raw" / "dinov2_stills_raw.npy")
    fpos_b = dict(zip(fm.filename, fm.embedding_idx))
    fpos_d = dict(zip(dm.filename, dm.embedding_idx))
    raw_pos = dict(zip(idx_r.key, range(len(idx_r))))
    rb, rd = [], []
    for _, r in idx_c.iterrows():
        if r.kind == "still":
            rb.append(rawb[raw_pos[r.key]]); rd.append(rawd[raw_pos[r.key]])
        else:
            rb.append(v1b[fpos_b[r.key]]); rd.append(v1d[fpos_d[r.key]])
    emb[("biomedclip", "raw")], emb[("dinov2", "raw")] = np.array(rb), np.array(rd)

    meta = idx_c.copy()
    proc = pd.read_csv(RES / "s0_integrity" / "media_inventory.csv").set_index("file")
    meta["is_still"] = (meta.kind == "still").astype(int)
    meta["ext"] = np.where(meta.kind == "still", meta.key.str[-4:].str.lower(), ".vid")
    meta["is_190"] = meta.key.str.contains("_190_").astype(int)
    # corrected 2026-11-05: panel depends on capture layout (controls 2623/2624 have none); see s6b_osd_probe_corrected.py
    meta["has_OSD"] = meta.layout.isin({"1280x720", "1920x1080", "2879x1799", "2880x1800", "650x497", "707x566"}).astype(int)
    rng = np.random.default_rng(42)
    keep_frames = meta[meta.kind == "video_frame"].groupby("case").apply(
        lambda g: g.sample(min(len(g), 20), random_state=42)).index.get_level_values(-1)
    sel_all = np.sort(np.concatenate([meta.index[meta.kind == "still"].values, keep_frames.values]))
    probes = {
        "still_vs_video_frame": (sel_all, "is_still"),
        "png_vs_jpg_stills": (meta.index[meta.kind == "still"].values, "ext"),
        "190_vs_X1_stills": (meta.index[meta.kind == "still"].values, "is_190"),
        "OSD_vs_none": (sel_all, "has_OSD"),
    }

    def grouped_probe(X, yb, groups):
        sc = np.zeros(len(yb))
        for g in np.unique(groups):
            te, tr = groups == g, groups != g
            if len(np.unique(yb[tr])) < 2:
                sc[te] = np.nan
                continue
            s = StandardScaler().fit(X[tr])
            pca = PCA(n_components=min(16, tr.sum() - 1), svd_solver="full", random_state=42).fit(s.transform(X[tr]))
            m = LogisticRegression(C=1.0, max_iter=5000, random_state=42).fit(pca.transform(s.transform(X[tr])), yb[tr])
            sc[te] = m.predict_proba(pca.transform(s.transform(X[te])))[:, 1]
        ok = ~np.isnan(sc)
        return roc_auc_score(yb[ok], sc[ok]), balanced_accuracy_score(yb[ok], (sc[ok] >= 0.5).astype(int))

    pr = []
    for pname, (rows, col) in probes.items():
        yb = meta.loc[rows, col].values
        yb = (yb == ".png").astype(int) if col == "ext" else yb.astype(int)
        groups = meta.loc[rows, "case"].values
        for (mdl, ver), E in emb.items():
            auc, bacc = grouped_probe(E[rows], yb, groups)
            pr.append({"probe": pname, "embedding": mdl, "version": ver, "n_items": len(rows),
                       "n_positive": int(yb.sum()), "n_patients": len(np.unique(groups)),
                       "grouped_LOPO_auc": auc, "grouped_LOPO_bacc@0.5": bacc})
    PR = pd.DataFrame(pr)

    # (4b) patient-level probe on the v1 pooled cohort features (nested LR, same pipeline as sub-step 2)
    for target, yy in (("source_video_vs_stills", (fac.source == "video frames only").astype(int).values),
                       ("OSD_present", (fac.OSD_text_panel == "present").astype(int).values)):
        for fs in ("b_biomedclip", "c_dinov2", "d_both_visual"):
            Xc, Xv = feature_blocks(b, fs)
            r = run_nested_lopo(None, Xv, yy, "LR")
            pr_row = {"probe": f"patient_{target}", "embedding": fs, "version": "raw (v1 pooled)", "n_items": 17,
                      "n_positive": int(yy.sum()), "n_patients": 17, "grouped_LOPO_auc": roc_auc_score(yy, r.score),
                      "grouped_LOPO_bacc@0.5": balanced_accuracy_score(yy, r.pred)}
            PR = pd.concat([PR, pd.DataFrame([pr_row])], ignore_index=True)
    PR.to_csv(OUT / "shortcut_probe.csv", index=False)

    champ = ST[ST.model == "v1_champion_SVC8"].set_index("subgroup")
    txt = f"""
Contingency (cohort n=17; contingency_tables.csv): processor 190 = 2 Incomplete / 0 Complete; png stills = 3 Incomplete / 0 Complete;
OSD text panel present = {CT[(CT.factor=='OSD_text_panel')&(CT.level=='present')].Incomplete.iloc[0]} Inc / {CT[(CT.factor=='OSD_text_panel')&(CT.level=='present')].Complete.iloc[0]} Comp,
absent (jpg stills) = {CT[(CT.factor=='OSD_text_panel')&(CT.level=='absent')].Incomplete.iloc[0]} Inc / {CT[(CT.factor=='OSD_text_panel')&(CT.level=='absent')].Complete.iloc[0]} Comp; stills-only = 7 Inc / 5 Comp, video = 2 Inc / 3 Comp.
v1 champion, stratified OOF (stratified_performance.csv): {'; '.join(f"{k}: {int(v.n_correct)}/{int(v.n)}" for k, v in champ.iterrows())}.
Retrained inside subgroups (subgroup_retrained.csv): see table; summary in STEP0_REPORT.
Shortcut probe (shortcut_probe.csv), grouped leave-one-patient-out AUC, raw -> cropped:
""" + "\n".join(
        f"  - {p}/{e}: raw {PR[(PR.probe==p)&(PR.embedding==e)&(PR.version=='raw')].grouped_LOPO_auc.iloc[0]:.3f} -> cropped "
        f"{PR[(PR.probe==p)&(PR.embedding==e)&(PR.version=='cropped')].grouped_LOPO_auc.iloc[0]:.3f}"
        for p in probes for e in ("biomedclip", "dinov2")) + "\n" + "\n".join(
        f"  - {r.probe}/{r.embedding}: AUC {r.grouped_LOPO_auc:.3f}" for r in PR[PR.probe.str.startswith('patient_')].itertuples())
    log("Sub-step 6 - Confounder analysis and shortcut probe", txt)
    print(txt)
    print(SUB.round(3).to_string(index=False))
