"""Goal 1, Step B-0 - shortcut baselines on A2-NBI, S1, S2, S3. No image model.

Features are patient-level summaries of the images in capped_list.csv (plus year). Same outer folds as the image
models, same LR protocol (inner CV for C, Youden threshold), plus an untuned RF. Everything from config_B.json.
Outputs: results/goal1/shortcut_baselines_B0.csv, best_shortcut.json; D:/gim_outputs/goal1/baselines/*.npz
Exit code 3 if any baseline AUC exceeds the stop limit in the config (stop and report).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from b_common import OUT, Unexpected, bootstrap, load_config, outer_fold, outer_splits, summarise, youden

warnings.filterwarnings("ignore")
SAVE = "D:/gim_outputs/goal1/baselines/"
os.makedirs(SAVE, exist_ok=True)


def features(d, stats):
    s = d.merge(stats, on="path", how="left", suffixes=("", "_s"))
    assert s.bytes.notna().all()
    a = s.groupby("patient")
    pts = sorted(d.patient.unique())
    m = lambda c: a[c].mean().reindex(pts).values
    sd = lambda c: a[c].std().reindex(pts).fillna(0).values
    G = {"colour_stats": np.c_[m("r_over_g"), sd("r_over_g"), m("g_over_b"), m("r_over_b"), m("sat"), m("val"), m("std_gray"), m("hf_ratio"), m("lap_var")],
         "file_size": np.c_[m("bytes"), sd("bytes"), m("bytes_per_pixel")],
         "resolution_aspect": np.c_[m("width"), m("height"), m("aspect"), sd("aspect"), a.width.apply(lambda x: (x == 1158).mean()).reindex(pts).values],
         "images_per_patient": np.log1p(a.size().reindex(pts).values)[:, None],
         "year": pd.get_dummies(a.year.first().reindex(pts)).values.astype(float)}
    G["ALL_five_groups"] = np.hstack(list(G.values()))
    return pts, G


def rf_fold(X, y, tr, te, seed, cfg):
    r = cfg["shortcuts"]["rf"]
    mk = lambda: RandomForestClassifier(n_estimators=r["n_estimators"], min_samples_leaf=r["min_samples_leaf"], class_weight="balanced",
                                        random_state=cfg["seeds"]["model"], n_jobs=1)
    o = np.zeros(len(tr))
    for a, b in StratifiedKFold(cfg["cv"]["inner_folds"], shuffle=True, random_state=seed).split(tr, y[tr]):
        o[b] = mk().fit(X[tr[a]], y[tr[a]]).predict_proba(X[tr[b]])[:, 1]
    return mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], youden(y[tr], o)


def one(name, gname, kind, X, y, pts, splits, cfg):
    warnings.filterwarnings("ignore")
    R = cfg["cv"]["repeats"]
    P, TH = np.zeros((R, len(pts))), np.zeros((R, len(pts)))
    pa = np.array(pts)
    for r, k, tr, te in splits:
        seed = cfg["seeds"]["inner_base"] + 100 * r + k
        if kind == "LR":
            pt, ps, thr, *_ = outer_fold(X, y, pa, tr, te, seed, cfg)
            pos = np.searchsorted(pa, pt)
            P[r, pos], TH[r, pos] = ps, thr
        else:
            ps, thr = rf_fold(X, y, tr, te, seed, cfg)
            P[r, te], TH[r, te] = ps, thr
    auc, bacc, aucs, _ = summarise(y, P, TH)
    b = bootstrap(y, P, TH, cfg["bootstrap"]["B"], cfg["seeds"]["bootstrap"])
    np.savez(f"{SAVE}{name}__{gname}__{kind}.npz", patients=pa, y=y, P=P, TH=TH)
    return {"dataset": name, "shortcut": gname, "model": kind, "n_patients": len(y), "n_pos": int(y.sum()), "auc": auc, "auc_lo": b["auc_ci"][0],
            "auc_hi": b["auc_ci"][1], "bal_acc": bacc, "bal_acc_lo": b["bacc_ci"][0], "bal_acc_hi": b["bacc_ci"][1],
            "auc_repeat_min": min(aucs), "auc_repeat_max": max(aucs)}


if __name__ == "__main__":
    cfg = load_config()
    if cfg is None:
        raise Unexpected("config_B.json must be frozen before any evaluation")
    cl = pd.read_csv(OUT / "capped_list.csv")
    stats = pd.read_csv("D:/gim_outputs/goal1/image_stats.csv").drop(columns=["label", "patient", "file", "year", "sha256"])
    tasks = []
    for name in cfg["datasets"]:
        d = cl[cl.dataset == name].sort_values(["patient", "file"]).reset_index(drop=True)
        pts, G = features(d, stats)
        ymap = d.groupby("patient").y.first()
        y = ymap.reindex(pts).values.astype(int)
        pa = np.array(pts)
        splits = []
        for r in range(cfg["cv"]["repeats"]):
            for k, (tr, te) in enumerate(outer_splits(d, d.y.values, cfg["cv"]["outer_folds"], cfg["seeds"]["outer_base"] + r)):
                tep = set(d.patient.values[te])
                mask = np.array([p in tep for p in pa])
                splits.append((r, k, np.flatnonzero(~mask), np.flatnonzero(mask)))
        for gname, X in G.items():
            if not np.isfinite(X).all():
                raise Unexpected(f"non-finite baseline features: {name}/{gname}")
            for kind in ("LR", "RF"):
                tasks.append((name, gname, kind, X, y, pts, splits, cfg))
    res = Parallel(n_jobs=cfg["compute"]["cpu_workers"], verbose=2)(delayed(one)(*t) for t in tasks)
    R = pd.DataFrame(res).round(4)
    R.to_csv(OUT / "shortcut_baselines_B0.csv", index=False)
    best = {n: R[R.dataset == n].sort_values("auc", ascending=False).iloc[0].to_dict() for n in cfg["datasets"]}
    json.dump(best, open(OUT / "best_shortcut.json", "w"), indent=1, default=float)
    print(R[["dataset", "shortcut", "model", "n_pos", "n_patients", "auc", "auc_lo", "auc_hi", "bal_acc"]].sort_values(["dataset", "auc"], ascending=[True, False]).to_string(index=False))
    if R.auc.max() > cfg["stop_conditions"]["shortcut_auc_above"]:
        print("STOP: a shortcut baseline exceeds the stop limit")
        sys.exit(3)
