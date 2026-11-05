"""Goal 1 - EXPLORATORY artefact test, step 2: can image areas or file properties WITHOUT tissue predict the class?

Patient-level features from b6_artefact_features.py. Same outer folds, LR protocol and untuned RF as the shortcut baselines.
Reading rule fixed before running, for tissue-free groups (best AUC over LR/RF):
  >= 0.85 strong evidence of an artefact; 0.65-0.85 partial; < 0.65 no evidence in these areas (this does NOT prove
  the embedding signal is tissue, because an artefact may exist only inside tissue pixels).
Low-level groups are not tissue-free, so they cannot separate artefact from tissue on their own.
Outputs: results/goal1/artefact_test.csv, artefact_test_by_class.csv, artefact_permutation.csv
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from b2_baselines import rf_fold
from b_common import OUT, bootstrap, load_config, outer_fold, outer_splits, summarise

warnings.filterwarnings("ignore")
GROUPS = {
    "corner_value": (["corner_val_r", "corner_val_g", "corner_val_b", "corner_max", "corner_nonblack"], True),
    "corner_noise": (["corner_std_r", "corner_std_g", "corner_std_b", "corner_frac_zero", "corner_frac_le2"], True),
    "corner_geometry": (["corner_diag_min", "corner_diag_median"], True),
    "file_header": (["hdr_len"] + [f"hdr_top{i}" for i in range(8)], True),
    "histogram_comb": (["comb_r", "comb_g", "comb_b", "empty_r", "empty_g", "empty_b"], False),
    "blockiness": (["blockiness"], False),
    "global_noise": (["noise_r", "noise_g", "noise_b"], False),
}
LUMEN = ["lumen_mean_r", "lumen_mean_g", "lumen_mean_b", "lumen_std_r", "lumen_std_g", "lumen_std_b", "lumen_hp"]


def patient_matrix(d):
    a = d.groupby("patient")
    pts = sorted(d.patient.unique())
    cols = [c for g, _ in GROUPS.values() for c in g]
    M = pd.concat([a[cols].mean().add_suffix("__mean"), a[cols].std().fillna(0).add_suffix("__std")], axis=1).reindex(pts)
    L = a[LUMEN].mean().reindex(pts)
    return pts, M, L


def splits_for(d, pts, cfg, keep=None):
    pa = np.array(pts)
    out = []
    for r in range(cfg["cv"]["repeats"]):
        for k, (tr, te) in enumerate(outer_splits(d, d.y.values, cfg["cv"]["outer_folds"], cfg["seeds"]["outer_base"] + r)):
            tep = set(d.patient.values[te])
            mask = np.array([p in tep for p in pa])
            out.append((r, k, np.flatnonzero(~mask), np.flatnonzero(mask)))
    return out


def run_group(name, gname, X, y, pts, labels, splits, kind, tissue_free, cfg):
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
    auc, bacc, _, _ = summarise(y, P, TH)
    b = bootstrap(y, P, TH, cfg["bootstrap"]["B"], cfg["seeds"]["bootstrap"])
    row = {"dataset": name, "group": gname, "tissue_free": tissue_free, "model": kind, "n_patients": len(y), "n_pos": int(y.sum()),
           "auc": auc, "auc_lo": b["auc_ci"][0], "auc_hi": b["auc_ci"][1], "bal_acc": bacc}
    for neg in sorted(set(labels) - {"IM&CG"}):
        s = np.isin(labels, ["IM&CG", neg])
        row[f"auc_vs_{neg}"] = float(np.mean([roc_auc_score(y[s], q[s]) for q in P]))
    return row


def perm_one(i, X, yp, pts, cfg):
    warnings.filterwarnings("ignore")
    pa = np.array(pts)
    dd = pd.DataFrame({"patient": pa})
    R = cfg["cv"]["repeats"]
    P, TH = np.zeros((R, len(pa))), np.zeros((R, len(pa)))
    for r in range(R):
        for k, (tr, te) in enumerate(outer_splits(dd, yp, cfg["cv"]["outer_folds"], cfg["seeds"]["outer_base"] + r)):
            pt, ps, thr, *_ = outer_fold(X, yp, pa, tr, te, cfg["seeds"]["inner_base"] + 100 * r + k, cfg)
            pos = np.searchsorted(pa, pt)
            P[r, pos], TH[r, pos] = ps, thr
    a, b, _, _ = summarise(yp, P, TH)
    return {"perm": i, "auc": a, "bal_acc": b}


if __name__ == "__main__":
    cfg = load_config()
    cl = pd.read_csv(OUT / "capped_list.csv")
    F = pd.read_csv("D:/gim_outputs/goal1/artefact_image_features.csv")
    F["corner_nonblack"] = (F.corner_frac_le2 < 0.5).astype(float)
    top = F.hdr_hash.value_counts().index[:8]
    for i, h in enumerate(top):
        F[f"hdr_top{i}"] = (F.hdr_hash == h).astype(float)
    tasks = []
    meta = {}
    for name in ("A2-NBI", "S3"):
        d = cl[cl.dataset == name].sort_values(["patient", "file"]).reset_index(drop=True).merge(F, on="path", how="left")
        assert d.corner_val_r.notna().all()
        pts, M, L = patient_matrix(d)
        pinfo = d.groupby("patient").agg(y=("y", "first"), label=("label", "first")).reindex(pts)
        y, labels = pinfo.y.values.astype(int), pinfo.label.values
        splits = splits_for(d, pts, cfg)
        meta[name] = (pts, M, y)
        allv, alll = [], []
        for g, (cols, tf) in GROUPS.items():
            X = M[[f"{c}__{s}" for c in cols for s in ("mean", "std")]].values
            (allv if tf else alll).append(X)
            for kind in ("LR", "RF"):
                tasks.append((name, g, X, y, pts, labels, splits, kind, tf, cfg))
        for g, arr, tf in (("ALL_tissue_free", allv, True), ("ALL_low_level", alll, False)):
            X = np.hstack(arr)
            for kind in ("LR", "RF"):
                tasks.append((name, g, X, y, pts, labels, splits, kind, tf, cfg))
        # dark lumen: only patients with enough dark non-corner pixels
        ok = L.notna().all(axis=1).values
        idx = np.flatnonzero(ok)
        remap = -np.ones(len(pts), int)
        remap[idx] = np.arange(len(idx))
        sp = [(r, k, remap[tr][remap[tr] >= 0], remap[te][remap[te] >= 0]) for r, k, tr, te in splits]
        print(name, "patients with dark-lumen data:", int(ok.sum()), "of", len(pts), pd.Series(labels[ok]).value_counts().to_dict(), flush=True)
        for kind in ("LR", "RF"):
            tasks.append((name, "dark_lumen", L.values[ok], y[ok], list(np.array(pts)[ok]), labels[ok], sp, kind, True, cfg))
    rows = Parallel(n_jobs=10, verbose=2)(delayed(run_group)(*t) for t in tasks)
    R = pd.DataFrame(rows).round(4)
    R.to_csv(OUT / "artefact_test.csv", index=False)
    print(R.sort_values(["dataset", "auc"], ascending=[True, False])[["dataset", "group", "tissue_free", "model", "n_patients", "auc", "auc_lo", "auc_hi", "bal_acc", "auc_vs_EGC", "auc_vs_HGIN", "auc_vs_LGIN"]].to_string(index=False), flush=True)

    # permutation check of the tissue-free set, LR, whole protocol
    res = []
    for name in ("A2-NBI", "S3"):
        pts, M, y = meta[name]
        cols = [f"{c}__{s}" for g, (cc, tf) in GROUPS.items() if tf for c in cc for s in ("mean", "std")]
        X = M[cols].values
        rng = np.random.default_rng(cfg["seeds"]["permutation"])
        perms = [rng.permutation(y) for _ in range(200)]
        out = Parallel(n_jobs=10)(delayed(perm_one)(i, X, perms[i], pts, cfg) for i in range(200))
        o = pd.DataFrame(out).assign(dataset=name)
        res.append(o)
        obs = R[(R.dataset == name) & (R.group == "ALL_tissue_free") & (R.model == "LR")].auc.iloc[0]
        print(name, "ALL_tissue_free LR observed", obs, "null mean", round(o.auc.mean(), 3), "null 95th", round(np.percentile(o.auc, 95), 3),
              "p", round((1 + (o.auc >= obs).sum()) / 201, 4), flush=True)
    pd.concat(res).round(4).to_csv(OUT / "artefact_permutation.csv", index=False)
    print("done")
