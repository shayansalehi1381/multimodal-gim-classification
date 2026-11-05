"""Shared Step B protocol. Every parameter comes from results/goal1/config_B.json (checked against its SHA-256)."""
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "goal1"
EMB = "D:/gim_outputs/goal1/embeddings/"


class Unexpected(RuntimeError):
    """Stop-and-report condition from the pre-specified rules."""


def load_config():
    p = OUT / "config_B.json"
    if not p.exists():
        return None
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rec = (OUT / "config_B.sha256").read_text().strip()
    if sha != rec:
        raise Unexpected(f"config_B.json changed after freezing: {sha} != {rec}")
    return json.loads(raw)


def load_dataset(name, emb):
    """Rows of capped_list for one dataset, with the matching embedding matrix."""
    cl = pd.read_csv(OUT / "capped_list.csv")
    d = cl[cl.dataset == name].sort_values(["patient", "file"]).reset_index(drop=True)
    idx = pd.read_csv(f"{EMB}{emb}_index.csv").set_index("path").row
    X = np.load(f"{EMB}{emb}.npy", mmap_mode="r")[idx.loc[d.path].values].astype(np.float64)
    if not np.isfinite(X).all():
        raise Unexpected(f"non-finite embeddings for {name}/{emb}")
    return d, X


def patient_table(d):
    p = d.groupby("patient").agg(y=("y", "first"), label=("label", "first"), year=("year", "first")).reset_index()
    return p.sort_values("patient").reset_index(drop=True)


def outer_splits(d, y_img, n_splits, seed):
    """Image rows, grouped by patient, stratified on the image label."""
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(d.index.values, y_img, d.patient.values))


def _pat_mean(score, pat_codes, n):
    return np.bincount(pat_codes, weights=score, minlength=n) / np.maximum(np.bincount(pat_codes, minlength=n), 1)


def youden(y, s):
    u = np.unique(s)
    if len(u) < 2:
        return float(u[0])
    cuts = (u[:-1] + u[1:]) / 2
    pos, neg = y == 1, y == 0
    j = np.array([(s[pos] >= c).mean() + (s[neg] < c).mean() - 1 for c in cuts])
    best = np.flatnonzero(np.isclose(j, j.max()))
    return float(cuts[best[len(best) // 2]])


def _make_lr(cfg, C):
    l = cfg["logistic_regression"]
    return LogisticRegression(C=C, class_weight="balanced", solver=l["solver"], max_iter=l["max_iter"], tol=l["tol"],
                              warm_start=True, random_state=cfg["seeds"]["model"])


def fit_path(Xtr, ytr, Xte, cgrid, cfg):
    """Standardise on the training rows, fit LR along the C grid (ascending, warm start). Returns scores per C."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out, m = np.zeros((len(cgrid), len(B))), None
    conv = 0
    for i, C in enumerate(cgrid):
        if m is None:
            m = _make_lr(cfg, C)
        else:
            m.set_params(C=C)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(A, ytr)
        conv += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        out[i] = m.decision_function(B)
    return out, conv


def outer_fold(X, y_img, patients, tr, te, seed, cfg):
    """One outer fold. Inner grouped CV picks C on patient-level AUC; threshold = Youden J on inner patient-level
    out-of-fold probabilities for that C; refit on all training images; patient score = mean image probability."""
    cgrid = np.logspace(cfg["logistic_regression"]["C_log10_start"], cfg["logistic_regression"]["C_log10_stop"],
                        cfg["logistic_regression"]["C_n"])
    k_in = cfg["cv"]["inner_folds"]
    pat_tr, codes = np.unique(patients[tr], return_inverse=True)
    y_pat = np.zeros(len(pat_tr))
    y_pat[codes] = y_img[tr]
    oof = np.zeros((len(cgrid), len(tr)))
    conv = 0
    for itr, ite in StratifiedGroupKFold(k_in, shuffle=True, random_state=seed).split(tr, y_img[tr], patients[tr]):
        if len(np.unique(y_img[tr][ite])) < 2 or len(np.unique(y_img[tr][itr])) < 2:
            raise Unexpected("class empty in an inner fold")
        scores, c = fit_path(X[tr[itr]], y_img[tr[itr]], X[tr[ite]], cgrid, cfg)
        oof[:, ite] = 1 / (1 + np.exp(-scores))
        conv += c
    pmean = np.array([_pat_mean(o, codes, len(pat_tr)) for o in oof])
    aucs = np.array([roc_auc_score(y_pat, p) for p in pmean])
    b = int(np.argmax(aucs))             # ties: smallest C (strongest regularisation)
    thr = youden(y_pat, pmean[b])
    sc = StandardScaler().fit(X[tr])
    m = _make_lr(cfg, cgrid[b])
    m.warm_start = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(sc.transform(X[tr]), y_img[tr])
    conv += sum(issubclass(x.category, ConvergenceWarning) for x in w)
    p_img = m.predict_proba(sc.transform(X[te]))[:, 1]
    if not np.isfinite(p_img).all():
        raise Unexpected("non-finite probabilities")
    pat_te, codes_te = np.unique(patients[te], return_inverse=True)
    return pat_te, _pat_mean(p_img, codes_te, len(pat_te)), thr, float(cgrid[b]), float(aucs[b]), conv


def run_pipeline(d, X, y_pat_map, cfg):
    """Whole repeated CV for one labelling (true or permuted). y_pat_map: patient -> label.
    Returns dict of arrays per repeat (patients in sorted order)."""
    patients = d.patient.values
    y_img = pd.Series(patients).map(y_pat_map).values.astype(int)
    pts = np.array(sorted(set(patients)))
    y = pd.Series(pts).map(y_pat_map).values.astype(int)
    R = cfg["cv"]["repeats"]
    P, TH, FOLD = np.zeros((R, len(pts))), np.zeros((R, len(pts))), np.zeros((R, len(pts)), int)
    info = []
    pos = {p: i for i, p in enumerate(pts)}
    for r in range(R):
        seed = cfg["seeds"]["outer_base"] + r
        for k, (tr, te) in enumerate(outer_splits(d, y_img, cfg["cv"]["outer_folds"], seed)):
            if set(patients[tr]) & set(patients[te]):
                raise Unexpected("patient in both train and test")
            if len(np.unique(y_img[te])) < 2 or len(np.unique(y_img[tr])) < 2:
                raise Unexpected("class empty in an outer fold")
            pt, ps, thr, C, a, conv = outer_fold(X, y_img, patients, tr, te, cfg["seeds"]["inner_base"] + 100 * r + k, cfg)
            for q, s in zip(pt, ps):
                P[r, pos[q]], TH[r, pos[q]], FOLD[r, pos[q]] = s, thr, k
            info.append({"repeat": r, "fold": k, "C": C, "inner_auc": a, "threshold": thr, "convergence_warnings": conv})
    return {"patients": pts, "y": y, "P": P, "TH": TH, "FOLD": FOLD, "info": pd.DataFrame(info)}


def summarise(y, P, TH):
    """Mean over repeats of pooled patient-level AUC and balanced accuracy (fold thresholds from inner CV)."""
    aucs = [roc_auc_score(y, p) for p in P]
    bacc = [balanced_accuracy_score(y, p >= t) for p, t in zip(P, TH)]
    return float(np.mean(aucs)), float(np.mean(bacc)), aucs, bacc


def _auc_fast(y, s):
    n1 = y.sum()
    n0 = len(y) - n1
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(y, P, TH, B, seed, P2=None, TH2=None):
    """Stratified patient bootstrap; each draw is evaluated on all repeats and averaged. Optional paired difference."""
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    A, Ba, D, Db = [], [], [], []
    for _ in range(B):
        i = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
        yi = y[i]
        a = np.mean([_auc_fast(yi, p[i]) for p in P])
        b = np.mean([((p[i] >= t[i])[yi == 1].mean() + (p[i] < t[i])[yi == 0].mean()) / 2 for p, t in zip(P, TH)])
        A.append(a); Ba.append(b)
        if P2 is not None:
            a2 = np.mean([_auc_fast(yi, p[i]) for p in P2])
            b2 = np.mean([((p[i] >= t[i])[yi == 1].mean() + (p[i] < t[i])[yi == 0].mean()) / 2 for p, t in zip(P2, TH2)])
            D.append(a - a2); Db.append(b - b2)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = {"auc_ci": q(A), "bacc_ci": q(Ba)}
    if P2 is not None:
        out.update(diff_auc_ci=q(D), diff_bacc_ci=q(Db),
                   p_two_sided_auc=float(min(1.0, 2 * min(np.mean(np.array(D) <= 0), np.mean(np.array(D) >= 0)))))
    return out
