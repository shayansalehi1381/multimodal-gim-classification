"""Shared helpers for Step Zero (Goal 3: Complete vs Incomplete, n=17).

Seeds: SEED=42 for every model / PCA / CV split (same as v1).
All preprocessing (scaler, PCA), hyper-parameter choice and decision threshold
are fit on the training patients of each fold only.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "step0"
LOG = RES / "LOG.md"
SEED = 42
CLINICAL_COLS = ["LBC", "MTB", "WOS", "TVF", "MLE", "TM", "Age"]
MEDIA_EXT_IMG = (".jpg", ".jpeg", ".png")
MEDIA_EXT_VID = (".mp4", ".mkv", ".avi", ".mov")


def out_dir(name):
    d = RES / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(title, text):
    """Append one sub-step paragraph to results/step0/LOG.md."""
    RES.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{text.strip()}\n")


# ── filename parsing ─────────────────────────────────────────────────────────
LOCATIONS = {"Antrum", "Body", "Prepyloric"}
MODALITIES = {"NBI", "M-NBI", "WLE"}
PROCESSORS = {"X1", "190"}


def parse_media_name(fname):
    """2610_7_IM_Prepyloric_X1_WLE_vid.mp4 -> dict of tokens."""
    stem, ext = os.path.splitext(fname)
    is_video = stem.endswith("_vid")
    if is_video:
        stem = stem[:-4]
    parts = stem.split("_")
    tok = parts[2:]
    return {
        "file": fname,
        "case": int(parts[0]),
        "index": int(parts[1]),
        "ext": ext.lower(),
        "is_video": is_video,
        "processor": next((t for t in tok if t in PROCESSORS), None),
        "modality": next((t for t in tok if t in MODALITIES), None),
        "location": next((t for t in tok if t in LOCATIONS), None),
        "tokens": "+".join(t for t in tok if t not in PROCESSORS | MODALITIES | LOCATIONS),
    }


def list_media(root=ROOT):
    files = sorted(f for f in os.listdir(root)
                   if f.lower().endswith(MEDIA_EXT_IMG + MEDIA_EXT_VID) and re.match(r"^\d{4}_\d+_", f))
    return pd.DataFrame([parse_media_name(f) for f in files])


# ── cohort / features ────────────────────────────────────────────────────────
def load_cohort(csv=PROC / "multimodal_biomedclip_dinov2_n17.csv"):
    """Return dataframe (17 rows, sorted by Case Code) plus feature blocks."""
    df = pd.read_csv(csv).sort_values("Case Code").reset_index(drop=True)
    y = df["Subtype"].map({"Complete": 0, "Incomplete": 1}).values.astype(int)
    blocks = {
        "clinical": df[CLINICAL_COLS].values.astype(float),
        "biomedclip": df[[c for c in df.columns if c.startswith("biomedclip_")]].values.astype(float),
        "dinov2": df[[c for c in df.columns if c.startswith("dinov2_")]].values.astype(float),
    }
    return df, y, blocks


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(y, score, pred):
    y, score, pred = map(np.asarray, (y, score, pred))
    return {
        "n_correct": int((y == pred).sum()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
    }


def youden_threshold_v1(y_true, y_prob):
    """Exact v1 rule: linspace(0.01, 0.99, 99), strict '>' -> first (lowest) max."""
    best_j, best_t = -1, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        p = (y_prob >= t).astype(int)
        j = (recall_score(y_true, p, pos_label=1, zero_division=0)
             + recall_score(y_true, p, pos_label=0, zero_division=0) - 1)
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def youden_threshold(y_true, score):
    """Step-0 rule: candidate cuts = midpoints between sorted unique scores;
    maximise Youden J; ties -> middle of the tied set. Returns (threshold, J)."""
    s = np.unique(score)
    cuts = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    pos, neg = y_true == 1, y_true == 0
    js = np.array([(score[pos] >= c).mean() + (score[neg] < c).mean() - 1 for c in cuts])
    best = np.flatnonzero(np.isclose(js, js.max()))
    return float(cuts[best[len(best) // 2]]), float(js.max())


# ── estimators ───────────────────────────────────────────────────────────────
class ScaledGammaSVC(BaseEstimator, ClassifierMixin):
    """RBF-SVC with gamma = gamma_mult * 'scale'. Score = decision_function
    (no Platt scaling: Platt on 15 samples is unstable)."""

    def __init__(self, C=1.0, gamma_mult=1.0):
        self.C, self.gamma_mult = C, gamma_mult

    def fit(self, X, y):
        g = self.gamma_mult / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
        self.model_ = SVC(kernel="rbf", C=self.C, gamma=g, random_state=SEED).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def score_samples(self, X):
        return self.model_.decision_function(X)


def lr_factory(C):
    return LogisticRegression(C=C, max_iter=5000, random_state=SEED)


def rf_factory(max_depth, max_features):
    return RandomForestClassifier(n_estimators=100, max_depth=max_depth, max_features=max_features,
                                  random_state=SEED, n_jobs=1)


def model_score(model, X):
    if hasattr(model, "score_samples") and isinstance(model, ScaledGammaSVC):
        return model.score_samples(X)
    return model.predict_proba(X)[:, 1]


MODEL_GRIDS = {
    "LR": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    "SVC": [dict(C=c, gamma_mult=g) for c in (0.1, 1.0, 10.0) for g in (0.3, 1.0, 3.0)],
    "RF": [dict(max_depth=d, max_features=m) for d in (2, None) for m in ("sqrt", None)],
}
PCA_K_GRID = (2, 4, 8)


def build_model(name, params):
    if name == "LR":
        return lr_factory(**params)
    if name == "SVC":
        return ScaledGammaSVC(**params)
    if name == "RF":
        return rf_factory(**params)
    raise ValueError(name)


# ── fold-local feature construction ──────────────────────────────────────────
def make_transform(Xc_tr, Xv_tr, kmax):
    """Fit scaler(s) + PCA(kmax, full SVD) on training rows only.
    Returns a function rows -> (clinical_scaled, visual_pca[:, :kmax])."""
    sc_c = StandardScaler().fit(Xc_tr) if Xc_tr is not None else None
    if Xv_tr is not None:
        sc_v = StandardScaler().fit(Xv_tr)
        pca = PCA(n_components=min(kmax, Xv_tr.shape[0] - 1), svd_solver="full",
                  random_state=SEED).fit(sc_v.transform(Xv_tr))
    def tf(Xc, Xv):
        c = sc_c.transform(Xc) if sc_c is not None else None
        v = pca.transform(sc_v.transform(Xv)) if Xv_tr is not None else None
        return c, v
    return tf


def assemble(c, v, k):
    parts = []
    if c is not None:
        parts.append(c)
    if v is not None:
        parts.append(v[:, :k])
    return np.hstack(parts)


def nested_fold(Xc, Xv, y, tr, te, model_name, k_grid=PCA_K_GRID):
    """One outer fold: inner LOO over the training patients for every
    (hyper-params, K) combo -> pooled inner AUC -> best combo (ties: first,
    grid is ordered simple->complex) -> Youden threshold on the pooled inner
    scores of the best combo -> refit on all training patients -> score test."""
    ks = k_grid if Xv is not None else (None,)
    kmax = max(k for k in ks) if Xv is not None else None
    combos = [(p, k) for k in ks for p in MODEL_GRIDS[model_name]]
    inner_scores = np.zeros((len(combos), len(tr)))
    for ii, i_te in enumerate(range(len(tr))):
        i_tr = np.delete(np.arange(len(tr)), i_te)
        rows_tr, rows_te = tr[i_tr], tr[[i_te]]
        tf = make_transform(Xc[rows_tr] if Xc is not None else None,
                            Xv[rows_tr] if Xv is not None else None, kmax)
        c_tr, v_tr = tf(Xc[rows_tr] if Xc is not None else None, Xv[rows_tr] if Xv is not None else None)
        c_te, v_te = tf(Xc[rows_te] if Xc is not None else None, Xv[rows_te] if Xv is not None else None)
        for ci, (p, k) in enumerate(combos):
            m = build_model(model_name, p).fit(assemble(c_tr, v_tr, k), y[rows_tr])
            inner_scores[ci, ii] = model_score(m, assemble(c_te, v_te, k))[0]
    y_tr = y[tr]
    aucs = np.array([roc_auc_score(y_tr, s) for s in inner_scores])
    best = int(np.argmax(aucs))
    p_best, k_best = combos[best]
    thr, j = youden_threshold(y_tr, inner_scores[best])
    tf = make_transform(Xc[tr] if Xc is not None else None, Xv[tr] if Xv is not None else None, kmax)
    c_tr, v_tr = tf(Xc[tr] if Xc is not None else None, Xv[tr] if Xv is not None else None)
    c_te, v_te = tf(Xc[te] if Xc is not None else None, Xv[te] if Xv is not None else None)
    m = build_model(model_name, p_best).fit(assemble(c_tr, v_tr, k_best), y_tr)
    s_all = model_score(m, assemble(c_te, v_te, k_best))
    s_te = float(s_all[0])
    # default cut (no threshold tuning): 0.5 for probabilities, 0 for SVC decision function
    default_cut = 0.0 if model_name == "SVC" else 0.5
    return {
        "score": s_te, "threshold": thr, "pred": int(s_te >= thr),
        "pred_default": int(s_te >= default_cut),
        "inner_auc": float(aucs[best]), "inner_J": j,
        "params": str(p_best), "pca_k": k_best,
        "scores_all_test": [float(s) for s in s_all],   # used by leave-pair-out (2 test patients)
    }


FEATURE_SETS = {
    "a_clinical": ("clinical", None),
    "b_biomedclip": (None, ["biomedclip"]),
    "c_dinov2": (None, ["dinov2"]),
    "d_both_visual": (None, ["biomedclip", "dinov2"]),
    "e_visual+clinical": ("clinical", ["biomedclip", "dinov2"]),
}


def feature_blocks(blocks, set_name):
    c, v = FEATURE_SETS[set_name]
    Xc = blocks["clinical"] if c else None
    Xv = np.hstack([blocks[b] for b in v]) if v else None
    return Xc, Xv


def run_nested_lopo(Xc, Xv, y, model_name, n_jobs=8, k_grid=PCA_K_GRID):
    """Outer leave-one-patient-out; everything else inside nested_fold()."""
    from joblib import Parallel, delayed
    n = len(y)
    folds = [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
    res = Parallel(n_jobs=n_jobs)(delayed(nested_fold)(Xc, Xv, y, tr, te, model_name, k_grid) for tr, te in folds)
    out = pd.DataFrame(res)
    out.insert(0, "row", np.arange(n))
    out.insert(1, "y", y)
    return out


# ── exact replica of the v1 DINOv2 evaluation protocol (fallback removed) ────
def v1_models():
    from sklearn.ensemble import GradientBoostingClassifier
    return {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        "SVC": SVC(kernel="rbf", probability=True, random_state=42),
        "GB": GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=2, random_state=42),
    }


def v1_run(Xc, Xv, y, model_name, n_pcs):
    """phase3_dinov2_multimodal_evaluation.py, one (model, PCs) cell: LOPOCV, in-fold scaler+PCA,
    threshold = v1 Youden rule on inner StratifiedKFold(4) cross_val_predict probabilities."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    n = len(y)
    prob, thr = np.zeros(n), np.zeros(n)
    for i in range(n):
        tr, te = np.delete(np.arange(n), i), np.array([i])
        sc_c, sc_v = StandardScaler().fit(Xc[tr]), StandardScaler().fit(Xv[tr])
        pca = PCA(n_components=n_pcs, random_state=42).fit(sc_v.transform(Xv[tr]))
        A = np.hstack((sc_c.transform(Xc[tr]), pca.transform(sc_v.transform(Xv[tr]))))
        B = np.hstack((sc_c.transform(Xc[te]), pca.transform(sc_v.transform(Xv[te]))))
        model = clone(v1_models()[model_name])
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        p_cv = cross_val_predict(model, A, y[tr], cv=cv, method="predict_proba")[:, 1]
        thr[i] = youden_threshold_v1(y[tr], p_cv)
        model.fit(A, y[tr])
        prob[i] = model.predict_proba(B)[0, 1]
    return prob, thr


V1_CONFIGS = [(m, k) for k in (5, 8) for m in ("RF", "SVC", "GB")]  # v1 loop order
