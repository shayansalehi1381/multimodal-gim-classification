"""Goal 1, Step B - EXPLORATORY diagnostics after the stop condition (AUC 0.99+). Not part of the frozen protocol; the frozen
config, capped list and embeddings are not changed. Nothing here is a performance result.

1. Code check: 10 label permutations through the whole frozen pipeline (expected AUC near 0.5).
2. Near-duplicates: smallest dHash distance and highest embedding cosine to an image of ANOTHER patient, by class pair.
3. What carries the signal: one random capped image per patient (seed 42), BiomedCLIP on altered copies, patient-level
   LR (C=1, standardised in-fold, 5-fold x 5 repeats):
     original / center_only (inner 50% square, no borders) / border_only (centre blacked out) /
     blur_16px (colour layout, no texture) / patch_shuffle_16 (texture and colour, no structure) /
     gray (no colour) / lumen_free not attempted
Outputs: results/goal1/diagnostics_B_{permutation,neighbours,alterations}.csv
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import warnings

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.functional import normalize, pil_to_tensor

from b_common import OUT, load_config, load_dataset, patient_table, run_pipeline, summarise

warnings.filterwarnings("ignore")


def perm_one(i, d, X, pts, yp, cfg):
    warnings.filterwarnings("ignore")
    r = run_pipeline(d, X, dict(zip(pts, yp)), cfg)
    a, b, _, _ = summarise(r["y"], r["P"], r["TH"])
    return {"perm": i, "auc": a, "bal_acc": b}


def alt(im, kind, rng):
    a = np.array(im.resize((224, 224), Image.BICUBIC))
    if kind == "center_only":
        return Image.fromarray(a[56:168, 56:168]).resize((224, 224), Image.BICUBIC)
    if kind == "border_only":
        a = a.copy(); a[28:196, 28:196] = 0
        return Image.fromarray(a)
    if kind == "blur_16px":
        return Image.fromarray(a).resize((14, 14), Image.BOX).resize((224, 224), Image.BICUBIC)
    if kind == "patch_shuffle_16":
        p = a.reshape(14, 16, 14, 16, 3).transpose(0, 2, 1, 3, 4).reshape(196, 16, 16, 3)
        p = p[rng.permutation(196)].reshape(14, 14, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(224, 224, 3)
        return Image.fromarray(p)
    if kind == "gray":
        return Image.fromarray(a).convert("L").convert("RGB")
    return Image.fromarray(a)


def cv_auc(X, y):
    P = np.zeros(len(y))
    for r in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42 + r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="newton-cholesky"))
            P[te] += m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / 5
    return roc_auc_score(y, P), P


if __name__ == "__main__":
    cfg = load_config()
    d, X = load_dataset("A2-NBI", "biomedclip")
    p = patient_table(d)
    rng = np.random.default_rng(cfg["seeds"]["permutation"])
    perms = [rng.permutation(p.y.values) for _ in range(10)]
    r = Parallel(n_jobs=10)(delayed(perm_one)(i, d, X, p.patient.values, perms[i], cfg) for i in range(10))
    pd.DataFrame(r).round(4).to_csv(OUT / "diagnostics_B_permutation.csv", index=False)
    print(pd.DataFrame(r).round(3).describe().loc[["mean", "min", "max"]], flush=True)

    # neighbours across patients
    stats = pd.read_csv("D:/gim_outputs/goal1/image_stats.csv", dtype={"dhash": str}).set_index("path")
    H = np.array([[c == "1" for c in stats.dhash.loc[q]] for q in d.path], dtype=np.uint8)
    pats, labs = d.patient.values, d.label.values
    S = X @ X.T
    rows = []
    for i in range(len(d)):
        other = pats != pats[i]
        ham = (H[other] != H[i]).sum(1)
        j = np.argmax(S[i][other])
        rows.append({"patient": pats[i], "label": labs[i], "min_hamming_other_patient": ham.min(), "max_cos_other_patient": S[i][other][j],
                     "nn_label": labs[other][j], "nn_patient": pats[other][j]})
    N = pd.DataFrame(rows)
    N.round(4).to_csv(OUT / "diagnostics_B_neighbours.csv", index=False)
    print(N.groupby("label").agg(min_ham=("min_hamming_other_patient", "min"), median_ham=("min_hamming_other_patient", "median"),
                                 max_cos=("max_cos_other_patient", "max"), median_cos=("max_cos_other_patient", "median")).round(3))
    print("nearest other-patient image, same class share:\n", (N.label == N.nn_label).groupby(N.label).mean().round(3), flush=True)

    # alterations
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    m = m.eval()
    tn = pre.transforms[-1]
    one = d.groupby("patient").sample(1, random_state=42).sort_values("patient").reset_index(drop=True)
    y = one.y.values
    res = []
    for kind in ("original", "center_only", "border_only", "blur_16px", "patch_shuffle_16", "gray"):
        rg = np.random.default_rng(42)
        E = []
        with torch.inference_mode():
            for i in range(0, len(one), 8):
                xs = [normalize(pil_to_tensor(alt(Image.open(q).convert("RGB"), kind, rg).resize((224, 224), Image.BICUBIC)).float() / 255.0, list(tn.mean), list(tn.std))
                      for q in one.path.iloc[i:i + 8]]
                E.append(torch.nn.functional.normalize(m.encode_image(torch.stack(xs)), dim=-1).numpy())
        E = np.vstack(E).astype(np.float64)
        a, P = cv_auc(E, y)
        row = {"alteration": kind, "auc_all": a}
        for neg in ("EGC", "HGIN", "LGIN"):
            s = one.label.isin(["IM&CG", neg]).values
            row[f"auc_vs_{neg}"] = roc_auc_score(y[s], P[s])
        res.append(row)
        print(row, flush=True)
    pd.DataFrame(res).round(4).to_csv(OUT / "diagnostics_B_alterations.csv", index=False)
    print("done")
