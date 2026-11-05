"""Sub-step 7d - stress-test the best FOV-cropped result (nested BiomedCLIP-only LR, cropped, same frames as v1).

It was chosen after looking at 27 cells (3 feature versions x 9 models in 7b), so it gets the same checks as v1:
  1. permutation test of the whole nested pipeline (N label permutations, RNG default_rng(2026)), AUC and accuracy
  2. Leave-Pair-Out AUC (72 pairs, nested tuning inside each 15-patient training set)
  3. confounder subsets, re-trained inside each subset: image-only n=12, X1-only n=15, drop 190+png n=14, jpg-only n=9
Outputs: results/step0/s7_fov/cropped_biomedclip_lr_checks.csv, cropped_biomedclip_lr_null.csv,
         results/step0/figures/permutation_null_cropped_biomedclip_lr.png
Usage: python s7d_cropped_biomedclip_checks.py [N=1000] [n_jobs=4]
"""
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score

from common import RES, ROOT, load_cohort, log, metrics, nested_fold, out_dir

warnings.filterwarnings("ignore")
OUT, FIG = out_dir("s7_fov"), out_dir("figures")


def cropped_biomedclip(cases):
    idx = pd.read_csv(ROOT / "step0" / "embeddings_cropped" / "index.csv")
    M = np.load(ROOT / "step0" / "embeddings_cropped" / "biomedclip_cropped.npy")
    rows = []
    for c in cases:
        v = M[((idx.case == c) & idx.used_in_v1).values].mean(0)
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


def lopo_scores(Xv, y):
    n = len(y)
    res = [nested_fold(None, Xv, y, np.delete(np.arange(n), i), np.array([i]), "LR") for i in range(n)]
    return np.array([r["score"] for r in res]), np.array([r["pred"] for r in res])


def perm_task(Xv, yp):
    import warnings as w
    w.filterwarnings("ignore")
    s, p = lopo_scores(Xv, yp)
    return roc_auc_score(yp, s), accuracy_score(yp, p)


def pair_task(Xv, y, i, j):
    import warnings as w
    w.filterwarnings("ignore")
    tr = np.setdiff1d(np.arange(len(y)), [i, j])
    s = nested_fold(None, Xv, y, tr, np.array([i, j]), "LR")["scores_all_test"]
    return 1.0 if s[1] > s[0] else (0.5 if s[1] == s[0] else 0.0)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    df, y, b = load_cohort()
    cases = df["Case Code"].values
    Xv = cropped_biomedclip(cases)
    s_obs, p_obs = lopo_scores(Xv, y)
    m_obs = metrics(y, s_obs, p_obs)

    rng = np.random.default_rng(2026)
    perms = [rng.permutation(y) for _ in range(N)]
    null = Parallel(n_jobs=n_jobs)(delayed(perm_task)(Xv, yp) for yp in perms)
    null = pd.DataFrame(null, columns=["auc", "accuracy"])
    null.to_csv(OUT / "cropped_biomedclip_lr_null.csv", index=False)
    p_auc = (1 + (null.auc >= m_obs["auc"] - 1e-12).sum()) / (N + 1)
    p_acc = (1 + (null.accuracy >= m_obs["accuracy"] - 1e-12).sum()) / (N + 1)

    pairs = [(i, j) for i in np.flatnonzero(y == 0) for j in np.flatnonzero(y == 1)]
    lpo = float(np.mean(Parallel(n_jobs=n_jobs)(delayed(pair_task)(Xv, y, i, j) for i, j in pairs)))

    fac = pd.read_csv(RES / "s6_confounders" / "patient_factors.csv").set_index("Case Code").loc[cases]
    subsets = {"all_n17": np.ones(len(y), bool), "image_only_n12": (fac.source == "stills only").values,
               "X1_only_n15": (fac.processor == "X1").values,
               "drop_190_and_png_n14": ~np.isin(cases, [2613, 2614, 2615]),
               "jpg_stills_noOSD_n9": (fac.file_type == "jpg stills").values}
    rows = []
    for name, msk in subsets.items():
        s, p = lopo_scores(Xv[msk], y[msk])
        rows.append({"check": f"subset {name} (re-trained)", "n": int(msk.sum()), **metrics(y[msk], s, p)})
    wrong = cases[p_obs != y].tolist()
    rows += [{"check": "permutation p (AUC)", "value": p_auc, "null_mean": null.auc.mean(),
              "null_95th": np.percentile(null.auc, 95)},
             {"check": "permutation p (accuracy)", "value": p_acc, "null_mean": null.accuracy.mean(),
              "null_95th": np.percentile(null.accuracy, 95)},
             {"check": "leave-pair-out AUC", "value": lpo},
             {"check": "misclassified patients (all_n17)", "note": str(wrong)}]
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cropped_biomedclip_lr_checks.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(null.auc, bins=25, color="#8c8c8c", edgecolor="white")
    ax[0].axvline(m_obs["auc"], color="#c0392b", lw=2, label=f"observed {m_obs['auc']:.3f}, p = {p_auc:.3f}")
    ax[0].set_xlabel("AUC")
    ax[1].hist(null.accuracy, bins=np.arange(0, 18) / 17 - 1 / 34, color="#8c8c8c", edgecolor="white")
    ax[1].axvline(m_obs["accuracy"], color="#c0392b", lw=2, label=f"observed {m_obs['accuracy']:.3f}, p = {p_acc:.3f}")
    ax[1].set_xlabel("accuracy")
    for a in ax:
        a.legend(fontsize=9)
    ax[0].set_ylabel(f"count (N = {N})")
    fig.suptitle("Permutation null - nested BiomedCLIP-only LR on FOV-cropped embeddings (fixed config, chosen among 27)")
    fig.tight_layout()
    fig.savefig(FIG / "permutation_null_cropped_biomedclip_lr.png", dpi=200)

    sub = R[R.check.str.startswith("subset")]
    log("Sub-step 7d - Stress test of the best cropped result (nested BiomedCLIP-LR, FOV-cropped)", f"""
Observed (n=17): {m_obs['n_correct']}/17, balanced acc {m_obs['balanced_accuracy']:.3f}, AUC {m_obs['auc']:.3f}; misclassified {wrong}.
Fixed-config whole-pipeline permutation test (N={N}): p(AUC) = {p_auc:.4f}, p(accuracy) = {p_acc:.4f}
(null mean AUC {null.auc.mean():.3f}, 95th pct {np.percentile(null.auc, 95):.3f}).
This config was picked after looking at 27 cells, so these p-values are not selection-corrected.
A Bonferroni-style bound over 27 cells is min(1, 27 x p).
Leave-Pair-Out AUC: {lpo:.3f}.
Re-trained in subsets: """ + "; ".join(f"{r.check.split()[1]}: {int(r.n_correct)}/{int(r.n)} (AUC {r.auc:.2f})" for r in sub.itertuples()) + """.
Files: results/step0/s7_fov/cropped_biomedclip_lr_checks.csv, cropped_biomedclip_lr_null.csv,
results/step0/figures/permutation_null_cropped_biomedclip_lr.png
""")
    print(R.round(3).to_string(index=False))
