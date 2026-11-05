"""Sub-step 4 - permutation test of the WHOLE v1 pipeline.

For each of N label permutations (patient labels shuffled, class counts kept) the full v1 protocol is rerun:
in-fold scaler + PCA, inner-CV Youden threshold, for all 6 (model, PCs) cells -> 12 configs.
Recorded per permutation: accuracy and AUC of every config.
Null distributions:
  * champion fixed  : accuracy / AUC of (SVC, 8 PCs, Youden) - answers "is this one pre-specified config better than chance?"
  * selection-aware : max accuracy over the 12 configs, max AUC over the 6 cells - this mimics what v1 did
                      (pick the best of 12 after looking), so this is the honest test of the reported 82.4%.
p = (1 + #{null >= observed}) / (N + 1).   RNG: np.random.default_rng(2026).
Outputs: results/step0/s4_permutation/{null_distribution.csv, permutation_summary.csv}, figures/permutation_null.png
Usage: python s4_permutation.py [N=1000] [n_jobs=10]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import V1_CONFIGS, load_cohort, log, out_dir, v1_run

warnings.filterwarnings("ignore")
OUT = out_dir("s4_permutation")


def one_perm(Xc, Xv, y_perm):
    import warnings as w
    w.filterwarnings("ignore")
    from sklearn.metrics import accuracy_score, roc_auc_score
    rec = {}
    for model, k in V1_CONFIGS:
        prob, thr = v1_run(Xc, Xv, y_perm, model, k)
        rec[f"{model}{k}_auc"] = roc_auc_score(y_perm, prob)
        rec[f"{model}{k}_acc05"] = accuracy_score(y_perm, (prob >= 0.5).astype(int))
        rec[f"{model}{k}_accY"] = accuracy_score(y_perm, (prob >= thr).astype(int))
    return rec


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    df, y, b = load_cohort()
    Xc, Xv = b["clinical"], np.hstack([b["biomedclip"], b["dinov2"]])
    rng = np.random.default_rng(2026)
    perms = [rng.permutation(y) for _ in range(N)]
    t0 = time.time()
    observed = one_perm(Xc, Xv, y)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_perm)(Xc, Xv, yp) for yp in perms)
    null = pd.DataFrame(res)
    null.insert(0, "perm", np.arange(N))
    acc_cols = [c for c in null.columns if c.endswith(("_acc05", "_accY"))]
    auc_cols = [c for c in null.columns if c.endswith("_auc")]
    null["max_acc_12"] = null[acc_cols].max(axis=1)
    null["max_auc_6"] = null[auc_cols].max(axis=1)
    null.to_csv(OUT / "null_distribution.csv", index=False)
    obs = pd.Series(observed)
    obs_max_acc, obs_max_auc = obs[acc_cols].max(), obs[auc_cols].max()

    def p(nullv, o):
        return (1 + np.sum(nullv >= o - 1e-12)) / (len(nullv) + 1)

    summ = pd.DataFrame([
        ("champion SVC8-Youden accuracy (fixed config)", obs["SVC8_accY"], null["SVC8_accY"].mean(),
         np.percentile(null["SVC8_accY"], 95), p(null["SVC8_accY"], obs["SVC8_accY"])),
        ("champion SVC8 AUC (fixed config)", obs["SVC8_auc"], null["SVC8_auc"].mean(),
         np.percentile(null["SVC8_auc"], 95), p(null["SVC8_auc"], obs["SVC8_auc"])),
        ("max accuracy over 12 configs (selection-aware)", obs_max_acc, null["max_acc_12"].mean(),
         np.percentile(null["max_acc_12"], 95), p(null["max_acc_12"], obs_max_acc)),
        ("max AUC over 6 cells (selection-aware)", obs_max_auc, null["max_auc_6"].mean(),
         np.percentile(null["max_auc_6"], 95), p(null["max_auc_6"], obs_max_auc)),
    ], columns=["statistic", "observed", "null_mean", "null_95th_pct", "p_value"])
    summ.to_csv(OUT / "permutation_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, o, title in ((ax[0], "SVC8_accY", obs["SVC8_accY"], "Champion config accuracy (fixed)"),
                             (ax[1], "max_acc_12", obs_max_acc, "Max accuracy over 12 configs"),
                             (ax[2], "SVC8_auc", obs["SVC8_auc"], "Champion config AUC (fixed)")):
        vals = null[col].values
        bins = np.arange(0, 18) / 17 - 1 / 34 if "acc" in col else 20
        a.hist(vals, bins=bins, color="#8c8c8c", edgecolor="white")
        a.axvline(o, color="#c0392b", lw=2, label=f"observed {o:.3f}\np = {p(vals, o):.3f}")
        a.set_title(title)
        a.set_xlabel("accuracy" if "acc" in col else "AUC")
        a.legend(loc="upper left", fontsize=9)
    ax[0].set_ylabel(f"count (N = {N} permutations)")
    fig.suptitle("Whole-pipeline permutation test (labels shuffled across patients; PCA, threshold, selection re-run)")
    fig.tight_layout()
    figdir = out_dir("figures")
    fig.savefig(figdir / "permutation_null.png", dpi=200)

    lines = "\n".join(f"  - {r.statistic}: observed {r.observed:.3f}, null mean {r.null_mean:.3f}, "
                      f"null 95th pct {r.null_95th_pct:.3f}, p = {r.p_value:.4f}" for r in summ.itertuples())
    log("Sub-step 4 - Whole-pipeline permutation test", f"""
N = {N} label permutations (RNG default_rng(2026)). Each one reruns the full v1 protocol for all 12 configs (in-fold scaler/PCA,
inner-CV Youden threshold). Runtime {time.time()-t0:.0f}s on {n_jobs} processes.
{lines}
The fixed-config p-values answer whether that one config (if it had been pre-specified) beats chance. The
selection-aware p-values are the honest test of the reported result, because v1 picked the best of 12 after looking.
Files: results/step0/s4_permutation/{{null_distribution, permutation_summary}}.csv, results/step0/figures/permutation_null.png
""")
    print(summ.round(4).to_string(index=False))
