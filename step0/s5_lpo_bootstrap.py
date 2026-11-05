"""Sub-step 5 - Leave-Pair-Out AUC and confidence intervals.

LPO-AUC: for every (Complete, Incomplete) pair (8 x 9 = 72 pairs) the model is trained on the other 15 patients
(all preprocessing / tuning inside, nested for the step-0 models) and both held-out patients are scored;
LPO-AUC = fraction of pairs ranked correctly (ties = 0.5). It avoids pooling scores from 17 different models
and the LOO class-balance bias.
CIs from the LOPOCV out-of-fold predictions: patient-level stratified bootstrap (2000 resamples,
RNG default_rng(12345)) for AUC, accuracy, balanced accuracy; Clopper-Pearson exact 95% CI for accuracy.
(Bootstrap of fixed OOF predictions ignores training variability -> CIs are, if anything, too narrow.)
Models: v1 champion (exact v1 protocol, SVC 8 PCs) and the step-0 nested LR and SVC for all 5 feature sets.
Nested RF is not run with LPO because of compute cost (72 x nested RF); its LOPOCV CIs are reported.
Outputs: results/step0/s5_lpo_ci/{lpo_auc.csv, lpo_pairs.csv, ci_table.csv}
"""
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import (FEATURE_SETS, RES, feature_blocks, load_cohort, log, nested_fold, out_dir, v1_models)

warnings.filterwarnings("ignore")
OUT = out_dir("s5_lpo_ci")


def v1_fit_score(Xc, Xv, y, tr, te, model_name="SVC", n_pcs=8):
    sc_c, sc_v = StandardScaler().fit(Xc[tr]), StandardScaler().fit(Xv[tr])
    pca = PCA(n_components=n_pcs, random_state=42).fit(sc_v.transform(Xv[tr]))
    f = lambda r: np.hstack((sc_c.transform(Xc[r]), pca.transform(sc_v.transform(Xv[r]))))
    m = clone(v1_models()[model_name]).fit(f(tr), y[tr])
    return m.predict_proba(f(te))[:, 1]


def pair_task(kind, Xc, Xv, y, i, j, model):
    import warnings as w
    w.filterwarnings("ignore")
    n = len(y)
    tr = np.setdiff1d(np.arange(n), [i, j])
    te = np.array([i, j])
    if kind == "v1":
        s = v1_fit_score(Xc, Xv, y, tr, te)
    else:
        s = np.array(nested_fold(Xc, Xv, y, tr, te, model)["scores_all_test"])
    return float(s[0]), float(s[1])        # (Complete patient i, Incomplete patient j)


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def boot_ci(y, score, pred, B=2000, seed=12345):
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    aucs, accs, baccs = [], [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
        aucs.append(roc_auc_score(y[idx], score[idx]))
        accs.append(accuracy_score(y[idx], pred[idx]))
        baccs.append(balanced_accuracy_score(y[idx], pred[idx]))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(aucs), q(accs), q(baccs)


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    df, y, b = load_cohort()
    cases = df["Case Code"].values
    pairs = [(i, j) for i in np.flatnonzero(y == 0) for j in np.flatnonzero(y == 1)]

    jobs = [("v1_champion_SVC8", "v1", b["clinical"], np.hstack([b["biomedclip"], b["dinov2"]]), None)]
    for fs in FEATURE_SETS:
        Xc, Xv = feature_blocks(b, fs)
        for model in ("LR", "SVC"):
            jobs.append((f"{fs}/{model}", "nested", Xc, Xv, model))

    lpo_rows, pair_rows = [], []
    for name, kind, Xc, Xv, model in jobs:
        res = Parallel(n_jobs=n_jobs)(delayed(pair_task)(kind, Xc, Xv, y, i, j, model) for i, j in pairs)
        wins = [1.0 if sj > si else (0.5 if sj == si else 0.0) for si, sj in res]
        lpo_rows.append({"model": name, "lpo_auc": np.mean(wins), "n_pairs": len(pairs)})
        for (i, j), (si, sj), wv in zip(pairs, res, wins):
            pair_rows.append({"model": name, "complete_case": cases[i], "incomplete_case": cases[j],
                              "score_complete": si, "score_incomplete": sj, "correct_order": wv})
        print(f"{name:28s} LPO-AUC = {np.mean(wins):.3f}", flush=True)
    LPO = pd.DataFrame(lpo_rows)
    LPO.to_csv(OUT / "lpo_auc.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "lpo_pairs.csv", index=False)

    # ── CIs from LOPOCV out-of-fold predictions ─────────────────────────────
    ci_rows = []
    v1p = pd.read_csv(RES / "s3_v1_configs" / "per_patient_probs.csv")
    v1t = pd.read_csv(RES / "s3_v1_configs" / "fold_thresholds.csv")
    sv = v1p[(v1p.model == "SVC") & (v1p.PCs == 8)].sort_values("Case Code")
    th = v1t[(v1t.model == "SVC") & (v1t.PCs == 8)].sort_values("Case Code").youden_threshold.values
    sources = [("v1_champion_SVC8", sv.prob_incomplete.values, (sv.prob_incomplete.values >= th).astype(int))]
    s2 = pd.read_csv(RES / "s2_baselines" / "per_patient_scores.csv")
    for (fs, model), g in s2.groupby(["feature_set", "model"], sort=False):
        g = g.sort_values("Case Code")
        sources.append((f"{fs}/{model}", g.score.values, g.pred.values))
    for name, score, pred in sources:
        (alo, ahi), (clo, chi), (blo, bhi) = boot_ci(y, score, pred)
        k = int((pred == y).sum())
        cp_lo, cp_hi = clopper_pearson(k, len(y))
        ci_rows.append({"model": name, "n_correct": k, "accuracy": k / len(y), "acc_clopper_pearson_lo": cp_lo,
                        "acc_clopper_pearson_hi": cp_hi, "acc_boot_lo": clo, "acc_boot_hi": chi,
                        "balanced_accuracy": balanced_accuracy_score(y, pred), "bacc_boot_lo": blo, "bacc_boot_hi": bhi,
                        "auc_lopocv": roc_auc_score(y, score), "auc_boot_lo": alo, "auc_boot_hi": ahi})
    CI = pd.DataFrame(ci_rows).merge(LPO[["model", "lpo_auc"]], on="model", how="left")
    CI.to_csv(OUT / "ci_table.csv", index=False)

    champ = CI.iloc[0]
    lines = "\n".join(f"  - {r.model}: acc {r.accuracy:.3f} CP [{r.acc_clopper_pearson_lo:.3f}, {r.acc_clopper_pearson_hi:.3f}], "
                      f"AUC {r.auc_lopocv:.3f} boot [{r.auc_boot_lo:.3f}, {r.auc_boot_hi:.3f}], LPO-AUC "
                      f"{'n/a' if pd.isna(r.lpo_auc) else f'{r.lpo_auc:.3f}'}" for r in CI.itertuples())
    log("Sub-step 5 - Leave-Pair-Out AUC and confidence intervals", f"""
LPO-AUC over all 72 (Complete, Incomplete) pairs, each trained on the remaining 15 patients (nested tuning inside).
CIs: Clopper-Pearson exact for accuracy; patient-level stratified bootstrap (2000 resamples) for AUC and balanced accuracy.
{lines}
v1 champion: LPO-AUC {champ.lpo_auc:.3f} vs pooled LOPOCV AUC {champ.auc_lopocv:.3f}. Clopper-Pearson accuracy CI
[{champ.acc_clopper_pearson_lo:.3f}, {champ.acc_clopper_pearson_hi:.3f}] (v1 reported bootstrap [0.588, 1.000]). Even before correcting for
selection, the AUC CI includes 0.5.
Files: results/step0/s5_lpo_ci/{{lpo_auc, lpo_pairs, ci_table}}.csv
""")
    print(CI.round(3).to_string(index=False))
