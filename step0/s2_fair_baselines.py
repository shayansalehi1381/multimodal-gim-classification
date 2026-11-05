"""Sub-step 2 - fair baselines with fully nested LOPOCV.

Feature sets: (a) clinical (6 endoscopic signs from metadata + Age), (b) BiomedCLIP, (c) DINOv2,
(d) both visual, (e) both visual + clinical. Models: L2-LR, RBF-SVC, RF.
Per outer fold (16 training patients): inner leave-one-out over the 16 for every (hyper-params, PCA_K) combo,
with scaler + PCA refit on each inner training set; best combo = highest pooled inner AUC (ties -> first,
grid ordered simple->complex); decision threshold = Youden J on the pooled inner scores of the best combo;
refit on the 16 and score the held-out patient. Grids: LR C{0.01,0.1,1,10}; SVC C{0.1,1,10} x gamma
{0.3,1,3}x'scale'; RF(100 trees) max_depth{2,None} x max_features{sqrt,all}; PCA_K{2,4,8} for visual sets.
SVC score = decision_function (threshold in score units, default cut 0); LR/RF score = P(Incomplete).
Outputs: results/step0/s2_baselines/{summary.csv, per_patient_scores.csv}
"""
import time
import warnings

import pandas as pd

from common import FEATURE_SETS, feature_blocks, load_cohort, log, metrics, out_dir, run_nested_lopo

warnings.filterwarnings("ignore")
OUT = out_dir("s2_baselines")

if __name__ == "__main__":
    df, y, blocks = load_cohort()
    summ, per = [], []
    for fs in FEATURE_SETS:
        Xc, Xv = feature_blocks(blocks, fs)
        for model in ("LR", "SVC", "RF"):
            t0 = time.time()
            r = run_nested_lopo(Xc, Xv, y, model)
            m = metrics(y, r.score, r.pred)
            m0 = metrics(y, r.score, r.pred_default)
            summ.append({"feature_set": fs, "model": model, **m,
                         "acc_default_cut": m0["accuracy"], "bacc_default_cut": m0["balanced_accuracy"],
                         "chosen_K": "/".join(map(str, sorted(r.pca_k.dropna().astype(int).unique()))) if Xv is not None else "",
                         "median_inner_auc": r.inner_auc.median(), "seconds": round(time.time() - t0, 1)})
            r.insert(0, "Case Code", df["Case Code"].values)
            r.insert(0, "model", model)
            r.insert(0, "feature_set", fs)
            per.append(r)
            print(f"{fs:18s} {model:3s} acc={m['accuracy']:.3f} bacc={m['balanced_accuracy']:.3f} "
                  f"auc={m['auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    S = pd.DataFrame(summ)
    P = pd.concat(per, ignore_index=True)
    S.to_csv(OUT / "summary.csv", index=False)
    P.to_csv(OUT / "per_patient_scores.csv", index=False)
    best = S.sort_values("auc", ascending=False).head(3)
    lines = "\n".join(f"  - {r.feature_set} / {r.model}: acc {r.accuracy:.3f} ({r.n_correct}/17), bacc "
                      f"{r.balanced_accuracy:.3f}, F1 {r.f1:.3f}, AUC {r.auc:.3f}" for r in S.itertuples())
    log("Sub-step 2 - Fair baselines (fully nested LOPOCV)", f"""
Protocol: see the docstring of step0/s2_fair_baselines.py. Inner selection: pooled inner-LOO AUC. Threshold:
Youden J on pooled inner scores. Nothing is fit on the held-out patient. 15 combinations (5 feature sets x 3 models),
all reported. No single "winner" is selected, because that would be another multiple comparison.
{lines}
Highest AUC: {', '.join(f'{r.feature_set}/{r.model} ({r.auc:.3f})' for r in best.itertuples())}.
Files: results/step0/s2_baselines/summary.csv (also acc/bacc at the default cut), per_patient_scores.csv
(score, threshold, prediction, chosen params and K for every patient x model).
""")
