"""Goal 1, Step B - pre-specified image-embedding models, with no tuning on test folds and no best-of-N.

Runs: A2-NBI/BiomedCLIP (primary), A2-NBI/DINOv2 (secondary), S1, S2, S3 with BiomedCLIP (sensitivity).
Paired comparison: bootstrap of the AUC / balanced-accuracy difference against the best shortcut on the same folds.
Outputs: results/goal1/{results_B.csv, per_patient_B.csv, fold_counts_B.csv, fold_details_B.csv, subgroup_B.csv}
Exit code 3 if a stop condition fires.
"""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from b_common import OUT, Unexpected, bootstrap, load_config, load_dataset, outer_splits, patient_table, run_pipeline, summarise

warnings.filterwarnings("ignore")
RUNS = [("A2-NBI", "biomedclip", "primary"), ("A2-NBI", "dinov2", "secondary"), ("S1", "biomedclip", "sensitivity"), ("S2", "biomedclip", "sensitivity"), ("S3", "biomedclip", "sensitivity")]


def fold_counts(name, d, cfg):
    rows = []
    for r in range(cfg["cv"]["repeats"]):
        for k, (tr, te) in enumerate(outer_splits(d, d.y.values, cfg["cv"]["outer_folds"], cfg["seeds"]["outer_base"] + r)):
            t = d.iloc[te].groupby("patient").label.first().value_counts()
            tt = d.iloc[tr].groupby("patient").label.first().value_counts()
            rows.append({"dataset": name, "repeat": r, "fold": k, **{f"test_{c}": int(t.get(c, 0)) for c in ("IM&CG", "EGC", "HGIN", "LGIN")},
                         "test_patients": int(t.sum()), "train_patients": int(tt.sum()), "test_images": len(te), "train_images": len(tr)})
    return rows


def run(name, emb, role, cfg):
    warnings.filterwarnings("ignore")
    d, X = load_dataset(name, emb)
    p = patient_table(d)
    res = run_pipeline(d, X, dict(zip(p.patient, p.y)), cfg)
    return name, emb, role, p, res


if __name__ == "__main__":
    cfg = load_config()
    if cfg is None:
        raise Unexpected("config_B.json must be frozen before any evaluation")
    best = json.load(open(OUT / "best_shortcut.json"))
    out = Parallel(n_jobs=5, verbose=5)(delayed(run)(n, e, ro, cfg) for n, e, ro in RUNS)
    rows, per, det, sub, counts = [], [], [], [], []
    for name, emb, role, p, res in out:
        y, P, TH = res["y"], res["P"], res["TH"]
        assert (res["patients"] == p.patient.values).all()
        auc, bacc, aucs, baccs = summarise(y, P, TH)
        b0 = best[name]
        base = np.load(f"D:/gim_outputs/goal1/baselines/{name}__{b0['shortcut']}__{b0['model']}.npz", allow_pickle=True)
        if not (base["patients"] == res["patients"]).all():
            raise Unexpected("baseline and model patients differ")
        b = bootstrap(y, P, TH, cfg["bootstrap"]["B"], cfg["seeds"]["bootstrap"], base["P"], base["TH"])
        fold_aucs = []
        for r in range(P.shape[0]):
            for k in np.unique(res["FOLD"][r]):
                m = res["FOLD"][r] == k
                fold_aucs.append(roc_auc_score(y[m], P[r][m]))
        rows.append({"dataset": name, "embedding": emb, "role": role, "n_patients": len(y), "n_pos": int(y.sum()),
                     "auc": auc, "auc_lo": b["auc_ci"][0], "auc_hi": b["auc_ci"][1], "auc_repeat_min": min(aucs), "auc_repeat_max": max(aucs),
                     "auc_per_fold_mean": np.mean(fold_aucs), "auc_per_fold_sd": np.std(fold_aucs),
                     "bal_acc": bacc, "bal_acc_lo": b["bacc_ci"][0], "bal_acc_hi": b["bacc_ci"][1],
                     "best_shortcut": f"{b0['shortcut']}/{b0['model']}", "best_shortcut_auc": b0["auc"], "best_shortcut_bal_acc": b0["bal_acc"],
                     "diff_auc": auc - b0["auc"], "diff_auc_lo": b["diff_auc_ci"][0], "diff_auc_hi": b["diff_auc_ci"][1], "p_diff_auc_two_sided": b["p_two_sided_auc"],
                     "diff_bal_acc": bacc - b0["bal_acc"], "diff_bal_acc_lo": b["diff_bacc_ci"][0], "diff_bal_acc_hi": b["diff_bacc_ci"][1],
                     "convergence_warnings": int(res["info"].convergence_warnings.sum()), "C_values": "/".join(f"{c:.3g}" for c in sorted(res["info"].C.unique())),
                     "threshold_min": res["info"].threshold.min(), "threshold_max": res["info"].threshold.max()})
        per.append(p.assign(dataset=name, embedding=emb, prob_mean=P.mean(0), prob_sd=P.std(0)))
        det.append(res["info"].assign(dataset=name, embedding=emb))
        for neg in sorted(set(p.label) - {"IM&CG"}):
            m = p.label.isin(["IM&CG", neg]).values
            sub.append({"dataset": name, "embedding": emb, "negative_class": neg, "n_neg": int((p.label == neg).sum()),
                        "auc": np.mean([roc_auc_score(y[m], q[m]) for q in P])})
        for yr in sorted(p.year.unique()):
            m = (p.year == yr).values
            if len(np.unique(y[m])) == 2:
                sub.append({"dataset": name, "embedding": emb, "negative_class": f"year {yr}", "n_neg": int((y[m] == 0).sum()),
                            "auc": np.mean([roc_auc_score(y[m], q[m]) for q in P])})
    pd.DataFrame(rows).round(4).to_csv(OUT / "results_B.csv", index=False)
    pd.concat(per).round(5).to_csv(OUT / "per_patient_B.csv", index=False)
    pd.concat(det).round(5).to_csv(OUT / "fold_details_B.csv", index=False)
    pd.DataFrame(sub).round(4).to_csv(OUT / "subgroup_B.csv", index=False)
    cl = pd.read_csv(OUT / "capped_list.csv")
    for name in cfg["datasets"]:
        d = cl[cl.dataset == name].sort_values(["patient", "file"]).reset_index(drop=True)
        counts += fold_counts(name, d, cfg)
    pd.DataFrame(counts).to_csv(OUT / "fold_counts_B.csv", index=False)
    print(pd.DataFrame(rows).round(3).T.to_string())
