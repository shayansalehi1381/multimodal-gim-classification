"""Goal 1, Step B - permutation test: shuffle patient labels and repeat the WHOLE pipeline (repeats, outer folds, inner CV).

Runs listed in config_B.json. Results append to results/goal1/permutation_B_<dataset>_<embedding>.csv in chunks, so a
restart skips finished permutations. p = (1 + #{null >= observed}) / (N + 1).
Usage: python b4_permutation.py <dataset> <embedding> [n_jobs]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from b_common import OUT, Unexpected, load_config, load_dataset, patient_table, run_pipeline, summarise

warnings.filterwarnings("ignore")


def one(i, d, X, pts, yperm, cfg):
    warnings.filterwarnings("ignore")
    try:
        res = run_pipeline(d, X, dict(zip(pts, yperm)), cfg)
    except Unexpected as ex:
        return {"perm": i, "auc": np.nan, "bal_acc": np.nan, "problem": str(ex)}
    auc, bacc, _, _ = summarise(res["y"], res["P"], res["TH"])
    return {"perm": i, "auc": auc, "bal_acc": bacc, "problem": ""}


if __name__ == "__main__":
    name, emb = sys.argv[1], sys.argv[2]
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    cfg = load_config()
    if cfg is None:
        raise Unexpected("config_B.json must be frozen before any evaluation")
    N = cfg["permutation"]["N"]
    d, X = load_dataset(name, emb)
    p = patient_table(d)
    rng = np.random.default_rng(cfg["seeds"]["permutation"])
    perms = [rng.permutation(p.y.values) for _ in range(N)]
    f = OUT / f"permutation_B_{name}_{emb}.csv"
    done = set(pd.read_csv(f).perm) if os.path.exists(f) else set()
    todo = [i for i in range(N) if i not in done]
    for s in range(0, len(todo), n_jobs * 2):
        chunk = todo[s:s + n_jobs * 2]
        rows = Parallel(n_jobs=n_jobs)(delayed(one)(i, d, X, p.patient.values, perms[i], cfg) for i in chunk)
        pd.DataFrame(rows).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        print(f"{name}/{emb}: {s + len(chunk)}/{len(todo)}", flush=True)
    print("done", flush=True)
