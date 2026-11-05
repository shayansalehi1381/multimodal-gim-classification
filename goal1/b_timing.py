"""Goal 1, Step B - time one outer fold per solver on RANDOM labels. Prints time and convergence only, never a metric,
so the solver choice cannot depend on results."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import warnings

import numpy as np

from b_common import load_dataset, outer_fold, outer_splits, patient_table

warnings.filterwarnings("ignore")
d, X = load_dataset("A2-NBI", "biomedclip")
p = patient_table(d)
rng = np.random.default_rng(999)
ymap = dict(zip(p.patient, rng.permutation(p.y.values)))
y = np.array([ymap[q] for q in d.patient])
tr, te = outer_splits(d, y, 5, 7)[0]
for solver, mi in (("lbfgs", 5000), ("newton-cholesky", 200)):
    cfg = {"logistic_regression": {"C_log10_start": -3, "C_log10_stop": 2, "C_n": 11, "solver": solver, "max_iter": mi, "tol": 1e-4},
           "cv": {"inner_folds": 5}, "seeds": {"model": 42}}
    t = time.time()
    *_, conv = outer_fold(X, y, d.patient.values, tr, te, 123, cfg)
    print(f"{solver}: {time.time() - t:.1f}s per outer fold, convergence warnings {conv}", flush=True)
