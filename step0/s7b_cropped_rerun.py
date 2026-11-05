"""Sub-step 7b - re-run the champion protocol and the visual-only models on FOV-cropped embeddings.

Patient features = mean of frame embeddings, then L2 (as v1), built three ways:
  uncropped_v1      : the tracked v1 features (same frames as v1)
  cropped_v1frames  : FOV-cropped embeddings of exactly the same frames/stills v1 used
  cropped_allmedia  : FOV-cropped embeddings of ALL media of each patient (adds the 26 stills of cohort video
                      patients and the 3 blur-dropped 2613 stills that v1 never used - see sub-step 0)
Models: v1 champion protocol (SVC, 8 PCs, clinical + visual, inner-CV Youden) and the nested step-0 models
(b BiomedCLIP, c DINOv2, d both, e both+clinical; LR and SVC).
Outputs: results/step0/s7_fov/cropped_vs_uncropped.csv, per_patient_cropped.csv
"""
import warnings

import numpy as np
import pandas as pd

from common import CLINICAL_COLS, ROOT, load_cohort, log, metrics, out_dir, run_nested_lopo, v1_run

warnings.filterwarnings("ignore")
OUT = out_dir("s7_fov")

df, y, b = load_cohort()
cases = df["Case Code"].values
idx = pd.read_csv(ROOT / "step0" / "embeddings_cropped" / "index.csv")
E = {"biomedclip": np.load(ROOT / "step0" / "embeddings_cropped" / "biomedclip_cropped.npy"),
     "dinov2": np.load(ROOT / "step0" / "embeddings_cropped" / "dinov2_cropped.npy")}


def pool(mask):
    out = {}
    for k, M in E.items():
        rows = []
        for c in cases:
            sel = (idx.case == c).values & mask
            v = M[sel].mean(0)
            rows.append(v / np.linalg.norm(v))
        out[k] = np.array(rows)
    return out


variants = {"uncropped_v1": {"biomedclip": b["biomedclip"], "dinov2": b["dinov2"]},
            "cropped_v1frames": pool(idx.used_in_v1.values.astype(bool)),
            "cropped_allmedia": pool(np.ones(len(idx), bool))}
n_items = {v: None for v in variants}
n_items["cropped_v1frames"] = int(idx[idx.case.isin(cases) & idx.used_in_v1].shape[0])
n_items["cropped_allmedia"] = int(idx[idx.case.isin(cases)].shape[0])

rows, per = [], []
for vname, V in variants.items():
    Xc = b["clinical"]
    prob, thr = v1_run(Xc, np.hstack([V["biomedclip"], V["dinov2"]]), y, "SVC", 8)
    pred = (prob >= thr).astype(int)
    rows.append({"variant": vname, "model": "v1_champion_SVC8 (v1 protocol)", **metrics(y, prob, pred)})
    per.append(pd.DataFrame({"variant": vname, "model": "v1_champion_SVC8", "Case Code": cases, "y": y,
                             "score": prob, "pred": pred}))
    for fs, (useC, vis) in {"b_biomedclip": (False, ["biomedclip"]), "c_dinov2": (False, ["dinov2"]),
                            "d_both_visual": (False, ["biomedclip", "dinov2"]),
                            "e_visual+clinical": (True, ["biomedclip", "dinov2"])}.items():
        Xv = np.hstack([V[k] for k in vis])
        for m in ("LR", "SVC"):
            r = run_nested_lopo(Xc if useC else None, Xv, y, m)
            rows.append({"variant": vname, "model": f"{fs}/{m} (nested)", **metrics(y, r.score, r.pred)})
            per.append(pd.DataFrame({"variant": vname, "model": f"{fs}/{m}", "Case Code": cases, "y": y,
                                     "score": r.score.values, "pred": r.pred.values}))
    print("variant done:", vname, flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "cropped_vs_uncropped.csv", index=False)
pd.concat(per).to_csv(OUT / "per_patient_cropped.csv", index=False)
wide = R.pivot(index="model", columns="variant", values="auc")[list(variants)]
wacc = R.pivot(index="model", columns="variant", values="n_correct")[list(variants)]
lines = "\n".join(f"  - {m}: correct {'/'.join(str(int(x)) for x in wacc.loc[m])} ; AUC {'/'.join(f'{x:.3f}' for x in wide.loc[m])}"
                  for m in wide.index)
log("Sub-step 7b - Champion + visual-only models on FOV-cropped embeddings", f"""
Three feature versions (uncropped v1 / cropped, same frames as v1 ({n_items['cropped_v1frames']} items) / cropped, all media
of the 17 patients ({n_items['cropped_allmedia']} items)). Values are given in that order, as n_correct/17 and AUC:
{lines}
Files: results/step0/s7_fov/cropped_vs_uncropped.csv, per_patient_cropped.csv
""")
print(R[["variant", "model", "n_correct", "balanced_accuracy", "auc"]].round(3).to_string(index=False))
