"""Correction to sub-step 6 - text-panel (OSD) probe with labels assigned per capture layout.

The first version labelled every png still and every video frame as "panel present". That is wrong for the
controls 2623/2624 (480x288 videos and six 26xx-wide png stills have no side panel), so 46 positives were mislabelled.
Panel layouts below were checked by eye on results/step0/figures/fov_qc_*.png.
Same frame sample, model and grouped leave-one-patient-out scheme as s6_confounders.py.
Output: results/step0/s6_confounders/shortcut_probe_osd_corrected.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import PROC, ROOT, log, out_dir

warnings.filterwarnings("ignore")
OUT = out_dir("s6_confounders")
PANEL_LAYOUTS = {"1280x720", "1920x1080", "2879x1799", "2880x1800", "650x497", "707x566"}

idx = pd.read_csv(ROOT / "step0" / "embeddings_cropped" / "index.csv")
idx_r = pd.read_csv(ROOT / "step0" / "embeddings_raw" / "index_stills_raw.csv")
emb = {("biomedclip", "cropped"): np.load(ROOT / "step0" / "embeddings_cropped" / "biomedclip_cropped.npy"),
       ("dinov2", "cropped"): np.load(ROOT / "step0" / "embeddings_cropped" / "dinov2_cropped.npy")}
fm, dm = pd.read_csv(PROC / "frame_metadata.csv"), pd.read_csv(PROC / "dinov2_frame_metadata.csv")
v1b, v1d = np.load(PROC / "frame_embeddings_1976x512.npy"), np.load(PROC / "dinov2_frame_embeddings_2016x768.npy")
rawb = np.load(ROOT / "step0" / "embeddings_raw" / "biomedclip_stills_raw.npy")
rawd = np.load(ROOT / "step0" / "embeddings_raw" / "dinov2_stills_raw.npy")
fpos_b, fpos_d = dict(zip(fm.filename, fm.embedding_idx)), dict(zip(dm.filename, dm.embedding_idx))
raw_pos = dict(zip(idx_r.key, range(len(idx_r))))
rb, rd = [], []
for r in idx.itertuples():
    if r.kind == "still":
        rb.append(rawb[raw_pos[r.key]]); rd.append(rawd[raw_pos[r.key]])
    else:
        rb.append(v1b[fpos_b[r.key]]); rd.append(v1d[fpos_d[r.key]])
emb[("biomedclip", "raw")], emb[("dinov2", "raw")] = np.array(rb), np.array(rd)

idx["has_panel"] = idx.layout.isin(PANEL_LAYOUTS).astype(int)
keep = idx[idx.kind == "video_frame"].groupby("case").apply(
    lambda g: g.sample(min(len(g), 20), random_state=42)).index.get_level_values(-1)
rows_sel = np.sort(np.concatenate([idx.index[idx.kind == "still"].values, keep.values]))


def grouped_probe(X, yb, groups):
    sc = np.zeros(len(yb))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        s = StandardScaler().fit(X[tr])
        pca = PCA(n_components=min(16, tr.sum() - 1), svd_solver="full", random_state=42).fit(s.transform(X[tr]))
        m = LogisticRegression(C=1.0, max_iter=5000, random_state=42).fit(pca.transform(s.transform(X[tr])), yb[tr])
        sc[te] = m.predict_proba(pca.transform(s.transform(X[te])))[:, 1]
    return roc_auc_score(yb, sc), balanced_accuracy_score(yb, (sc >= 0.5).astype(int))


subsets = {"all items": rows_sel,
           "stills only (no video frames)": idx.index[idx.kind == "still"].values,
           "excluding controls 2623/2624": rows_sel[~idx.loc[rows_sel, "case"].isin([2623, 2624]).values]}
res = []
for sname, rows in subsets.items():
    yb, gr = idx.loc[rows, "has_panel"].values, idx.loc[rows, "case"].values
    for (mdl, ver), E in emb.items():
        auc, bacc = grouped_probe(E[rows], yb, gr)
        res.append({"subset": sname, "embedding": mdl, "version": ver, "n_items": len(rows), "n_panel": int(yb.sum()),
                    "n_patients": len(np.unique(gr)), "grouped_LOPO_auc": auc, "grouped_LOPO_bacc@0.5": bacc})
R = pd.DataFrame(res)
R.to_csv(OUT / "shortcut_probe_osd_corrected.csv", index=False)
wide = R.pivot_table(index=["subset", "embedding"], columns="version", values="grouped_LOPO_auc")
lines = "\n".join(f"  - {s} / {e}: raw {r['raw']:.3f} -> cropped {r['cropped']:.3f}" for (s, e), r in wide.iterrows())
log("Correction (2026-11-05) - text-panel probe re-run with labels per capture layout", f"""
Replaces the OSD_vs_none rows in shortcut_probe.csv, which mislabelled 46 items from controls 2623/2624 as "panel present".
Grouped leave-one-patient-out AUC:
{lines}
Panel and patient are not independent here, so these AUCs still mix panel with other acquisition differences.
File: results/step0/s6_confounders/shortcut_probe_osd_corrected.csv
""")
print(lines)
