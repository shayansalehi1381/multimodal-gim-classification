"""Sub-step 0 - data integrity (read-only on the data).

Outputs (results/step0/s0_integrity/):
  media_inventory.csv       one row per media file (sha256, decode check, size, blur score, used?, reason)
  patient_inventory.csv     one row per patient (images, videos, ext, processor, cohort status, label)
  excluded_images.csv       the 58 still images that never reached data/filtered_frames, with reason
  duplicates.csv            exact (sha256) and near (dHash <= 6 bits) duplicate image pairs
  contingency_*.csv         label vs processor / media source / file type (+ Fisher exact p)
"""
import hashlib
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import fisher_exact

from common import ROOT, PROC, list_media, log, out_dir

os.chdir(ROOT)  # cv2.VideoCapture cannot open the non-ASCII absolute path; use relative names
OUT = out_dir("s0_integrity")
BLUR_THRESHOLD = 100.0                     # phase1_filter_blurry.py:12
HARDCODED = ['2601', '2602', '2603', '2604', '2605', '2606', '2607', '2608',
             '2611', '2613', '2614', '2615']  # phase2_task3_fix_missing_frames.py:7


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(gray, n=16):
    small = cv2.resize(gray, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


media = list_media()
filtered = set(os.listdir(os.path.join("data", "filtered_frames")))
rows, hashes = [], {}
for _, m in media.iterrows():
    f = m["file"]
    r = dict(m)
    r["bytes"] = os.path.getsize(f)
    r["sha256"] = sha256(f)
    if not m["is_video"]:
        try:
            with Image.open(f) as im:
                im.load()
            r["decodes_ok"] = True
        except Exception as ex:
            r["decodes_ok"] = False
            r["decode_error"] = str(ex)
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r["height"], r["width"] = gray.shape
            r["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hashes[f] = dhash(gray)
        used_name = f"CASE_{m['case']}_{f}"
        r["used_in_v1"] = used_name in filtered
        if r["used_in_v1"]:
            r["reason"] = "used"
        elif str(m["case"]) not in HARDCODED:
            has_video = media[(media.case == m["case"]) & media.is_video].shape[0] > 0
            r["reason"] = ("not in hard-coded case list (fix_missing_frames.py:7): patient has videos, "
                           "only its video frames were used") if has_video else \
                          "not in hard-coded case list (fix_missing_frames.py:7): patient not in cohort"
        elif img is None:
            r["reason"] = "cv2 could not decode"
        elif r["laplacian_var"] < BLUR_THRESHOLD:
            r["reason"] = f"blur filter: Laplacian var {r['laplacian_var']:.1f} < {BLUR_THRESHOLD}"
        else:
            r["reason"] = "UNEXPLAINED"
    else:
        cap = cv2.VideoCapture(f)
        r["decodes_ok"] = bool(cap.isOpened())
        r["fps"] = cap.get(cv2.CAP_PROP_FPS)
        r["n_frames"] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        r["width"], r["height"] = cap.get(3), cap.get(4)
        r["duration_s"] = r["n_frames"] / r["fps"] if r["fps"] else np.nan
        r["fps_int_used_by_v1"] = int(r["fps"])
        cap.release()
        base = os.path.splitext(f)[0]
        r["v1_frames_kept"] = sum(1 for x in filtered if x.startswith(f"CASE_{m['case']}_{base}_frame"))
        r["used_in_v1"] = r["v1_frames_kept"] > 0
        r["reason"] = "used" if r["used_in_v1"] else "no frames kept"
    rows.append(r)
inv = pd.DataFrame(rows)
inv.to_csv(OUT / "media_inventory.csv", index=False)

# ── duplicates ───────────────────────────────────────────────────────────────
dup = []
for sha, g in inv.groupby("sha256"):
    if len(g) > 1:
        dup.append(dict(kind="exact_sha256", a=g.file.iloc[0], b=g.file.iloc[1], hamming=0))
names = sorted(hashes)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        d = int((hashes[a] != hashes[b]).sum())
        if d <= 6:
            dup.append(dict(kind="near_dhash256", a=a, b=b, hamming=d))
dups = pd.DataFrame(dup, columns=["kind", "a", "b", "hamming"])
if len(dups):
    dups["cross_patient"] = dups.a.str[:4] != dups.b.str[:4]
dups.to_csv(OUT / "duplicates.csv", index=False)

# ── cohort reproduction from the raw metadata ───────────────────────────────
meta = pd.read_excel("GIMENDO_v2_Metadata.xlsx", header=1)
meta = meta[pd.to_numeric(meta["Case Code"], errors="coerce").notna()].copy()
meta["Case Code"] = meta["Case Code"].astype(int)


def cohort_status(r):
    s = str(r["Histological Subtype"]).strip()
    dys = str(r.get("Dysplasia", "")).strip()
    if pd.isna(r["Histological Subtype"]):
        return "excluded: no histology (control)"
    if s.lower() == "negative":
        return "excluded: histology negative for IM"
    if s == "IM Positive":
        return "excluded: IM positive, subtype not reported"
    if dys not in ("Negative", "nan"):
        return f"excluded: dysplasia ({dys})"
    return "included"


meta["cohort_status"] = meta.apply(cohort_status, axis=1)
meta["label"] = np.where(meta["Histological Subtype"].astype(str).str.contains("Incomplete"), "Incomplete",
                         np.where(meta["Histological Subtype"].astype(str).str.contains("Complete"), "Complete", ""))
meta["mixed_histology"] = meta["Histological Subtype"].astype(str).str.contains(r"Complete \+ Incomplete")
cohort_csv = pd.read_csv(PROC / "multimodal_biomedclip_dinov2_n17.csv")
assert set(meta.loc[meta.cohort_status == "included", "Case Code"]) == set(cohort_csv["Case Code"]), "cohort mismatch"

img, vid = inv[~inv.is_video], inv[inv.is_video]
pt = pd.DataFrame({
    "n_images": img.groupby("case").size(),
    "n_videos": vid.groupby("case").size(),
    "img_ext": img.groupby("case").ext.agg(lambda s: "/".join(sorted(set(s)))),
    "vid_ext": vid.groupby("case").ext.agg(lambda s: "/".join(sorted(set(s)))),
    "processor_filename": inv.groupby("case").processor.agg(lambda s: "/".join(sorted(set(s.dropna())))),
    "modalities": inv.groupby("case").modality.agg(lambda s: "/".join(sorted(set(s.dropna())))),
    "images_used_v1": img.groupby("case").used_in_v1.sum(),
    "video_frames_used_v1": vid.groupby("case").v1_frames_kept.sum(),
}).fillna({"n_images": 0, "n_videos": 0, "img_ext": "", "vid_ext": "", "images_used_v1": 0,
           "video_frames_used_v1": 0})
pt.index.name = "Case Code"
pt = pt.join(meta.set_index("Case Code")[["Histological Subtype", "label", "mixed_histology", "cohort_status",
                                          "Media Type", "Processor", "Scope"]])
pt = pt.rename(columns={"Processor": "processor_metadata", "Scope": "scope_metadata"})
pt["v1_visual_source"] = np.where(pt.video_frames_used_v1 > 0, "video frames only",
                                  np.where(pt.images_used_v1 > 0, "stills only", "none"))
pt.to_csv(OUT / "patient_inventory.csv")

exc = inv[(~inv.is_video) & (~inv.used_in_v1.astype(bool))][["file", "case", "ext", "bytes", "width", "height",
                                                             "laplacian_var", "decodes_ok", "reason"]]
exc = exc.merge(pt[["cohort_status"]], left_on="case", right_index=True)
exc.to_csv(OUT / "excluded_images.csv", index=False)

# ── contingency tables (cohort) ─────────────────────────────────────────────
coh = pt[pt.cohort_status == "included"].copy()
coh["png_stills"] = np.where(coh.v1_visual_source == "video frames only", "video (jpg frames)",
                             np.where(coh.img_ext == ".png", "png stills", "jpg stills"))
tables = {}
for name, col in [("processor", "processor_filename"), ("media_source", "v1_visual_source"),
                  ("file_type", "png_stills"), ("processor_metadata", "processor_metadata")]:
    ct = pd.crosstab(coh[col].fillna("missing"), coh["label"])
    p = fisher_exact(ct.values)[1] if ct.shape == (2, 2) else np.nan
    ct["fisher_exact_p"] = p
    ct.to_csv(OUT / f"contingency_{name}.csv")
    tables[name] = ct

# all GIM-positive patients (incl. unconfirmed subtype) vs processor
allpos = pt[pt.label != ""]
pd.crosstab(allpos.processor_filename, allpos.label).to_csv(OUT / "contingency_processor_all_with_label.csv")

# ── summary + log ────────────────────────────────────────────────────────────
corrupt = inv[inv.decodes_ok == False].file.tolist()  # noqa: E712
reasons = exc.reason.str.split(":").str[0].value_counts().to_dict()
summary = f"""
Inventory: {inv.case.nunique()} patients with media (metadata rows: {len(meta)}; 2616 absent in both),
{len(img)} images ({(img.ext=='.jpg').sum()} jpg, {(img.ext=='.png').sum()} png), {len(vid)} videos
({(vid.ext=='.mp4').sum()} mp4, {(vid.ext=='.mkv').sum()} mkv). Corrupt/undecodable files: {corrupt}.
Exact SHA256 duplicates: {(dups.kind=='exact_sha256').sum() if len(dups) else 0}; near-duplicate image pairs
(dHash-256 <= 6 bits): {(dups.kind=='near_dhash256').sum() if len(dups) else 0}
(cross-patient: {int(dups.cross_patient.sum()) if len(dups) else 0}). 2612_8 is a truncated JPEG with the same byte size
and header as 2612_9 (not a byte duplicate). .mkv: all 3 videos of 2619 open in OpenCV (30 fps, 1530 frames)
and 153 frames (3x51) are in the v1 embeddings. Video fps is cast with int() in v1 (13.73->13, 50.02->50).
Cohort: reproduced from GIMENDO_v2_Metadata.xlsx column 'Histological Subtype' (rules of
phase1_process_metadata.py:40-57) -> 17 patients, 9 Incomplete / 8 Complete; 2 of the 9 Incomplete are mixed
'Complete + Incomplete' (2602, 2611) mapped to Incomplete by substring; 2612 (mixed + LGD) excluded for dysplasia.
Excluded images: {len(exc)} of {len(img)} stills never reached data/filtered_frames (only {int(img.used_in_v1.sum())} used).
Reasons: {reasons}. Root cause is a hard-coded case list in phase2_task3_fix_missing_frames.py:7, not the blur
filter: 45 stills of the 10 video patients (26 of them in cohort patients 2610/2619/2620/2622/2625) and 10 stills
of 2609/2612 (out of cohort) were never considered; the blur filter removed only 3 stills (2613) and 1 video frame.
Processor (filename token) vs label in the cohort: CV-190 = {tables['processor'].get('Incomplete', {}).get('190', 0)}
Incomplete / {tables['processor'].get('Complete', {}).get('190', 0)} Complete (Fisher p={tables['processor']['fisher_exact_p'].iloc[0]:.3f}).
png stills are Incomplete only as well (2613, 2614, 2615). These are primary confounders (see contingency_*.csv).
Lopocv pkl: label column removed in step0/phase3_task1_lopocv_setup_fixed.py (sub-step 1).
Files: results/step0/s0_integrity/{{media_inventory, patient_inventory, excluded_images, duplicates, contingency_*}}.csv
"""
log("Sub-step 0 - Data integrity", summary)
print(summary)
for k, v in tables.items():
    print(f"\n[{k}]\n{v}")
print(dups.to_string(index=False) if len(dups) else "no duplicates")
