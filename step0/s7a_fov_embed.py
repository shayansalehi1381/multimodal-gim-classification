"""Sub-step 7a - detect the endoscopic field of view (FOV), crop it, and cache embeddings for both models.

FOV detection (once per capture layout = image resolution, on the max-projection of up to 200 images of that layout), at <=640 px:
  gray = max(R,G,B) > 40 (screen-recording background is ~20-24, OSD text is separate small blobs)
  -> morphological open (3x3) -> largest connected component -> convex hull (endoscope FOV is an octagon)
Crop = largest rectangle with the FOV-bbox aspect ratio, centred on the bbox, whose pixels are >= 99.5% inside
the hull (so black corners and the on-screen text panel are removed).
Items: the 1976 video frames used by v1 (data/filtered_frames) + all 98 raw stills (40 of them used by v1).
Embeddings (fp32, batch 8, eval/no_grad, offline cache):
  BiomedCLIP open_clip 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224' -> 512-d, L2-normalised
  DINOv2 'facebook/dinov2-base' CLS token -> 768-d, L2-normalised (same as v1)
Outputs:
  step0/embeddings_cropped/{biomedclip,dinov2}_cropped.npy + index.csv     (all 2074 items, FOV-cropped)
  step0/embeddings_raw/{biomedclip,dinov2}_stills_raw.npy + index_stills_raw.csv  (98 stills, uncropped)
  results/step0/s7_fov/fov_boxes.csv, results/step0/figures/fov_qc_*.png
"""
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import ROOT, list_media, log, out_dir

STEP0 = ROOT / "step0"
EMB_C = STEP0 / "embeddings_cropped"
EMB_R = STEP0 / "embeddings_raw"
for d in (EMB_C, EMB_R):
    d.mkdir(exist_ok=True)
OUT = out_dir("s7_fov")
FIG = out_dir("figures")
FF = ROOT / "data" / "filtered_frames"
THR, COVER, WORK = 40, 0.995, 640
torch.manual_seed(42)


def imread(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im


def fov_from_gray(gmax, full_hw):
    """gmax: max-channel image at working scale. Returns hull mask (working scale), bbox and crop (full scale)."""
    h, w = gmax.shape
    m = (gmax > THR).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    mask = np.zeros_like(m)
    cv2.fillPoly(mask, [hull], 1)
    x, y, bw, bh = cv2.boundingRect(hull)
    cx, cy = x + bw / 2, y + bh / 2
    lo, hi = 0.3, 1.0
    for _ in range(30):
        s = (lo + hi) / 2
        x0, x1 = int(round(cx - s * bw / 2)), int(round(cx + s * bw / 2))
        y0, y1 = int(round(cy - s * bh / 2)), int(round(cy + s * bh / 2))
        cover = mask[max(y0, 0):y1, max(x0, 0):x1].mean()
        lo, hi = (s, hi) if cover >= COVER else (lo, s)
    s = lo
    sx, sy = full_hw[1] / w, full_hw[0] / h
    rect = [int(round((cx - s * bw / 2) * sx)), int(round((cy - s * bh / 2) * sy)),
            int(round((cx + s * bw / 2) * sx)), int(round((cy + s * bh / 2) * sy))]
    bbox = [int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)]
    return mask, hull, bbox, rect, s


def work_scale(im):
    h, w = im.shape[:2]
    f = WORK / max(h, w)
    return cv2.resize(im, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)


def center_crop_nonfov(mask, frac_side=1.0):
    """Fraction of pixels outside the FOV in the centred square a model sees after resize+center-crop."""
    h, w = mask.shape
    side = int(min(h, w) * frac_side)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return 1 - mask[y0:y0 + side, x0:x0 + side].mean()


# ── 1. items ────────────────────────────────────────────────────────────────
media = list_media()
stills = media[~media.is_video].copy()
filtered = sorted(os.listdir(FF))
vframes = [f for f in filtered if "_vid_frame" in f]
items = []
for f in vframes:
    m = re.match(r"^CASE_(\d{4})_(.+_vid)_frame\d+\.jpg$", f)
    items.append({"key": f, "path": FF / f, "case": int(m.group(1)), "kind": "video_frame",
                  "group": m.group(2), "used_in_v1": True})
for f in stills.file:
    items.append({"key": f, "path": ROOT / f, "case": int(f[:4]), "kind": "still", "group": f,
                  "used_in_v1": f"CASE_{f[:4]}_{f}" in set(filtered)})
items = pd.DataFrame(items)

# ── 2. FOV per capture layout ───────────────────────────────────────────────
# The FOV octagon and the OSD panel are fixed by the capture device/layout, which is identified by the image
# resolution. A per-image mask fails when a dark lumen touches the FOV edge, so the FOV is estimated once per
# resolution from the max-projection over (a sample of) all images of that resolution, then applied to each item.
def imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))          # cv2.imwrite cannot handle the non-ASCII project path


hw = {}
for g, gi in items.groupby("group", sort=False):
    hw[g] = imread(gi.path.iloc[0]).shape[:2]
items["layout"] = [f"{hw[g][1]}x{hw[g][0]}" for g in items.group]
fov, qc, layouts = {}, [], {}
for lay, li in items.groupby("layout"):
    paths = li.path.tolist()
    sample = paths if len(paths) <= 200 else paths[:: max(1, len(paths) // 200)]
    full_hw = hw[li.group.iloc[0]]
    acc = None
    for p in sample:
        gm = work_scale(imread(p)).max(axis=2)
        acc = gm if acc is None else np.maximum(acc, gm)
    mask, hull, bbox, rect, s = fov_from_gray(acc, full_hw)
    layouts[lay] = dict(layout=lay, n_items=len(li), n_sources=li.group.nunique(), n_used_for_projection=len(sample),
                        bbox=str(bbox), rect=str(rect), scale=s,
                        nonfov_v1_biomedclip=center_crop_nonfov(mask, 1.0),
                        nonfov_v1_dinov2=center_crop_nonfov(mask, 224 / 256))
    for g in li.group.unique():
        fov[g] = dict(layout=lay, bbox=bbox, rect=rect, scale=s, full_hw=full_hw,
                      nonfov_v1_biomedclip=layouts[lay]["nonfov_v1_biomedclip"],
                      nonfov_v1_dinov2=layouts[lay]["nonfov_v1_dinov2"])
    # QC: up to 6 example items of this layout with hull (green) and crop (red)
    for p in li.path.iloc[np.linspace(0, len(li) - 1, min(6, len(li))).astype(int)]:
        full = imread(p)
        vis = work_scale(full)
        cv2.polylines(vis, [hull], True, (0, 255, 0), 2)
        f2 = vis.shape[1] / full.shape[1]
        cv2.rectangle(vis, (int(rect[0] * f2), int(rect[1] * f2)), (int(rect[2] * f2), int(rect[3] * f2)), (0, 0, 255), 2)
        cv2.putText(vis, f"{lay} {os.path.basename(str(p))[:32]}", (5, vis.shape[0] - 8), 0, 0.45, (0, 255, 255), 1)
        qc.append(cv2.resize(vis, (480, 300)))
pd.DataFrame(layouts.values()).to_csv(OUT / "fov_layouts.csv", index=False)
boxes = pd.DataFrame([{"group": g, **{k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in d.items()}}
                      for g, d in fov.items()])
boxes.to_csv(OUT / "fov_boxes.csv", index=False)
for page in range(0, len(qc), 20):
    tiles = qc[page:page + 20]
    tiles += [np.zeros_like(qc[0])] * (20 - len(tiles))
    grid = np.vstack([np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(4)])
    imwrite(FIG / f"fov_qc_{page // 20 + 1}.png", grid)
if os.environ.get("FOV_ONLY"):
    print(pd.DataFrame(layouts.values()).to_string(index=False))
    raise SystemExit(0)

# ── 3. embeddings ───────────────────────────────────────────────────────────
dev = "cuda" if torch.cuda.is_available() else "cpu"
import open_clip
from transformers import AutoImageProcessor, AutoModel

clip_model, _, clip_pre = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
clip_model = clip_model.to(dev).eval()
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
dino = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(dev).eval()


def load_pil(path, rect=None):
    im = cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)
    if rect is not None:
        x0, y0, x1, y1 = rect
        im = im[max(y0, 0):y1, max(x0, 0):x1]
    return Image.fromarray(im)


@torch.no_grad()
def embed(paths, rects, bs=8):
    E1, E2 = [], []
    for i in range(0, len(paths), bs):
        ims = [load_pil(p, r) for p, r in zip(paths[i:i + bs], rects[i:i + bs])]
        x = torch.stack([clip_pre(im) for im in ims]).to(dev)
        E1.append(torch.nn.functional.normalize(clip_model.encode_image(x), dim=-1).cpu().numpy())
        inp = dino_proc(images=ims, return_tensors="pt").to(dev)
        cls = dino(**inp).last_hidden_state[:, 0, :]
        E2.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        if (i // bs) % 50 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.vstack(E1).astype(np.float32), np.vstack(E2).astype(np.float32)


rects = [fov[g]["rect"] for g in items.group]
bc, dc = embed(items.path.tolist(), rects)
np.save(EMB_C / "biomedclip_cropped.npy", bc)
np.save(EMB_C / "dinov2_cropped.npy", dc)
idx = items.drop(columns="path").assign(crop_rect=[str(r) for r in rects])
idx.to_csv(EMB_C / "index.csv", index=False)

st = items[items.kind == "still"].reset_index(drop=True)
br, dr = embed(st.path.tolist(), [None] * len(st))
np.save(EMB_R / "biomedclip_stills_raw.npy", br)
np.save(EMB_R / "dinov2_stills_raw.npy", dr)
st.drop(columns="path").to_csv(EMB_R / "index_stills_raw.csv", index=False)

# ── 4. checks ───────────────────────────────────────────────────────────────
from common import PROC
fm = pd.read_csv(PROC / "frame_metadata.csv")
dm = pd.read_csv(PROC / "dinov2_frame_metadata.csv")
v1b = np.load(PROC / "frame_embeddings_1976x512.npy")
v1d = np.load(PROC / "dinov2_frame_embeddings_2016x768.npy")
cos_b, cos_d = [], []
for i, r in st[st.used_in_v1].iterrows():
    name = f"CASE_{r.case}_{r.key}"
    cos_b.append(float(br[i] @ v1b[fm[fm.filename == name].embedding_idx.iloc[0]]))
    cos_d.append(float(dr[i] @ v1d[dm[dm.filename == name].embedding_idx.iloc[0]]))
bx = boxes.assign(kind=np.where(boxes.group.str.endswith("_vid"), "video", "still"))
txt = f"""
FOV estimated per capture layout (resolution): {len(layouts)} layouts covering {len(fov)} sources ({(bx.kind=='video').sum()} videos, {(bx.kind=='still').sum()} stills); see fov_layouts.csv.
QC montages: results/step0/figures/fov_qc_*.png (green = FOV hull, red = crop).
Crop scale vs FOV bbox: median {boxes.scale.median():.2f} (min {boxes.scale.min():.2f}).
Share of the v1 model input (center square) that lay outside the FOV, i.e. black border or on-screen text:
BiomedCLIP median {bx.nonfov_v1_biomedclip.median():.1%} (videos {bx[bx.kind=='video'].nonfov_v1_biomedclip.median():.1%},
stills {bx[bx.kind=='still'].nonfov_v1_biomedclip.median():.1%}, max {bx.nonfov_v1_biomedclip.max():.1%}).
Embeddings cached: step0/embeddings_cropped/ ({len(items)} items x 512 / 768) and step0/embeddings_raw/ (98 raw stills).
Check: raw-still embeddings recomputed here vs the v1 embeddings of the same 40 stills: cosine BiomedCLIP min {min(cos_b):.4f},
DINOv2 min {min(cos_d):.4f} (1.0 = identical preprocessing).
"""
log("Sub-step 7a - FOV detection, crop and embedding cache (run before 6 because 6e uses the raw-still embeddings)", txt)
print(txt)
