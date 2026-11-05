"""Sub-step 7c - Attention Rollout with the geometry fixed and FOV-cropped input.

v1 bug (phase4_task3_attention_rollout.py:156 + :170): the model sees only the CENTRE SQUARE of the frame
(open_clip preprocess = Resize(224, shortest side) + CenterCrop(224)), but v1 stretched the 14x14 map over
the WHOLE non-square frame -> attention is drawn horizontally stretched and shifted (wrong location).
Fix: the map is placed exactly on the region the model saw: centre square of the input region
(BiomedCLIP: side = min(w,h); DINOv2: Resize(256)+CenterCrop(224) -> side = min(w,h)*224/256),
where the input region is either the full frame (fixed geometry only) or the FOV crop rectangle (sub-step 7a).
Check: the preprocessed tensor itself is mapped back onto the original frame with (a) the v1 mapping and
(b) the fixed mapping; the pixel error against the original frame shows which mapping is geometrically correct.
Cases: 2601, 2602, 2603 (the GI advisor's reference image), 2605 - same frames as v1 (first still of each case).
Rollout: mean over heads, + identity, row-normalised, multiplied over layers (Abnar & Zuidema 2020),
CLS row, per-image min-max. It is class-agnostic: it shows where the encoder attends, not evidence for a class.
Outputs: results/step0/figures/rollout_<case>.png, results/step0/s7_fov/rollout_mapping_check.csv
"""
import os
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import ast

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import RES, ROOT, log, out_dir

FIG, OUT = out_dir("figures"), out_dir("s7_fov")
CASES = {"2601": "Incomplete", "2602": "Incomplete (mixed C+I)", "2603": "Complete", "2605": "Incomplete"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
boxes = pd.read_csv(OUT / "fov_boxes.csv").set_index("group")

# first still of each case in sorted order = the frame v1 used
ff = sorted(os.listdir(ROOT / "data" / "filtered_frames"))
frames = {c: next(f for f in ff if f.startswith(f"CASE_{c}_"))[10:] for c in CASES}

# ── models ──────────────────────────────────────────────────────────────────
import open_clip
from transformers import AutoImageProcessor, AutoModel

clip_model, _, clip_pre = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
clip_model = clip_model.to(dev).eval()
attn_store = []


def patched_forward(self, x, attn_mask=None, is_causal=False):   # identical maths to v1 (eager attention)
    B, N, C = x.shape
    gate = self.gate(x).sigmoid() if getattr(self, "gate", None) is not None else None
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn_store.append(attn.mean(dim=1).detach())
    x = (attn @ v).transpose(1, 2).reshape(B, N, getattr(self, "attn_dim", C))
    x = self.norm(x)
    if gate is not None:
        x = x * gate
    return self.proj_drop(self.proj(x))


for blk in clip_model.visual.trunk.blocks:
    blk.attn.forward = types.MethodType(patched_forward, blk.attn)
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
dino = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True,
                                 attn_implementation="eager").to(dev).eval()


def rollout(atts):
    N = atts[0].shape[-1]
    res = torch.eye(N, device=atts[0].device)
    for a in atts:
        a = a[0] + torch.eye(N, device=a.device)
        a = a / a.sum(-1, keepdim=True)
        res = a @ res
    m = res[0, 1:]
    g = int(np.sqrt(m.shape[0]))
    m = m.reshape(g, g).cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


@torch.no_grad()
def clip_map(pil):
    attn_store.clear()
    x = clip_pre(pil).unsqueeze(0).to(dev)
    clip_model.visual(x)
    return rollout(attn_store), x[0].cpu()


@torch.no_grad()
def dino_map(pil):
    inp = dino_proc(images=pil, return_tensors="pt").to(dev)
    out = dino(**inp, output_attentions=True)
    return rollout([a.mean(dim=1) for a in out.attentions]), inp["pixel_values"][0].cpu()


def seen_square(region, side_frac):
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    s = min(w, h) * side_frac
    cx, cy = x0 + w / 2, y0 + h / 2
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def place(map_small, sq, H, W):
    """Fixed mapping: put the map on the square the model saw; NaN elsewhere (not seen)."""
    out = np.full((H, W), np.nan)
    x0, y0, x1, y1 = [int(round(v)) for v in sq]
    up = cv2.resize(map_small.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    xs, ys = max(x0, 0), max(y0, 0)
    out[ys:min(y1, H), xs:min(x1, W)] = up[ys - y0:min(y1, H) - y0, xs - x0:min(x1, W) - x0]
    return np.clip(out, 0, 1)


def denorm(t, mean, std):
    return np.clip((t.numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)) * 255, 0, 255).astype(np.uint8)


CLIP_MS = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
DINO_MS = (dino_proc.image_mean, dino_proc.image_std)
checks = []
for case, label in CASES.items():
    fname = frames[case]
    img = np.array(Image.open(ROOT / fname).convert("RGB"))
    H, W = img.shape[:2]
    rect = ast.literal_eval(boxes.loc[fname, "rect"])
    full = [0, 0, W, H]
    crop_pil = Image.fromarray(img[rect[1]:rect[3], rect[0]:rect[2]])
    res = {}
    for mdl, fn, frac, ms in (("BiomedCLIP", clip_map, 1.0, CLIP_MS), ("DINOv2", dino_map, 224 / 256, DINO_MS)):
        m_full, t_full = fn(Image.fromarray(img))
        m_crop, _ = fn(crop_pil)
        sq_full, sq_crop = seen_square(full, frac), seen_square(rect, frac)
        # geometric check with the model input itself
        inp = denorm(t_full, *ms)
        naive = cv2.resize(inp, (W, H), interpolation=cv2.INTER_AREA)                 # v1 mapping
        x0, y0, x1, y1 = [int(round(v)) for v in sq_full]
        fixed = cv2.resize(inp, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)    # fixed mapping
        ref = cv2.GaussianBlur(img, (0, 0), max(W, H) / 448)
        err_fixed = np.abs(fixed.astype(float) - ref[y0:y1, x0:x1].astype(float)).mean()
        err_naive_same = np.abs(naive[y0:y1, x0:x1].astype(float) - ref[y0:y1, x0:x1].astype(float)).mean()
        checks.append({"case": case, "model": mdl, "frame": fname, "W": W, "H": H,
                       "seen_square_full": [round(v) for v in sq_full], "fov_crop_rect": rect,
                       "seen_square_after_fov_crop": [round(v) for v in sq_crop],
                       "share_of_frame_seen_v1": (x1 - x0) * (y1 - y0) / (W * H),
                       "mae_fixed_mapping": err_fixed, "mae_v1_mapping": err_naive_same})
        res[mdl] = dict(v1=cv2.resize(m_full, (W, H), interpolation=cv2.INTER_CUBIC),
                        fixed=place(m_full, sq_full, H, W), crop=place(m_crop, sq_crop, H, W),
                        sq_full=sq_full, sq_crop=sq_crop)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))
    for r, mdl in enumerate(("BiomedCLIP", "DINOv2")):
        d = res[mdl]
        ax = axes[r, 0]
        ax.imshow(img)
        for sq, col, lab in ((d["sq_full"], "yellow", "model input, full frame (centre crop)"),
                             (rect, "lime", "FOV crop (step 0)"), (d["sq_crop"], "cyan", "model input after FOV crop")):
            ax.add_patch(plt.Rectangle((sq[0], sq[1]), sq[2] - sq[0], sq[3] - sq[1], fill=False, ec=col, lw=2, label=lab))
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"Case {case} ({label}) - original frame" if r == 0 else "DINOv2 input regions")
        for c, (key, title) in enumerate((("v1", "v1 drawing (map stretched over whole frame)\nGEOMETRICALLY WRONG"),
                                          ("fixed", "Fixed geometry, full-frame input"),
                                          ("crop", "Fixed geometry + FOV-cropped input")), start=1):
            ax = axes[r, c]
            ax.imshow(img)
            im = ax.imshow(d[key], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(f"{mdl}: {title}", fontsize=10)
        for ax in axes[r]:
            ax.axis("off")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.set_label("Attention rollout, relative (per-image min-max): red = most attended, blue = least.\n"
                 "Class-agnostic; uncoloured area = not seen by the model.", fontsize=9)
    fig.suptitle(f"Attention Rollout - case {case}. Where the frozen encoder attends; NOT evidence for Complete/Incomplete.",
                 fontsize=12)
    fig.savefig(FIG / f"rollout_{case}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

CK = pd.DataFrame(checks)
CK.to_csv(OUT / "rollout_mapping_check.csv", index=False)
txt = f"""
Attention Rollout regenerated for 2601, 2602, 2603 and 2605 (the same frames v1 used) with BiomedCLIP and DINOv2.
Three versions each: v1 drawing, fixed geometry, and fixed geometry + FOV crop. Every panel has a colorbar and legend.
Geometry check (rollout_mapping_check.csv): the 224x224 model input, mapped back onto the original frame, matches the
frame with mean abs error {CK.mae_fixed_mapping.mean():.1f} (fixed mapping) vs {CK.mae_v1_mapping.mean():.1f} (v1 mapping), on 0-255 intensity.
The v1 map covers the whole frame, but the model saw only {CK[CK.model=='BiomedCLIP'].share_of_frame_seen_v1.mean():.0%} (BiomedCLIP) /
{CK[CK.model=='DINOv2'].share_of_frame_seen_v1.mean():.0%} (DINOv2) of the frame area. The unseen margin is up to
{((CK.W - np.minimum(CK.W, CK.H) * np.where(CK.model == 'DINOv2', 224 / 256, 1.0)) / 2 / CK.W).max():.0%} of the frame width on each side.
v1 painted this margin with attention, stretched the map horizontally, and so displaced every spot away from the frame
centre (displacement grows toward the left/right edges). The geometry error is confirmed and fixed.
It does not explain everything. Visual review of the corrected maps for all four frames (not quantified):
2603 hot spots are mostly on glare; 2602 and 2605 (BiomedCLIP) are mostly on the lumen or fold edge; 2601 is on the lumen,
bubbles and image edges; DINOv2 shows small spots on glare, bubbles and lumen edges. In the uncropped input the
black octagon corners can also be hot. Maps change noticeably between full-frame and cropped input (e.g. 2601).
Conclusion: the geometric bug is confirmed; that it caused the "wrong location" feedback is not proven. Attention often
sits on non-mucosal structures, but no single cause explains it. Whether the maps look clinically sensible needs
the GI advisor's review. Both models still centre-crop the FOV rectangle to a square, so the left/right part of the
FOV is never seen (marked in the figures).
Figures: results/step0/figures/rollout_<case>.png
"""
log("Sub-step 7c - Attention Rollout: geometry fix, FOV crop, colorbar", txt)
print(txt)
print(CK[["case", "model", "share_of_frame_seen_v1", "mae_fixed_mapping", "mae_v1_mapping"]].round(3).to_string(index=False))
