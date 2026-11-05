"""Goal 1 - EXPLORATORY artefact test, step 1: per-image features from A2-NBI capped images at native resolution.
Not part of the frozen protocol. No file in the frozen config, capped list or embeddings changes.

Tissue-free:
  corner_value    : colour inside a fixed corner template (pixels black in >= 90% of images, eroded; no labels used)
  corner_noise    : pixel spread and exact-value shares inside that template
  corner_geometry : black run along each corner diagonal (minimum and median over the four corners)
  dark_lumen      : colour and noise inside eroded very dark regions that are not corners (only images with >= 500 pixels)
  file_header     : JPEG structure before the scan data (hash, length, progressive/optimised flags, JFIF info)
Low-level, NOT tissue-free (image content still affects these):
  histogram_comb  : gaps/spikes in 256-bin channel histograms (post-processing)
  blockiness      : gradient at 8-px block boundaries vs inside blocks
  global_noise    : Immerkaer noise estimate per channel
Output: D:/gim_outputs/goal1/artefact_image_features.csv
"""
import hashlib

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

OUT = "../results/goal1/"
ERODE, SIZE = 9, (1158, 1008)
IMK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def header_info(raw):
    i, n = 2, len(raw)
    markers = []
    while i < n - 3 and raw[i] == 0xFF:
        m = raw[i + 1]
        if m == 0xDA:
            break
        L = int.from_bytes(raw[i + 2:i + 4], "big")
        markers.append(m)
        i += 2 + L
    return {"hdr_hash": hashlib.md5(raw[:i]).hexdigest()[:10], "hdr_len": i, "hdr_markers": ".".join(f"{m:02x}" for m in markers),
            "progressive": int(0xC2 in markers), "n_dht": markers.count(0xC4), "n_app": sum(0xE0 <= m <= 0xEF for m in markers)}


def build_template(paths):
    """Pixels that are black (<=6) in at least 90% of 1158x1008 images, eroded so they stay away from the FOV edge.
    Uses no labels. Tissue-free by construction."""
    n, acc = 0, np.zeros(SIZE[::-1], np.float32)
    for p in paths:
        g = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g.shape == SIZE[::-1]:
            acc += g <= 6
            n += 1
    tpl = (acc / n >= 0.90).astype(np.uint8)
    return cv2.erode(tpl, np.ones((ERODE, ERODE), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1), n


def corners(rgb, gray, tpl):
    h, w = gray.shape
    r = {}
    t = tpl if (h, w) == tpl.shape else cv2.resize(tpl, (w, h), interpolation=cv2.INTER_NEAREST)
    t = t.astype(bool)
    px = rgb[t]
    for i, c in enumerate("rgb"):
        r[f"corner_val_{c}"], r[f"corner_std_{c}"] = float(px[:, i].mean()), float(px[:, i].std())
    r["corner_frac_zero"], r["corner_frac_le2"] = float((px == 0).all(1).mean()), float((px <= 2).all(1).mean())
    r["corner_max"] = float(px.max())
    # geometry: black run along each corner diagonal, as a share of the short side. Dark tissue can only lengthen a run,
    # so the minimum and median over the four corners are the robust measures.
    n = min(h, w) // 2
    idx = np.arange(n)
    runs = []
    for ys, xs in ((idx, idx), (idx, w - 1 - idx), (h - 1 - idx, idx), (h - 1 - idx, w - 1 - idx)):
        above = np.flatnonzero(gray[ys, xs] > 25)
        runs.append((above[0] if len(above) else n) / min(h, w))
    r["corner_diag_min"], r["corner_diag_median"] = float(np.min(runs)), float(np.median(runs))
    return r, t


def one(path, tpl):
    raw = open(path, "rb").read()
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r = {"path": path, **header_info(raw)}
    c, mask = corners(rgb, gray, tpl)
    r.update(c)
    edge = cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)   # keep clear of the corners
    dark = ((gray < 25) & ~edge).astype(np.uint8)
    er = cv2.erode(dark, np.ones((ERODE, ERODE), np.uint8)).astype(bool)
    r["lumen_pixels"] = int(er.sum())
    if er.sum() >= 500:
        px = rgb[er].astype(np.float32)
        hp = np.abs(cv2.filter2D(gray.astype(np.float32), -1, IMK))[er]
        for i, ch in enumerate("rgb"):
            r[f"lumen_mean_{ch}"], r[f"lumen_std_{ch}"] = px[:, i].mean(), px[:, i].std()
        r["lumen_hp"] = float(hp.mean())
    tissue = ~edge
    for i, ch in enumerate("rgb"):
        hist = np.bincount(rgb[..., i][tissue].ravel(), minlength=256).astype(np.float64)[5:250]
        r[f"comb_{ch}"] = float((np.abs(hist[1:-1] - 0.5 * (hist[:-2] + hist[2:])) / (hist[1:-1] + 1)).mean())
        r[f"empty_{ch}"] = float((hist == 0).mean())
        r[f"noise_{ch}"] = float(np.abs(cv2.filter2D(rgb[..., i].astype(np.float32), -1, IMK))[tissue].mean())
    g = gray.astype(np.float32)
    dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    bx, by = (np.arange(dx.shape[1]) % 8) == 7, (np.arange(dy.shape[0]) % 8) == 7
    r["blockiness"] = float((dx[:, bx].mean() + dy[by, :].mean()) / (dx[:, ~bx].mean() + dy[~by, :].mean() + 1e-6))
    return r


if __name__ == "__main__":
    cl = pd.read_csv(OUT + "capped_list.csv")
    paths = sorted(cl[cl.dataset == "A2-NBI"].path.unique())
    tpl, n_tpl = build_template(paths[::5])
    print("template from", n_tpl, "images; pixels", int(tpl.sum()), "share", round(float(tpl.mean()), 4), flush=True)
    cv2.imencode(".png", tpl * 255)[1].tofile("D:/gim_outputs/goal1/corner_template.png")
    rows = Parallel(n_jobs=8, verbose=1)(delayed(one)(p, tpl) for p in paths)
    F = pd.DataFrame(rows)
    F.to_csv("D:/gim_outputs/goal1/artefact_image_features.csv", index=False)
    d = cl[cl.dataset == "A2-NBI"].merge(F, on="path")
    print(F.shape, "unique header hashes", F.hdr_hash.nunique(), "marker patterns", F.hdr_markers.nunique(), "progressive", F.progressive.mean())
    print(d.groupby("label")[["corner_val_r", "corner_std_r", "corner_max", "corner_frac_zero", "corner_diag_min", "corner_diag_median", "lumen_pixels", "blockiness", "noise_r", "comb_r", "empty_r"]].median().round(3).T.to_string())
    print("images with lumen data:", (F.lumen_pixels >= 500).mean().round(3), "with corner stats:", F.corner_val_r.notna().mean().round(3))
    print("NaN share:", F.select_dtypes("number").isna().mean().sort_values(ascending=False).head(10).round(3).to_dict())
