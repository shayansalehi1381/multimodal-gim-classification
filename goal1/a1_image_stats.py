"""Goal 1, Step A1 - per-image statistics for EndoWLI-NBI (inspection only, no training).

Reads every JPEG once. Outputs to D:\\gim_outputs\\goal1 (dataset license: no redistribution):
  image_stats.csv   label, year, patient, size, JPEG size, JPEG tables, colour, sharpness, corner darkness, hashes
  sheets/sheet_<class>_<n>.jpg + sheet_index.csv   100 random images per class for visual review (seed 42,
                    at most one image per patient while patients remain)
"""
import hashlib
import os

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageDraw

DATA = r"D:\dataset_gim\EndoWLI-NBI\EndoWLI-NBI"
OUT = r"D:\gim_outputs\goal1"
SEED = 42
N_PER_CLASS, COLS, ROWS, TILE = 100, 10, 5, (200, 176)


def one(path, label, year, pid):
    raw = open(path, "rb").read()
    r = {"label": label, "year": int(year), "patient": pid, "file": os.path.basename(path),
         "path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with Image.open(path) as im:
        r["width"], r["height"] = im.size
        r["jpeg_tables"] = hashlib.md5(repr(sorted(im.quantization.items())).encode()).hexdigest()[:10]
        r["jpeg_q0_sum"] = int(sum(im.quantization[0]))
        r["subsampling"] = im.info.get("jfif_version", "")
        rgb = np.array(im.convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.float32)
    r["bytes_per_pixel"] = r["bytes"] / (r["width"] * r["height"])
    r["aspect"] = r["width"] / r["height"]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    mucosa = (gray > 30) & (gray < 235)                 # ignore dark lumen / borders and glare for colour
    m = mucosa if mucosa.sum() > 500 else np.ones_like(mucosa)
    r.update(mean_r=R[m].mean(), mean_g=G[m].mean(), mean_b=B[m].mean(),
             r_over_g=R[m].mean() / max(G[m].mean(), 1), g_over_b=G[m].mean() / max(B[m].mean(), 1),
             r_over_b=R[m].mean() / max(B[m].mean(), 1), sat=hsv[..., 1][m].mean(), val=hsv[..., 2][m].mean(),
             std_gray=gray[m].std(), dark_frac=float((gray < 20).mean()), glare_frac=float((gray > 240).mean()),
             lap_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
             corner_dark=float(np.mean([(gray[:20, :20] < 20).mean(), (gray[:20, -20:] < 20).mean(),
                                        (gray[-20:, :20] < 20).mean(), (gray[-20:, -20:] < 20).mean()])))
    # fine-texture energy: share of spectrum in high frequencies (magnification proxy)
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) - gray.mean())))
    yy, xx = np.indices(f.shape)
    rad = np.hypot(yy - 128, xx - 128)
    r["hf_ratio"] = float(f[(rad > 30) & (rad < 100)].sum() / (f[rad < 100].sum() + 1e-6))
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    r["dhash"] = "".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten())
    return r


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    tasks = []
    for label in ("EGC", "HGIN", "IM&CG", "LGIN"):
        for yr in sorted(os.listdir(os.path.join(DATA, label))):
            for pid in sorted(os.listdir(os.path.join(DATA, label, yr))):
                for f in sorted(os.listdir(os.path.join(DATA, label, yr, pid))):
                    tasks.append((os.path.join(DATA, label, yr, pid, f), label, yr, pid))
    print("images:", len(tasks), flush=True)
    rows = Parallel(n_jobs=8, verbose=1)(delayed(one)(*t) for t in tasks)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "image_stats.csv"), index=False)

    # contact sheets: at most one image per patient, patients in random order
    idx_rows = []
    rng = np.random.default_rng(SEED)
    for label, g in df.groupby("label"):
        pts = rng.permutation(g.patient.unique())
        pick = [g[g.patient == p].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0] for p in pts[:N_PER_CLASS]]
        for s in range(0, len(pick), COLS * ROWS):
            sheet = Image.new("RGB", (COLS * TILE[0], ROWS * TILE[1]))
            for i, r in enumerate(pick[s:s + COLS * ROWS]):
                im = Image.open(r.path).convert("RGB").resize(TILE)
                ImageDraw.Draw(im).text((3, 2), str(s + i), fill=(255, 255, 0))
                sheet.paste(im, ((i % COLS) * TILE[0], (i // COLS) * TILE[1]))
                idx_rows.append({"label": label, "tile": s + i, "sheet": s // (COLS * ROWS), "patient": r.patient, "year": r.year,
                                 "file": r.file, "path": r.path})
            sheet.save(os.path.join(OUT, "sheets", f"sheet_{label.replace('&', '')}_{s // (COLS * ROWS)}.jpg"), quality=85)
    pd.DataFrame(idx_rows).to_csv(os.path.join(OUT, "sheets", "sheet_index.csv"), index=False)
    print("done", df.shape, "sheets", len(idx_rows), flush=True)
