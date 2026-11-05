"""Goal 1, Step B-0 - residual forceps / overlays / white-light in A2-NBI (the primary capped images).

Builds review sheets (100 random capped images per class, seed 42, more than one per patient where a class has fewer
than 100 patients) and an automatic blue-pixel share (forceps sheath, blue markers and dye all raise it).
Outputs: D:/gim_outputs/goal1/sheets_primary/*.jpg + sheet_index.csv, results/goal1/residual_blue_B0.csv
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageDraw

OUT = "../results/goal1/"
SH = "D:/gim_outputs/goal1/sheets_primary/"
os.makedirs(SH, exist_ok=True)
COLS, ROWS, TILE = 10, 5, (200, 176)
cl = pd.read_csv(OUT + "capped_list.csv")
d = cl[cl.dataset == "A2-NBI"].reset_index(drop=True)


def blue(path):
    a = np.array(Image.open(path).convert("RGB").resize((384, 384))).astype(int)
    return float(((a[..., 2] > 120) & (a[..., 2] - a[..., 0] > 60) & (a[..., 2] - a[..., 1] > 30)).mean())


d["blue_frac"] = Parallel(n_jobs=4)(delayed(blue)(p) for p in d.path)
d["blue_flag"] = d.blue_frac > 0.002
t = d.groupby("label").agg(images=("file", "size"), patients=("patient", "nunique"), images_blue_flag_pct=("blue_flag", lambda x: x.mean() * 100))
t["patients_with_flagged_image_pct"] = d.groupby("patient").agg(l=("label", "first"), f=("blue_flag", "max")).groupby("l").f.mean() * 100
t.round(2).to_csv(OUT + "residual_blue_B0.csv")

rows = []
for label, g in d.groupby("label"):
    pick = g.sample(100, random_state=42) if len(g) >= 100 else g
    pick = pick.reset_index(drop=True)
    for s in range(0, len(pick), COLS * ROWS):
        sheet = Image.new("RGB", (COLS * TILE[0], ROWS * TILE[1]))
        for i, r in enumerate(pick.iloc[s:s + COLS * ROWS].itertuples()):
            im = Image.open(r.path).convert("RGB").resize(TILE)
            ImageDraw.Draw(im).text((3, 2), str(s + i), fill=(255, 255, 0))
            sheet.paste(im, ((i % COLS) * TILE[0], (i // COLS) * TILE[1]))
            rows.append({"label": label, "tile": s + i, "patient": r.patient, "file": r.file, "blue_frac": r.blue_frac})
        sheet.save(f"{SH}sheet_{label.replace('&', '')}_{s // (COLS * ROWS)}.jpg", quality=85)
pd.DataFrame(rows).to_csv(SH + "sheet_index.csv", index=False)
print(t.round(2).to_string())
