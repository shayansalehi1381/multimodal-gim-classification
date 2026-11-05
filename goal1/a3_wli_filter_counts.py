"""Goal 1, Step A3 - how many patients remain in A2 (2018-2021) if images the colour proxy calls white-light are dropped.
Input: D:/gim_outputs/goal1/image_stats.csv, image_modality_proxy.csv. Output: results/goal1/wli_filter_counts.csv
"""
import pandas as pd

D = "D:/gim_outputs/goal1/"
s = pd.read_csv(D + "image_stats.csv")[["label", "patient", "file", "year", "sha256"]]
p = pd.read_csv(D + "image_modality_proxy.csv")
s = s.merge(p[["patient", "file", "p_wli"]], on=["patient", "file"])
s = s[s.year >= 2018].drop_duplicates(["patient", "sha256"])
keep = s[s.p_wli < 0.5]
rows = []
for lab, g in s.groupby("label"):
    k = keep[keep.label == lab].groupby("patient").size()
    rows.append({"label": lab, "patients_2018_2021": g.patient.nunique(), "images_after_dedupe": len(g),
                 "images_non_wli": int(k.sum()), "patients_ge1_non_wli": int((k >= 1).sum()), "patients_ge3_non_wli": int((k >= 3).sum())})
out = pd.DataFrame(rows)
out.to_csv("../results/goal1/wli_filter_counts.csv", index=False)
print(out.to_string(index=False))
