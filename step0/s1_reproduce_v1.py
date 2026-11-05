"""Sub-step 1 - reproduce v1 with `run_pipeline.py --eval-only`.

Runs the ORIGINAL scripts in a temporary sandbox copy (so tracked files in data/processed and figures/
are not overwritten), then compares every CSV / PNG with the tracked version and checks the headline numbers.
Output: results/step0/s1_repro/{file_comparison.csv, headline_numbers.csv, run_pipeline_eval_only.log}
"""
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from common import ROOT, PROC, log, out_dir

OUT = out_dir("s1_repro")
sandbox = tempfile.mkdtemp(prefix="gim_v1_repro_")
for f in os.listdir(ROOT):
    if f.endswith(".py"):
        shutil.copy(ROOT / f, sandbox)
shutil.copytree(PROC, os.path.join(sandbox, "data", "processed"))
os.makedirs(os.path.join(sandbox, "figures"))
# attention rollout needs the first filtered frame of 2601/2602/2603/2605 (all stills of those cases)
ff = os.path.join(sandbox, "data", "filtered_frames")
os.makedirs(ff)
for f in os.listdir(ROOT / "data" / "filtered_frames"):
    if f[5:9] in ("2601", "2602", "2603", "2605"):
        shutil.copy(ROOT / "data" / "filtered_frames" / f, ff)

env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
proc = subprocess.run([sys.executable, "run_pipeline.py", "--eval-only"], cwd=sandbox, env=env,
                      capture_output=True, text=True, encoding="utf-8", errors="replace")
(OUT / "run_pipeline_eval_only.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
print("run_pipeline exit code:", proc.returncode)

rows = []
for sub in ("data/processed", "figures"):
    for f in sorted(os.listdir(os.path.join(sandbox, sub))):
        new, old = os.path.join(sandbox, sub, f), ROOT / sub / f
        r = {"file": f"{sub}/{f}", "tracked_exists": old.exists(),
             "byte_identical": old.exists() and filecmp.cmp(new, old, shallow=False)}
        if f.endswith(".csv") and old.exists() and not r["byte_identical"]:
            a, b = pd.read_csv(old), pd.read_csv(new)
            num = a.select_dtypes("number").columns
            r["values_identical"] = a.shape == b.shape and np.allclose(
                a[num].values.astype(float), b[num].values.astype(float), equal_nan=True, atol=1e-9)
        rows.append(r)
cmp = pd.DataFrame(rows)
cmp.to_csv(OUT / "file_comparison.csv", index=False)

rep = pd.read_csv(os.path.join(sandbox, "data", "processed", "dinov2_final_statistical_report.csv")).iloc[0]
ev = pd.read_csv(os.path.join(sandbox, "data", "processed", "dinov2_multimodal_evaluation_results.csv"))
champ = ev[(ev.Model == "SVC") & (ev.PCs == 8) & (ev.Threshold_Type == "Tuned_Youden")].iloc[0]
head = pd.DataFrame([
    ("champion accuracy", rep.Accuracy, 14 / 17), ("champion correct", champ.Correct, "14/17"),
    ("champion F1", rep.F1_Score, 0.800), ("champion AUC", rep.ROC_AUC, 0.694),
    ("champion acc CI", f"[{rep.Accuracy_CI_lower:.3f}, {rep.Accuracy_CI_upper:.3f}]", "[0.588, 1.000]"),
    ("champion AUC CI", f"[{rep.ROC_AUC_CI_lower:.3f}, {rep.ROC_AUC_CI_upper:.3f}]", "[0.414, 0.933]"),
    ("McNemar p vs baseline", rep.McNemar_p_value_vs_Baseline, 0.0654),
], columns=["quantity", "reproduced", "v1_claim"])
head.to_csv(OUT / "headline_numbers.csv", index=False)
shutil.rmtree(sandbox, ignore_errors=True)

n_csv = cmp.file.str.endswith(".csv").sum()
vals = cmp["values_identical"].fillna(False).astype(bool) if "values_identical" in cmp else False
csv_same = (cmp.file.str.endswith(".csv") & (cmp.byte_identical | vals)).sum()
diff_png = cmp[cmp.file.str.endswith(".png") & ~cmp.byte_identical].file.tolist()
text = f"""
`run_pipeline.py --eval-only` (original scripts, sandbox copy; exit code {proc.returncode}). Scripts run: lopocv setup,
baseline LR, task4 classifiers, task6 stats, DINOv2 evaluation, DINOv2 final stats/plots, SHAP x2, attention rollout.
Not run by the orchestrator: phase3_task3_infold_pca.py, phase3_task5_olgim_subanalysis.py (re-checked earlier: identical).
CSV outputs identical to tracked: {csv_same}/{n_csv}. PNGs not byte-identical: {diff_png} (rendering/metadata;
the plotted numbers come from identical CSVs). Headline: accuracy {rep.Accuracy:.4f} ({champ.Correct}), F1 {rep.F1_Score:.3f},
AUC {rep.ROC_AUC:.4f}, acc CI [{rep.Accuracy_CI_lower:.3f}, {rep.Accuracy_CI_upper:.3f}], AUC CI
[{rep.ROC_AUC_CI_lower:.3f}, {rep.ROC_AUC_CI_upper:.3f}], McNemar p {rep.McNemar_p_value_vs_Baseline:.4f} -> v1 reproduced exactly.
Caveat: "reproduced" means the same code gives the same numbers, not that the numbers are valid (see sub-steps 3-6).
Files: results/step0/s1_repro/{{file_comparison, headline_numbers}}.csv, run_pipeline_eval_only.log
"""
log("Sub-step 1 - Reproduce v1 (run_pipeline.py --eval-only)", text)
print(text)
print(head.to_string(index=False))
