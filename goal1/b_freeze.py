"""Goal 1, Step B - write the frozen config BEFORE any shortcut or model evaluation.
Usage: python b_freeze.py <solver> <max_iter> <cpu_workers>
Refuses to overwrite an existing config. Stores its SHA-256 in results/goal1/config_B.sha256.
"""
import hashlib
import json
import sys

from b_common import OUT

solver, max_iter, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p, ps = OUT / "config_B.json", OUT / "config_B.sha256"
if p.exists():
    raise SystemExit("config_B.json already exists; it is frozen")
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
filt = json.load(open(OUT / "modality_filter.json"))
cfg = {
    "step": "Goal 1 Step B; proxy tasks only, never IM vs non-IM",
    "frozen_on": "2026-11-05",
    "positive_class": "IM&CG",
    "datasets": {
        "A2-NBI": "PRIMARY. 2018-2021; exact within-patient duplicates removed; images with p_wli < threshold; patients with >= 3 such images; at most 10 images per patient. IM&CG vs LGIN+HGIN+EGC.",
        "S1": "Sensitivity. 2018-2021; duplicates removed; no modality filter; at most 10 images per patient. IM&CG vs LGIN+HGIN+EGC.",
        "S2": "Sensitivity. S1 without LGIN (same drawn images). IM&CG vs HGIN+EGC.",
        "S3": "Sensitivity. A2-NBI without LGIN (same drawn images). IM&CG vs HGIN+EGC. Added before any evaluation after review of residual dye/forceps/white-light in LGIN.",
    },
    "modality_filter": {"description": "colour LR, WLI vs non-WLI, threshold = Youden J on out-of-fold predictions against visual modality labels only. Removes white-light-looking images; it does not remove non-magnified NBI or dye.",
                        "threshold": filt["threshold"], "oof_auc": filt["oof_auc"], "labels": "results/goal1/visual_labels.csv (Claude, non-expert, not blind)"},
    "capped_list": {"file": "results/goal1/capped_list.csv", "sha256": sha(OUT / "capped_list.csv"), "cap_per_patient": 10, "drawn": "once; default_rng(42)"},
    "preprocessing": "resize the FULL image to 224x224 (bicubic), no center-crop, no custom crop; model-specific normalisation; embeddings L2-normalised per image",
    "embeddings": {"biomedclip": "open_clip hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, encode_image, 512-d",
                   "dinov2": "transformers facebook/dinov2-base, CLS of last_hidden_state, 768-d",
                   "device": "CPU fp32 batch 8 (user decision 2026-11-05: .venv torch has no CUDA)"},
    "runs": [{"dataset": "A2-NBI", "embedding": "biomedclip", "role": "primary"},
             {"dataset": "A2-NBI", "embedding": "dinov2", "role": "secondary"},
             {"dataset": "S1", "embedding": "biomedclip", "role": "sensitivity"},
             {"dataset": "S2", "embedding": "biomedclip", "role": "sensitivity"},
             {"dataset": "S3", "embedding": "biomedclip", "role": "sensitivity"}],
    "logistic_regression": {"unit": "image rows; labels from patient", "standardisation": "StandardScaler fit on the training rows of each fold",
                            "class_weight": "balanced", "C_log10_start": -3, "C_log10_stop": 2, "C_n": 11, "solver": solver, "max_iter": max_iter, "tol": 1e-4,
                            "C_selection": "inner 5-fold StratifiedGroupKFold on patient-level AUC (mean image probability); ties -> smallest C",
                            "threshold": "Youden J on inner-CV patient-level out-of-fold probabilities for the selected C"},
    "cv": {"outer_folds": 5, "repeats": 5, "inner_folds": 5, "splitter": "StratifiedGroupKFold on image rows, groups = patient"},
    "seeds": {"cap": 42, "model": 42, "outer_base": 42, "inner_base": 1000, "bootstrap": 12345, "permutation": 2026},
    "metrics": {"primary": "patient-level AUC, pooled over patients per repeat, mean over repeats", "also": "balanced accuracy at inner-CV threshold; mean per-fold AUC",
                "subgroups": "IM&CG vs each negative class; per year (descriptive, no retraining)"},
    "bootstrap": {"B": 2000, "unit": "patients, stratified by class; each draw evaluated on all repeats"},
    "paired_comparison": "bootstrap difference against the best shortcut of the same dataset on the same folds; best = highest AUC over groups and LR/RF, which favours the shortcut",
    "shortcuts": {"groups": ["colour_stats", "file_size", "resolution_aspect", "images_per_patient", "year", "ALL_five_groups"], "unit": "patient-level summaries of capped images",
                  "models": "LR with the same protocol; RF untuned", "rf": {"n_estimators": 200, "min_samples_leaf": 3}},
    "permutation": {"N": 200, "what": "shuffle patient labels and repeat the WHOLE pipeline (all repeats, outer folds, inner CV)",
                    "p": "(1 + #{null >= observed}) / (N + 1)", "runs_required": [["A2-NBI", "biomedclip"], ["A2-NBI", "dinov2"]]},
    "stop_conditions": {"shortcut_auc_above": 0.90, "others": "label/modality leakage, NaNs, class empty in a fold"},
    "untouched": "D:/dataset_gim/foroutan_new is an independent test set and is not used",
    "compute": {"cpu_workers": workers},
}
raw = json.dumps(cfg, indent=1).encode()
p.write_bytes(raw)
ps.write_text(hashlib.sha256(raw).hexdigest())
print("config frozen", hashlib.sha256(raw).hexdigest())
