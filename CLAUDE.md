# GIM project - read this first

## Project

Current scope: ONLY Goal 1 (IM vs non-IM from endoscopy). Goals 2-4 (atrophy, Complete vs Incomplete, endoscopic signs) are on hold.
This repository is public. Internal documents, names, contacts and data-access notes stay local (CLAUDE.local.md, docs/) and are never committed.

## Goal 1 plan

- Train on EndoWLI-NBI using proxy tasks. It has no normal class and IM&CG is not pure IM, so it cannot give a valid IM vs non-IM number.
- Primary dataset "A2-NBI" (positive = IM&CG; vs LGIN+HGIN+EGC): years 2018-2021; exact within-patient duplicates removed; images the colour check calls white-light removed (p_wli < 0.3618, threshold from modality labels only); patients with at least 3 images left; at most 10 images per patient (results/goal1/capped_list.csv, drawn once, seed 42). The filter keeps non-magnified NBI and dye images.
- Sensitivity: S1 = same years, no modality filter. S2 = S1 without LGIN. S3 = A2-NBI without LGIN. Frozen settings: results/goal1/config_B.json (SHA-256 in config_B.sha256).
- Secondary task idea: option B, 4-class (EGC/HGIN/LGIN/IM&CG); not run yet.
- Name the proxy tasks honestly in all reports. Never call them "IM vs non-IM".
- Later additions: (1) Yang IM/GA/Normal dataset, (2) ~10 patient videos from clinical collaborators (5 IM, 5 non-IM), (3) GIM-ENDO as a descriptive sanity check only.
- Status: Step B stopped under the stop rule (AUC 0.986-0.9996 is suspiciously high; not a finding). Code bug and cross-patient duplicates look ruled out. Exploratory artefact test: tissue-free features reach AUC 0.67-0.71 (partial evidence, does not explain 0.99 alone); tissue signal vs acquisition artefact is still unresolved. Full permutation test not run. See results/goal1/STEP_B.md and results/goal1/ARTEFACT_TEST.md.
- Next: Goal 1 NBI development is paused at the Step B / artefact-test result, and the 200-permutation test is not being run now. Work continues with inspecting new clinical data (Step D) while waiting for the Yang dataset.

## Datasets

**EndoWLI-NBI (external, NOT in repo):** D:\dataset_gim\EndoWLI-NBI\EndoWLI-NBI
- Folder structure: category/year/one patient (EGC, HGIN, IM&CG, LGIN), plus metadata.csv and resolution_count.csv. 11,971 images, 868 patients (verified count).
- Development data: EndoWLI-NBI dataset (Peking Union Medical College Hospital), Figshare, CC BY 4.0; images are not redistributed here.
- Never copy images or embeddings into the repo or commit them, including contact sheets or crops. File names, labels and statistics may be committed.
- Labels are patient-level (weak). IM&CG = IM merged with chronic gastritis.
- The authors already auto-cropped black borders and on-screen text: do NOT apply our own FOV crop. Because their crop is automatic, resolution varies with content, so resolution/aspect ratio must be included as a shortcut baseline.
- LGIN has ~33 images/patient and IM&CG ~13/patient: cap at most 10 random images per patient (fixed seed) and aggregate to patient level.
- The paper contradicts itself on modality (intro: WLI+NBI, magnified and non-magnified; Methods: all ME-NBI), so the modality mix per class must be measured.
- The paper says one scope (Olympus GIF-H260Z) and one processor (CV-290), but metadata.csv lists 67 patients with other device names and 195 with none. In 2018-2021 only 7 have other names and 107 have none (results/goal1/STEP_A.md).
- LGIN is mostly white-light and non-magnified; other classes are mostly magnified NBI (Step A, non-expert visual review). After filtering, LGIN still has more dye, forceps and white-light-looking images.
- Reference code (not part of repo): D:\dataset_gim\CAD-MENBI.

**Other data (Yang dataset, clinical videos, independent test data):** not in this repo. Independent test data are never used for training or tuning.

**GIM-ENDO v2 (Zenodo; license CC BY-NC 4.0 per paper, article page says CC-BY - conflict):**
- 24 patients, 98 images + 39 videos in the repo root (gitignored). No patient 2616. Histology in metadata: 21 IM positive, 1 negative (2621), 2 controls without histology (2623, 2624).
- Descriptive use only: 3 negatives, label tied to data source. No performance claims.
- Raw images are still in git history (commit e534b00, ~235 MB). Never commit new data.
- Filenames encode patient, index, signs, location, processor (X1 / 190), modality (WLE/NBI/M-NBI).

## Repo layout

- phase1..phase4 scripts: frozen v1 pilot (Goal 3), tag v1-pilot. Do not modify them.
- step0/ and results/step0/: Step Zero audit of v1 (done). Reports: results/step0/STEP0_REPORT.md, results/step0/LOG.md.
- goal1/: new Goal 1 code. results/goal1/: Goal 1 outputs. D:\gim_outputs: large files (embeddings, caches, contact sheets).
- Work on branch v2-rescope.

## What Step Zero found (Goal 3, n=17; on hold)

- v1 result 14/17 (82.4%) is the best of 12 configs and is compatible with chance: selection-aware permutation p=0.271, Leave-Pair-Out AUC 0.556. Do not reuse it as a performance claim.
- It depends on acquisition differences: 5/5 on video patients, 4/12 when re-trained on still-only patients. CV-190 and png stills occur only in Incomplete patients.
- Do not reuse figure2_comprehensive_performance_panel.png or patient_level_correction_analysis.csv; use results/step0/figures/step0_performance_panel.png and results/step0/patient_level_table.csv.
- Attention Rollout: the center-crop vs full-frame misalignment in v1 is a confirmed bug, fixed in step0/s7c_attention_rollout.py. Whether it caused the GI advisor's feedback is not proven.

## Known open issues

- Low priority: FOV crops of GIM-ENDO patient 2615 (CV-190) still contain readable on-screen text; cropped embeddings were not recomputed. Relevant if GIM-ENDO cropped embeddings are reused.
- Bugs in original phase*.py files stay in place by rule; see results/step0/STEP0_REPORT.md.

## Rules

- Patient-level splits always. Zero leakage: scaler/PCA/feature selection/hyperparameters/threshold fit inside the training fold only.
- Report patient counts with CIs. No clinical-performance claims. No "optical biopsy" or "unequivocal" wording.
- No git commit/push unless the user explicitly asks. Never run git checkout/reset/clean without git status first.
- Fix seeds and state them. Cache embeddings to disk. Do not download models/data other than those already used (BiomedCLIP, DINOv2) without asking.
- Keep chat output short: write details to results/ and show short summaries (<= 30 lines) with file names.
- If something is ambiguous, stop and ask instead of assuming.

## Hardware

i7-12650H, 16 GB RAM, RTX 3060 Laptop (6 GB VRAM): batch size <= 8, fp16 if safe. PyTorch in .venv is CPU-only, so embeddings run on CPU fp32 unless a CUDA environment is set up. Disk: C: is nearly full (~24 GB free), D: has ~559 GB free. Laptop is left running unattended; the session may be followed remotely from a phone.
