# Step Zero log (Goal 3: Complete vs Incomplete, n=17)

Branch `v2-rescope`, started 2026-10-26. Code in `step0/`, outputs in `results/step0/`. Original phase*.py files untouched.
Seeds: SEED=42 for all models, PCA, CV shuffles (same as v1); permutation RNG `np.random.default_rng(2026)`;
bootstrap RNG `np.random.default_rng(12345)`. Models loaded offline from the local HF cache (`HF_HUB_OFFLINE=1`).

## Sub-step 0 - Data integrity

Inventory: 24 patients with media (metadata rows: 24; 2616 absent in both),
98 images (45 jpg, 53 png), 39 videos
(36 mp4, 3 mkv). Corrupt/undecodable files: ['2612_8_MTB_IM_Antrum_X1_M-NBI.jpg'].
Exact SHA256 duplicates: 0; near-duplicate image pairs
(dHash-256 <= 6 bits): 3
(cross-patient: 0). 2612_8 is a truncated JPEG with the same byte size
and header as 2612_9 (not a byte duplicate). .mkv: all 3 videos of 2619 open in OpenCV (30 fps, 1530 frames)
and 153 frames (3x51) are in the v1 embeddings. Video fps is cast with int() in v1 (13.73->13, 50.02->50).
Cohort: reproduced from GIMENDO_v2_Metadata.xlsx column 'Histological Subtype' (rules of
phase1_process_metadata.py:40-57) -> 17 patients, 9 Incomplete / 8 Complete; 2 of the 9 Incomplete are mixed
'Complete + Incomplete' (2602, 2611) mapped to Incomplete by substring; 2612 (mixed + LGD) excluded for dysplasia.
Excluded images: 58 of 98 stills never reached data/filtered_frames (only 40 used).
Reasons: {'not in hard-coded case list (fix_missing_frames.py': 55, 'blur filter': 3}. Root cause is a hard-coded case list in phase2_task3_fix_missing_frames.py:7, not the blur
filter: 45 stills of the 10 video patients (26 of them in cohort patients 2610/2619/2620/2622/2625) and 10 stills
of 2609/2612 (out of cohort) were never considered; the blur filter removed only 3 stills (2613) and 1 video frame.
Processor (filename token) vs label in the cohort: CV-190 = 2
Incomplete / 0 Complete (Fisher p=0.471).
png stills are Incomplete only as well (2613, 2614, 2615). These are primary confounders (see contingency_*.csv).
Lopocv pkl: label column removed in step0/phase3_task1_lopocv_setup_fixed.py (sub-step 1).
Files: results/step0/s0_integrity/{media_inventory, patient_inventory, excluded_images, duplicates, contingency_*}.csv

Addendum (manual check of the CSVs):
- Breakdown of the 58 unused stills: 26 from cohort video patients, 3 cohort stills dropped by blur (2613_2, 2613_3,
  2613_7; Laplacian var 80/98/80), and 29 from patients outside the cohort (13 IM-positive-unsubtyped, 9 from 2612
  (LGD), 1 from 2621 (negative), 6 controls).
- Laplacian variance depends on resolution: png stills (2879x1799) have median 73, jpg stills median 335. Had
  the blur filter been applied to all png stills, most of them would have been rejected; the threshold 100 is not
  calibrated across sources. Both 2613 WLE stills were dropped, so 2613 is represented by NBI/M-NBI only.
- 2613_1 and 2613_4 are near-identical (dHash distance 0, sizes 5,272,359 vs 5,272,727 bytes); both are used, so
  that view counts twice in the 2613 mean. 2617_10 vs 2617_5/2617_9 are similar views (distance 6), not copies.
- Processor vs label (cohort, filename token): 190 = 2 Incomplete / 0 Complete; X1 = 7 Incomplete / 8 Complete.
  Fisher p=0.47 only because n=2. Every CV-190 case is Incomplete and every png still is Incomplete, so a model that
  detects "190/png" gets 2-3 Incomplete patients right for free. Treated as a primary confounder in sub-step 6.

## Sub-step 1 - Reproduce v1 (run_pipeline.py --eval-only)

`run_pipeline.py --eval-only` (original scripts, sandbox copy; exit code 0). Scripts run: lopocv setup,
baseline LR, task4 classifiers, task6 stats, DINOv2 evaluation, DINOv2 final stats/plots, SHAP x2, attention rollout.
Not run by the orchestrator: phase3_task3_infold_pca.py, phase3_task5_olgim_subanalysis.py (re-checked earlier: identical).
CSV outputs identical to tracked: 16/16. PNGs not byte-identical: ['figures/shap_summary_dot_plot.png'] (rendering/metadata;
the plotted numbers come from identical CSVs). Headline: accuracy 0.8235 (14/17), F1 0.800,
AUC 0.6944, acc CI [0.588, 1.000], AUC CI
[0.414, 0.933], McNemar p 0.0654 -> v1 reproduced exactly.
Caveat: "reproduced" means the same code gives the same numbers, not that the numbers are valid (see sub-steps 3-6).
Files: results/step0/s1_repro/{file_comparison, headline_numbers}.csv, run_pipeline_eval_only.log

## Sub-step 1b - Fixed copies of three v1 scripts (audit items)

- `step0/phase3_task1_lopocv_setup_fixed.py`: `Subtype_Binary` (the label) removed from X (original :74). Output
  `results/step0/fixed_v1/lopocv_splits_n17_fixed.pkl`, where clinical_cols = 6 signs + Age.
- `step0/phase3_dinov2_multimodal_evaluation_fixed.py` and `step0/phase3_dinov2_final_stats_and_plots_fixed.py`:
  the silent in-sample fallback (original :112-117 and :124-128) is removed, so an inner-CV failure now raises.
  Hard-coded numbers are removed (bar chart and printout). Per-fold Youden thresholds are saved
  (`fixed_v1/dinov2_multimodal_fold_thresholds.csv`). All outputs go to `results/step0/fixed_v1/`.
- Both fixed scripts give results identical to the tracked CSVs, which confirms the fallback never fired.
  Per-fold thresholds: SVC-8PC 0.35-0.69. RF and GB hit the degenerate cut 0.01 (predict all Incomplete) in some
  folds, because inner probabilities on 12 patients are often all on one side.

## Sub-step 2 - Fair baselines (fully nested LOPOCV)

Protocol: see the docstring of step0/s2_fair_baselines.py. Inner selection: pooled inner-LOO AUC. Threshold:
Youden J on pooled inner scores. Nothing is fit on the held-out patient. 15 combinations (5 feature sets x 3 models),
all reported. No single "winner" is selected, because that would be another multiple comparison.
  - a_clinical / LR: acc 0.353 (6/17), bacc 0.354, F1 0.353, AUC 0.444
  - a_clinical / SVC: acc 0.471 (8/17), bacc 0.472, F1 0.471, AUC 0.611
  - a_clinical / RF: acc 0.647 (11/17), bacc 0.653, F1 0.625, AUC 0.583
  - b_biomedclip / LR: acc 0.588 (10/17), bacc 0.590, F1 0.588, AUC 0.708
  - b_biomedclip / SVC: acc 0.647 (11/17), bacc 0.646, F1 0.667, AUC 0.583
  - b_biomedclip / RF: acc 0.706 (12/17), bacc 0.708, F1 0.706, AUC 0.597
  - c_dinov2 / LR: acc 0.353 (6/17), bacc 0.375, F1 0.000, AUC 0.250
  - c_dinov2 / SVC: acc 0.412 (7/17), bacc 0.424, F1 0.286, AUC 0.264
  - c_dinov2 / RF: acc 0.471 (8/17), bacc 0.486, F1 0.308, AUC 0.201
  - d_both_visual / LR: acc 0.471 (8/17), bacc 0.479, F1 0.400, AUC 0.389
  - d_both_visual / SVC: acc 0.294 (5/17), bacc 0.306, F1 0.143, AUC 0.292
  - d_both_visual / RF: acc 0.412 (7/17), bacc 0.403, F1 0.500, AUC 0.417
  - e_visual+clinical / LR: acc 0.471 (8/17), bacc 0.479, F1 0.400, AUC 0.306
  - e_visual+clinical / SVC: acc 0.412 (7/17), bacc 0.417, F1 0.375, AUC 0.361
  - e_visual+clinical / RF: acc 0.529 (9/17), bacc 0.535, F1 0.500, AUC 0.424
Highest AUC: b_biomedclip/LR (0.708), a_clinical/SVC (0.611), b_biomedclip/RF (0.597).
Files: results/step0/s2_baselines/summary.csv (also acc/bacc at the default cut), per_patient_scores.csv
(score, threshold, prediction, chosen params and K for every patient x model).
Sanity checks (`s2_baselines/positive_control.txt`): with a synthetic informative feature (y + N(0, 0.3), raw AUC
0.99) added to the clinical set, the same nested pipeline gets AUC 0.83 (LR), 0.71 (SVC), 0.89 (RF). With a synthetic
shift injected into 50 BiomedCLIP dimensions, LR gets AUC 0.81. So the pipeline can find signal when signal exists.
The low numbers on the real labels are therefore not a code bug.
Interpretation: AUC values clearly below 0.5 (DINOv2: 0.20-0.26) are the known negative bias of leave-one-out CV
when there is no signal: removing one patient shifts the training class balance against that patient's class.
They should be read as "no detectable signal", not as "anti-predictive".
No nested configuration comes near the v1 82.4%. The best by AUC is BiomedCLIP-only LR (AUC 0.708, acc 0.588).
The best by accuracy is BiomedCLIP-only RF (0.706, AUC 0.597). Clinical-only LR: acc 0.353, AUC 0.444.
Adding DINOv2 or clinical features to BiomedCLIP lowers performance.

## Sub-step 3 - All 12 v1 configurations

All 12 v1 configurations rerun with the exact v1 protocol (fallback removed). They match the tracked
dinov2_multimodal_evaluation_results.csv: True.
Accuracy over the 12 configs: min 0.294, median 0.500, mean 0.495, max 0.824.
Only 1 of 12 reaches 0.824 (SVC, 8 PCs, Youden). The second best is 0.588. The same SVC-8PC at the default 0.5 cut
gets 0.588.
Balanced accuracy: median 0.490, max 0.833. AUC does not depend on the threshold:
RF [0.361, 0.306], SVC [0.653, 0.694],
GB [0.292, 0.278] (5 vs 8 PCs).
Per-fold Youden thresholds vary widely (fold_thresholds.csv). RF, GB and SVC-5PC fall to the degenerate cut 0.01
(predict all Incomplete) in 22 fold-config cases. SVC-8PC never does (0.35-0.69).
Selection bias: the reported 82.4% is the maximum of 12 correlated estimates on the same 17 patients. The model, PCs
and threshold rule were chosen after looking at LOPOCV results, so 82.4% is an optimistic estimate. Sub-step 4 measures
how often a max-of-12 this high appears under permuted labels.
Files: results/step0/s3_v1_configs/{configs_table, fold_thresholds, per_patient_probs}.csv

## Sub-step 5 - Leave-Pair-Out AUC and confidence intervals

LPO-AUC over all 72 (Complete, Incomplete) pairs, each trained on the remaining 15 patients (nested tuning inside).
CIs: Clopper-Pearson exact for accuracy; patient-level stratified bootstrap (2000 resamples) for AUC and balanced accuracy.
  - v1_champion_SVC8: acc 0.824 CP [0.566, 0.962], AUC 0.694 boot [0.431, 0.917], LPO-AUC 0.556
  - a_clinical/LR: acc 0.353 CP [0.142, 0.617], AUC 0.444 boot [0.194, 0.736], LPO-AUC 0.458
  - a_clinical/SVC: acc 0.471 CP [0.230, 0.722], AUC 0.611 boot [0.319, 0.889], LPO-AUC 0.694
  - a_clinical/RF: acc 0.647 CP [0.383, 0.858], AUC 0.583 boot [0.306, 0.875], LPO-AUC n/a
  - b_biomedclip/LR: acc 0.588 CP [0.329, 0.816], AUC 0.708 boot [0.444, 0.917], LPO-AUC 0.639
  - b_biomedclip/SVC: acc 0.647 CP [0.383, 0.858], AUC 0.583 boot [0.292, 0.875], LPO-AUC 0.597
  - b_biomedclip/RF: acc 0.706 CP [0.440, 0.897], AUC 0.597 boot [0.285, 0.875], LPO-AUC n/a
  - c_dinov2/LR: acc 0.353 CP [0.142, 0.617], AUC 0.250 boot [0.056, 0.514], LPO-AUC 0.347
  - c_dinov2/SVC: acc 0.412 CP [0.184, 0.671], AUC 0.264 boot [0.069, 0.542], LPO-AUC 0.250
  - c_dinov2/RF: acc 0.471 CP [0.230, 0.722], AUC 0.201 boot [0.000, 0.451], LPO-AUC n/a
  - d_both_visual/LR: acc 0.471 CP [0.230, 0.722], AUC 0.389 boot [0.139, 0.681], LPO-AUC 0.472
  - d_both_visual/SVC: acc 0.294 CP [0.103, 0.560], AUC 0.292 boot [0.069, 0.597], LPO-AUC 0.389
  - d_both_visual/RF: acc 0.412 CP [0.184, 0.671], AUC 0.417 boot [0.153, 0.722], LPO-AUC n/a
  - e_visual+clinical/LR: acc 0.471 CP [0.230, 0.722], AUC 0.306 boot [0.083, 0.569], LPO-AUC 0.375
  - e_visual+clinical/SVC: acc 0.412 CP [0.184, 0.671], AUC 0.361 boot [0.125, 0.639], LPO-AUC 0.403
  - e_visual+clinical/RF: acc 0.529 CP [0.278, 0.770], AUC 0.424 boot [0.167, 0.722], LPO-AUC n/a
v1 champion: LPO-AUC 0.556 vs pooled LOPOCV AUC 0.694. Clopper-Pearson accuracy CI
[0.566, 0.962] (v1 reported bootstrap [0.588, 1.000]). Even before correcting for
selection, the AUC CI includes 0.5.
Files: results/step0/s5_lpo_ci/{lpo_auc, lpo_pairs, ci_table}.csv
Note: v1 champion LPO-AUC = 0.556 (72 pairs). The pooled LOPOCV AUC of 0.694 relies partly on scores from 17
different models being comparable. When each pair is scored by the same model, ranking is close to chance.
The bootstrap AUC CI includes 0.5 for every model. Best LPO-AUC: clinical-only SVC 0.694, then BiomedCLIP LR 0.639.
DINOv2-containing sets have LPO-AUC 0.25-0.47.

## Sub-step 7c - Attention Rollout: geometry fix, FOV crop, colorbar

Attention Rollout regenerated for 2601, 2602, 2603 and 2605 (the same frames v1 used) with BiomedCLIP and DINOv2.
Three versions each: v1 drawing, fixed geometry, and fixed geometry + FOV crop. Every panel has a colorbar and legend.
Geometry check (rollout_mapping_check.csv): the 224x224 model input, mapped back onto the original frame, matches the
frame with mean abs error 4.8 (fixed mapping) vs 17.0 (v1 mapping), on 0-255 intensity.
The v1 map covers the whole frame, but the model saw only 74% (BiomedCLIP) /
57% (DINOv2) of the frame area. The unseen margin is up to
19% of the frame width on each side.
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

## Sub-step 7a - FOV detection, crop and embedding cache (run before 6 because 6e uses the raw-still embeddings)

FOV estimated per capture layout (resolution): 15 layouts covering 137 sources (39 videos, 98 stills); see fov_layouts.csv.
QC montages: results/step0/figures/fov_qc_*.png (green = FOV hull, red = crop).
Crop scale vs FOV bbox: median 0.88 (min 0.86).
Share of the v1 model input (center square) that lay outside the FOV, i.e. black border or on-screen text:
BiomedCLIP median 15.1% (videos 15.1%,
stills 0.0%, max 27.5%).
Embeddings cached: step0/embeddings_cropped/ (2074 items x 512 / 768) and step0/embeddings_raw/ (98 raw stills).
Check: raw-still embeddings recomputed here vs the v1 embeddings of the same 40 stills: cosine BiomedCLIP min 1.0000,
DINOv2 min 1.0000 (1.0 = identical preprocessing).

## Sub-step 6 - Confounder analysis and shortcut probe

Contingency (cohort n=17; contingency_tables.csv): processor 190 = 2 Incomplete / 0 Complete; png stills = 3 Incomplete / 0 Complete;
OSD text panel present = 5 Inc / 3 Comp,
absent (jpg stills) = 4 Inc / 5 Comp; stills-only = 7 Inc / 5 Comp, video = 2 Inc / 3 Comp.
v1 champion, stratified OOF (stratified_performance.csv): all: 14/17; source=stills only: 9/12; source=video frames only: 5/5; file_type=jpg stills: 7/9; file_type=video frames: 5/5; file_type=png stills: 2/3; processor=X1: 13/15; processor=190: 1/2; OSD_text_panel=absent: 7/9; OSD_text_panel=present: 7/8.
Retrained inside subgroups (subgroup_retrained.csv): see table; summary in STEP0_REPORT.
Shortcut probe (shortcut_probe.csv), grouped leave-one-patient-out AUC, raw -> cropped:
  - still_vs_video_frame/biomedclip: raw 0.871 -> cropped 0.858
  - still_vs_video_frame/dinov2: raw 0.831 -> cropped 0.797
  - png_vs_jpg_stills/biomedclip: raw 0.974 -> cropped 0.615
  - png_vs_jpg_stills/dinov2: raw 0.886 -> cropped 0.626
  - 190_vs_X1_stills/biomedclip: raw 0.355 -> cropped 0.389
  - 190_vs_X1_stills/dinov2: raw 0.359 -> cropped 0.105
  - OSD_vs_none/biomedclip: raw 0.977 -> cropped 0.824 (SUPERSEDED: mislabelled controls, see correction section)
  - OSD_vs_none/dinov2: raw 0.957 -> cropped 0.803 (SUPERSEDED: mislabelled controls, see correction section)
  - patient_source_video_vs_stills/b_biomedclip: AUC 1.000
  - patient_source_video_vs_stills/c_dinov2: AUC 1.000
  - patient_source_video_vs_stills/d_both_visual: AUC 1.000
  - patient_OSD_present/b_biomedclip: AUC 1.000
  - patient_OSD_present/c_dinov2: AUC 0.944
  - patient_OSD_present/d_both_visual: AUC 0.972
Interpretation (manual):
- The v1 champion is 5/5 on video patients but 9/12 on still-only patients. When the whole v1 protocol is
  re-trained without the video patients (image-only, n=12) it gets 4/12 (AUC 0.17). Without 2613/2614/2615
  (190 + png) it gets 7/14 (AUC 0.35). Restricted to X1 it gets 8/15. So the 14/17 depends on the patients whose
  acquisition differs (video screen-captures, 190, png), which is the signature of a shortcut, not of mucosal signal.
- Patient-level probe: the pooled v1 embeddings predict "video vs stills" with nested-LOPOCV AUC 1.00 (BiomedCLIP,
  DINOv2, both) and "OSD present" with AUC 0.94-1.00. The source of each patient is perfectly readable from the
  same features the classifier uses. Classes are unbalanced across sources (video 2I/3C, png 3I/0C, 190 2I/0C).
- Image-level probe (grouped by patient): png-vs-jpg drops from AUC 0.97/0.89 (raw) to 0.62/0.63 after the FOV crop,
  consistent with black border and OSD carrying most of that signal (not separately tested). Still-vs-video-frame stays high after cropping (0.86/0.80):
  resolution, compression and frame-grab quality differ even inside the FOV, so the FOV crop alone cannot remove the
  media-type shortcut. 190-vs-X1 is not estimable (only 2 patients with 190; AUC < 0.5 is noise).
- No nested model is consistently above chance in any subgroup. Best: BiomedCLIP LR, AUC 0.73 on X1-only and
  10/14 on drop-190/png. BiomedCLIP LR is also the most stable model across subsets.

## Sub-step 7b - Champion + visual-only models on FOV-cropped embeddings

Three feature versions (uncropped v1 / cropped, same frames as v1 (1098 items) / cropped, all media
of the 17 patients (1127 items)). Values are given in that order, as n_correct/17 and AUC:
  - b_biomedclip/LR (nested): correct 10/13/13 ; AUC 0.708/0.736/0.722
  - b_biomedclip/SVC (nested): correct 11/7/6 ; AUC 0.583/0.375/0.444
  - c_dinov2/LR (nested): correct 6/4/5 ; AUC 0.250/0.389/0.417
  - c_dinov2/SVC (nested): correct 7/4/8 ; AUC 0.264/0.167/0.361
  - d_both_visual/LR (nested): correct 8/8/7 ; AUC 0.389/0.403/0.354
  - d_both_visual/SVC (nested): correct 5/7/5 ; AUC 0.292/0.264/0.333
  - e_visual+clinical/LR (nested): correct 8/9/9 ; AUC 0.306/0.389/0.306
  - e_visual+clinical/SVC (nested): correct 7/4/11 ; AUC 0.361/0.264/0.333
  - v1_champion_SVC8 (v1 protocol): correct 14/12/13 ; AUC 0.694/0.708/0.708
Files: results/step0/s7_fov/cropped_vs_uncropped.csv, per_patient_cropped.csv

## Sub-step 4 - Whole-pipeline permutation test

N = 1000 label permutations (RNG default_rng(2026)). Each one reruns the full v1 protocol for all 12 configs (in-fold scaler/PCA,
inner-CV Youden threshold). Runtime 11039s on 10 processes.
  - champion SVC8-Youden accuracy (fixed config): observed 0.824, null mean 0.578, null 95th pct 0.882, p = 0.1169
  - champion SVC8 AUC (fixed config): observed 0.694, null mean 0.450, null 95th pct 0.944, p = 0.2727
  - max accuracy over 12 configs (selection-aware): observed 0.824, null mean 0.728, null 95th pct 0.941, p = 0.2707
  - max AUC over 6 cells (selection-aware): observed 0.694, null mean 0.651, null 95th pct 0.972, p = 0.3946
The fixed-config p-values answer whether that one config (if it had been pre-specified) beats chance. The
selection-aware p-values are the honest test of the reported result, because v1 picked the best of 12 after looking.
Files: results/step0/s4_permutation/{null_distribution, permutation_summary}.csv, results/step0/figures/permutation_null.png
Interpretation (manual): with n=17, the Youden-tuned pipeline reaches 15/17 or better on shuffled labels in
more than 5% of permutations (fixed config). The best of 12 reaches 16/17 at the 95th percentile. The observed
14/17 is therefore compatible with chance, both for the pre-specified config (p = 0.117) and for the selection
that v1 actually did (p = 0.271). The null mean accuracy of the fixed config is 0.578, not 0.5, because Youden tuning on
small inner folds systematically favours some cut. That is why raw accuracy against 0.5 is the wrong reference.

## Sub-step 7d - Stress test of the best cropped result (nested BiomedCLIP-LR, FOV-cropped)

Observed (n=17): 13/17, balanced acc 0.757, AUC 0.736; misclassified [2604, 2606, 2610, 2619].
Fixed-config whole-pipeline permutation test (N=1000): p(AUC) = 0.0589, p(accuracy) = 0.0380
(null mean AUC 0.427, 95th pct 0.737).
This config was picked after looking at 27 cells, so these p-values are not selection-corrected.
A Bonferroni-style bound over 27 cells is min(1, 27 x p).
Leave-Pair-Out AUC: 0.729.
Re-trained in subsets: all_n17: 13/17 (AUC 0.74); image_only_n12: 5/12 (AUC 0.51); X1_only_n15: 9/15 (AUC 0.71); drop_190_and_png_n14: 10/14 (AUC 0.59); jpg_stills_noOSD_n9: 6/9 (AUC 0.90).
Files: results/step0/s7_fov/cropped_biomedclip_lr_checks.csv, cropped_biomedclip_lr_null.csv,
results/step0/figures/permutation_null_cropped_biomedclip_lr.png

## Sub-step 8 - Final report and traceable figures

- `step0/s8_figures.py` builds `results/step0/figures/step0_performance_panel.png` (accuracy with Clopper-Pearson CI,
  AUC with bootstrap CI and LPO-AUC, and the selection-aware permutation null) and `results/step0/patient_level_table.csv`
  (per-patient labels, acquisition factors and scores, with no clinical-impact text). Every number comes from step-0 CSVs.
  These replace the untraceable `figures/figure2_comprehensive_performance_panel.png` and
  `data/processed/patient_level_correction_analysis.csv`, which should not be used in the paper.
- Report: `results/step0/STEP0_REPORT.md` (Persian, with a suggested limitations paragraph in English).

## Correction (2026-11-05) - text-panel probe re-run with labels per capture layout

Replaces the OSD_vs_none rows in shortcut_probe.csv, which mislabelled 46 items from controls 2623/2624 as "panel present".
Grouped leave-one-patient-out AUC:
  - all items / biomedclip: raw 0.990 -> cropped 0.743
  - all items / dinov2: raw 0.980 -> cropped 0.849
  - excluding controls 2623/2624 / biomedclip: raw 0.998 -> cropped 0.838
  - excluding controls 2623/2624 / dinov2: raw 0.985 -> cropped 0.823
  - stills only (no video frames) / biomedclip: raw 1.000 -> cropped 0.497
  - stills only (no video frames) / dinov2: raw 0.953 -> cropped 0.559
Panel and patient are not independent here, so these AUCs still mix panel with other acquisition differences.
File: results/step0/s6_confounders/shortcut_probe_osd_corrected.csv

## Corrections to earlier entries (2026-11-05)

1. Attention rollout: the claim "mostly on specular highlights" came from viewing 2603 only. It is replaced in the
   sub-step 7c entry and in step0/s7c_attention_rollout.py after viewing all four figures (still visual, not quantified).
2. On-screen panel: not present in every video and png still. Controls 2623/2624 have no side panel. The image-level
   probe is re-run in the section above. The cohort-level factor and patient-level probe are unaffected.
3. New finding: the FOV crop does not remove all on-screen text. All 4 model inputs of 2615 (CV-190, Incomplete) still contain
   readable text ("...rary patient ID"), checked by viewing the centre-square inputs. 2614 and the three X1 samples viewed
   (2613_1, 2617_1, 2622_4) showed none. Not every item was viewed. Cropped results for 2615 are not text-free.
4. step0_audit report: tracked counts corrected (22 in data/processed, 10 in figures); 5, not 6, cohort patients lack a
   processor in metadata.
5. Suggested limitations paragraph in STEP0_REPORT.md reworded (rollout, on-screen text, "no model exceeded chance",
   and which model fell to chance).
Numeric results of the other sub-steps were not re-run or changed.
