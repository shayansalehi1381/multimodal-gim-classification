# گزارش Step Zero: هدف ۳ (Complete در برابر Incomplete، n=17)

branch: `v2-rescope` · تاریخ: 2026-10-26 · کد: `step0/` · خروجی‌ها: `results/step0/` · لاگ کامل: `results/step0/LOG.md`
هیچ commit یا push انجام نشد. فایل‌های اصلی `phase*.py` دست‌نخورده‌اند.
**اصلاحیه 2026-11-05:** توضیح rollout، ردیف‌های OSD و probe، و متن Limitations اصلاح شد (جزئیات در انتهای `LOG.md`). بقیه‌ی اعداد تغییری نکرده‌اند.
**Seeds:** `SEED=42` برای همه‌ی مدل‌ها، PCA و CV؛ `default_rng(2026)` برای permutation؛ `default_rng(12345)` برای bootstrap. مدل‌ها offline از cache لود شدند.

> این یک pilot امکان‌سنجی با ۱۷ بیمار است. هیچ عددی در این گزارش ادعای عملکرد بالینی نیست.

---

## ۱. جدول همه‌ی اعداد

**الف) مدل champion نسخه‌ی v1 و تحلیل‌هایش**

| کمیت | مقدار | فایل |
|---|---|---|
| بازتولید v1 (`run_pipeline.py --eval-only`) | 14/17، AUC 0.694، McNemar p=0.0654؛ هر ۱۶ CSV بایت‌به‌بایت یکسان | `s1_repro/headline_numbers.csv` |
| همان SVC-8PC با threshold پیش‌فرض 0.5 | 10/17 (0.588) | `s3_v1_configs/configs_table.csv` |
| ۱۲ config نسخه‌ی v1 | accuracy: min 0.294، **median 0.500**، max 0.824 (فقط 1 از 12) | همان |
| Clopper-Pearson 95% CI برای accuracy | [0.566, 0.962] | `s5_lpo_ci/ci_table.csv` |
| AUC (pooled LOPOCV) با bootstrap CI | 0.694 [0.431, 0.917] | همان |
| **Leave-Pair-Out AUC** | **0.556** | `s5_lpo_ci/lpo_auc.csv` |
| permutation، config ثابت (accuracy / AUC) | p = 0.117 / 0.273 | `s4_permutation/permutation_summary.csv` |
| **permutation، با احتساب انتخاب (بهترین از ۱۲)** | **p = 0.271** (95th صدک null = 16/17) | همان |
| champion روی ۵ بیمار ویدیودار / ۱۲ بیمار فقط-تصویری | 5/5 / 9/12 | `s6_confounders/stratified_performance.csv` |
| champion بازآموزی‌شده: فقط image-only (n=12) / فقط X1 (n=15) / بدون 190 و png (n=14) | 4/12 (AUC 0.17) / 8/15 / 7/14 | `s6_confounders/subgroup_retrained.csv` |
| champion روی embeddingهای FOV-cropped | 12/17 (AUC 0.708)؛ با همه‌ی mediaها 13/17 | `s7_fov/cropped_vs_uncropped.csv` |

**ب) baselineهای منصفانه (fully nested LOPOCV)**

در هر fold: tuning شامل C، gamma و PCA_K و انتخاب threshold فقط روی ۱۶ بیمار train انجام شد (inner LOO).
سطرها: `s2_baselines/summary.csv` و `s5_lpo_ci/ci_table.csv`

| feature set / مدل | correct | bal.acc | F1 | AUC [boot CI] | LPO-AUC | acc CP-CI |
|---|---|---|---|---|---|---|
| a clinical / LR | 6/17 | 0.354 | 0.353 | 0.444 [0.19, 0.74] | 0.458 | [0.14, 0.62] |
| a clinical / SVC | 8/17 | 0.472 | 0.471 | 0.611 [0.32, 0.89] | 0.694 | [0.23, 0.72] |
| a clinical / RF | 11/17 | 0.653 | 0.625 | 0.583 [0.31, 0.88] | – | [0.38, 0.86] |
| b BiomedCLIP / LR | 10/17 | 0.590 | 0.588 | 0.708 [0.44, 0.92] | 0.639 | [0.33, 0.82] |
| b BiomedCLIP / SVC | 11/17 | 0.646 | 0.667 | 0.583 [0.29, 0.88] | 0.597 | [0.38, 0.86] |
| b BiomedCLIP / RF | 12/17 | 0.708 | 0.706 | 0.597 [0.28, 0.88] | – | [0.44, 0.90] |
| c DINOv2 / LR | 6/17 | 0.375 | 0.000 | 0.250 [0.06, 0.51] | 0.347 | [0.14, 0.62] |
| c DINOv2 / SVC | 7/17 | 0.424 | 0.286 | 0.264 [0.07, 0.54] | 0.250 | [0.18, 0.67] |
| c DINOv2 / RF | 8/17 | 0.486 | 0.308 | 0.201 [0.00, 0.45] | – | [0.23, 0.72] |
| d both visual / LR | 8/17 | 0.479 | 0.400 | 0.389 [0.14, 0.68] | 0.472 | [0.23, 0.72] |
| d both visual / SVC | 5/17 | 0.306 | 0.143 | 0.292 [0.07, 0.60] | 0.389 | [0.10, 0.56] |
| d both visual / RF | 7/17 | 0.403 | 0.500 | 0.417 [0.15, 0.72] | – | [0.18, 0.67] |
| e visual+clinical / LR | 8/17 | 0.479 | 0.400 | 0.306 [0.08, 0.57] | 0.375 | [0.23, 0.72] |
| e visual+clinical / SVC | 7/17 | 0.417 | 0.375 | 0.361 [0.12, 0.64] | 0.403 | [0.18, 0.67] |
| e visual+clinical / RF | 9/17 | 0.535 | 0.500 | 0.424 [0.17, 0.72] | – | [0.28, 0.77] |
| *post hoc:* BiomedCLIP / LR، FOV-cropped | 13/17 | 0.757 | 0.800 | 0.736 | **0.729** | – |

- **کنترل مثبت:** pipeline سیگنال مصنوعی را پیدا می‌کند (AUC بین 0.71 و 0.89). یعنی اعداد پایین باگ نیستند (`s2_baselines/positive_control.txt`).
- **AUC کمتر از 0.5** (مثلاً در DINOv2) سوگیری شناخته‌شده‌ی LOO در نبود سیگنال است و به معنای «ضدپیش‌بینی» نیست.
- **ردیف post hoc** (sub-step 7d، `s7_fov/cropped_biomedclip_lr_checks.csv`):
  - permutation با config ثابت: p(AUC)=0.059 و p(acc)=0.038
  - این config از میان ۲۷ خانه انتخاب شد، پس اصلاح‌نشده است. با Bonferroni معنادار نیست.
  - بازآموزی در زیرگروه‌ها: image-only 5/12 (AUC 0.51)؛ X1-only 9/15 (0.71)؛ بدون 190/png 10/14 (0.59)؛ jpg-only 6/9 (0.90).

**ج) یکپارچگی داده و confounderها**

| مورد | مقدار | فایل |
|---|---|---|
| تعداد | ۲۴ بیمار، ۹۸ تصویر (45 jpg، 53 png)، ۳۹ ویدیو (36 mp4، 3 mkv) | `s0_integrity/media_inventory.csv` |
| duplicate بایتی / تقریباً یکسان | 0 / 3 جفت درون‌بیمار (2613_1≈2613_4 که هر دو استفاده شده‌اند) | `s0_integrity/duplicates.csv` |
| فایل خراب | `2612_8` (JPEG ناقص؛ هم‌اندازه با 2612_9 ولی duplicate نیست) | همان inventory |
| تصاویری که هرگز وارد pipeline نشدند | **58 از 98** (فقط 40 استفاده شده): 55 به‌خاطر لیست hard-coded در `phase2_task3_fix_missing_frames.py:7` (26 تای آن‌ها مال بیماران ویدیودار cohort)، 3 به‌خاطر فیلتر blur (2613) | `s0_integrity/excluded_images.csv` |
| processor در برابر label | 190: **2 Incomplete / 0 Complete** · X1: 7 / 8 | `s6_confounders/contingency_tables.csv` |
| file type در برابر label | png: 3I/0C · jpg: 4I/5C · video: 2I/3C | همان |
| OSD (پنل متنی روی صفحه) | در cohort: در همه‌ی ویدیوهای ۵ بیمار ویدیودار و png stillها وجود دارد و در jpg stillها نیست؛ 5I/3C در برابر 4I/5C. پنل تاریخ، نام بیمارستان و **مدل scope** (مثل GIF-EZ1500 و GIF-HQ190) را نشان می‌دهد. خارج از cohort: کنترل‌های 2623 و 2624 این پنل را ندارند. **FOV crop همه‌ی متن را حذف نمی‌کند:** در هر ۴ ورودی مدل از 2615 (CV-190) متن خوانا باقی مانده است | `s7_fov/fov_layouts.csv`، `figures/fov_qc_*.png` |
| shortcut probe (سطح بیمار) | embeddingها «video در برابر stills» را با AUC=1.00 و «OSD» را با AUC بین 0.94 و 1.00 تشخیص می‌دهند | `s6_confounders/shortcut_probe.csv` |
| shortcut probe (سطح تصویر، قبل → بعد از crop، BiomedCLIP) | png/jpg: 0.97 → 0.62 · still/video: 0.87 → 0.86 · پنل (نسخه‌ی اصلاح‌شده): 0.99 → 0.74؛ فقط روی stillها: 1.00 → 0.50 | `s6_confounders/shortcut_probe.csv`؛ برای پنل: `s6_confounders/shortcut_probe_osd_corrected.csv` |
| Attention rollout، مقایسه‌ی هندسی (MAE) | نگاشت اصلاح‌شده: 4.8 · نگاشت v1: 17.0. مدل فقط ۷۴٪ (BiomedCLIP) و ۵۷٪ (DINOv2) از قاب را دیده است | `s7_fov/rollout_mapping_check.csv` |

---

## ۲. پنج یافته‌ی اصلی

1. **❌ نتیجه‌ی 82.4% (14/17) باقی نماند.**
   - این عدد بهترینِ ۱۲ config است (median 0.50).
   - با threshold پیش‌فرض 10/17 است. LPO-AUC آن 0.556 است.
   - permutation test کل pipeline با احتساب انتخاب: p=0.271. با برچسب‌های تصادفی هم در بیش از ۵٪ موارد 15–16 از 17 به‌دست می‌آید.
   - McNemar p=0.065 که v1 گزارش کرده بود در برابر یک baseline ضعیف و بدون tuning بود.

2. **❌ 14/17 به confounderهای acquisition وابسته است.**
   - champion روی ۵ بیمار ویدیودار 5/5 است.
   - اگر بدون آن‌ها بازآموزی شود 4/12 می‌گیرد و بدون بیماران 190/png هم 7/14.
   - embeddingها منبع هر بیمار (video در برابر still) را با AUC 1.00 می‌شناسند.
   - processor 190 و فایل png فقط در Incomplete وجود دارند.
   - با n=17 نمی‌توان «سیگنال مخاطی» را از «سیگنال دستگاه/فرمت» جدا کرد.

3. **❌ DINOv2 و ویژگی‌های بالینی کمکی نکردند.**
   - در تنظیم nested، هر ترکیبی که DINOv2 دارد AUC بین 0.20 و 0.42 دارد (LPO بین 0.25 و 0.47).
   - clinical-only LR فقط 6/17 است.
   - افزودن DINOv2 یا clinical به BiomedCLIP عملکرد را پایین آورد.
   - ادعای «dual-vision بهتر است» پشتوانه ندارد.

4. **⚠️ تنها چیزی که «به‌سختی» زنده ماند: BiomedCLIP به‌تنهایی، با مدل خطی، بعد از FOV crop.**
   - 13/17، AUC 0.736، LPO-AUC 0.729؛ permutation با config ثابت p≈0.04–0.06.
   - اما post hoc است (یکی از ۲۷ خانه).
   - در زیرگروه image-only فرو می‌ریزد (5/12).
   - و still/video حتی بعد از crop هنوز با AUC 0.86 از embedding قابل خواندن است.
   - در crop بیمار 2615 هم هنوز متن روی صفحه دیده می‌شود، و این بیمار درست پیش‌بینی شده است.
   - این یک **فرضیه برای تأیید روی داده‌ی مستقل** است، نه یک نتیجه.

5. **✅ مشکل heatmapها بخشی‌اش باگ هندسی بود، و برطرف شد.**
   - در v1 نقشه‌ی attention مربعِ center-crop روی کل قاب غیرمربع کشیده می‌شد، پس همه‌ی نقاط جابه‌جا بودند (تا ۱۹٪ عرض قاب در هر طرف دیده نشده بود). این ثابت و اصلاح شد.
   - با FOV crop گوشه‌های سیاه هم حذف شدند.
   - **ولی** این باگ همه‌چیز را توضیح نمی‌دهد. در بررسی چشمی هر ۴ شکل (بدون اندازه‌گیری عددی): در 2603 نقاط داغ بیشتر روی glare هستند؛ در 2602 و 2605 (BiomedCLIP) بیشتر روی لبه‌ی lumen یا چین؛ در 2601 روی lumen، حباب‌ها و لبه‌های تصویر. DINOv2 نقاط کوچکی روی glare، حباب و لبه‌ی lumen دارد.
   - پس بازخورد پزشک مشاور فقط بخشی از آن با باگ توضیح داده می‌شود. نقشه‌های جدید باید به ایشان نشان داده شوند.
   - Attention Rollout class-agnostic است و شاهدی برای Complete یا Incomplete نیست.

**نکات فرعی ثبت‌شده در LOG:**
- آستانه‌ی blur (Laplacian ≥ 100) وابسته به رزولوشن است (میانه‌ی png=73، jpg=335).
- بیمار 2613 فقط با NBI نمایش داده شده است.
- stillهای بیماران ویدیودار هرگز استفاده نشده‌اند.
- figure2 و `patient_level_correction_analysis.csv` قدیمی جایگزین شدند با:
  - `results/step0/figures/step0_performance_panel.png`
  - `results/step0/patient_level_table.csv`

  هر دو با اسکریپت `step0/s8_figures.py` ساخته می‌شوند و هیچ متن «اثر بالینی» ندارند. **figure2 و جدول قدیمی نباید در مقاله استفاده شوند.**

---

## ۳. متن پیشنهادی برای بخش Limitations مقاله (انگلیسی)

> **Limitations.** This is an exploratory feasibility analysis of 17 patients (9 Incomplete, 8 Complete) from a single centre, and it is not powered for confirmatory inference. In our original pilot pipeline, the decision model, number of principal components and thresholding rule were chosen after inspecting leave-one-patient-out results across 12 configurations. When this selection is reproduced inside a label-permutation test of the whole pipeline, the best-of-12 accuracy (14/17) is not distinguishable from chance (p = 0.27). The leave-pair-out AUC of that configuration was 0.56. Under fully nested cross-validation, in which all scaling, dimensionality reduction, hyper-parameter tuning and threshold selection were done inside the training folds, no model had a 95% confidence interval lower bound above 0.5 for accuracy or AUC. Exact 95% confidence intervals for accuracy were wide (for example 0.33–0.82 for 10/17 correct). Acquisition factors were unevenly distributed between the subtypes: video screen-captures versus still images, PNG versus JPEG exports, and an older processor generation (CV-190) present only in Incomplete cases. Frozen image embeddings identified the media source of each patient almost perfectly. When these patients were removed, the performance of that configuration fell to about chance, so the observed discrimination may reflect acquisition rather than mucosal phenotype. On-screen text overlays and the black border outside the endoscopic field of view were present in part of the images. Cropping to the field of view reduced but did not remove the separability of media types, and readable text remained inside the crop for one CV-190 patient. Attention-rollout maps are class-agnostic and, on visual review of four frames, often highlighted non-mucosal structures such as the lumen edge, glare and image edges; they should not be read as localisation of metaplastic features. The histological reference was patient-level, and two mixed Complete+Incomplete cases were assigned to Incomplete. External, prospectively collected data with a uniform acquisition protocol are required before any statement about discrimination between GIM subtypes can be made.

---

## ۴. پیشنهاد برای قدم بعد (نیاز به تصمیم شما)

- در مقاله، هدف ۳ فقط به‌عنوان pilot منفی/غیرقطعی گزارش شود؛ عدد 82.4% حذف شود.
- اگر دوباره با GIM-ENDO کار کنیم:
  - همه‌ی mediaها FOV-crop شوند و fix_missing_frames اصلاح شود.
  - یک config از قبل ثبت شود (پیشنهاد: BiomedCLIP + LR، nested) و فقط همان گزارش شود.
  - برای هر patient، «منبع» در طراحی کنترل شود.
- نقشه‌های rollout جدید برای review به پزشک مشاور نشان داده شوند.
- طبق جلسه، شروع کار روی Goal 1 با EndoWLI-NBI.
