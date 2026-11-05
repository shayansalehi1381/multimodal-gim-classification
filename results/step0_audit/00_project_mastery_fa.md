# گزارش شناخت کامل پروژه GIM (مرحله read-only)
تاریخ: 2026-10-26 · branch: `v2-rescope` · HEAD: `6f03d5d`

> در این مرحله هیچ فایلی از repo تغییر نکرد و هیچ commit یا push انجام نشد. فقط پوشه‌ی `results/step0_audit/` (untracked) ساخته شد.
> اسکریپت‌های phase3 و SHAP در یک کپی داخل scratchpad دوباره اجرا شدند (حدود ۱ دقیقه، روی CPU) تا خروجی‌ها مقایسه شوند. فایل‌های جزئی این پوشه:
> `frames_per_case.csv`, `video_frames.csv`, `cohort_confounders.csv`, `per_patient_final_models.csv`, `git_show_stat.txt`, `rerun_and_csv_check.log`.

---

## (a) پروژه چیست و re-scope چهار هدفه (به زبان دانشجو برای استاد)

**v1 (pilot):** ما می‌خواستیم از روی تصویر/ویدیوی آندوسکوپی، زیرنوع بافتی GIM یعنی **Complete در برابر Incomplete** را پیش‌بینی کنیم (Incomplete خطر سرطان بالاتری دارد). دیتاست GIM-ENDO v2 است (۲۴ بیمار). از دو مدل foundation به‌صورت frozen، یعنی BiomedCLIP (512-d) و DINOv2-base (768-d)، برای هر frame یک embedding گرفتیم. بعد برای هر بیمار میانگین embeddingها را گرفتیم (mean pooling + L2). این را با ۶ علامت آندوسکوپی (LBC, MTB, WOS, TVF, MLE, TM) و Age ترکیب کردیم و با LOPOCV روی ۱۷ بیمار و PCA داخل fold، یک SVC را ارزیابی کردیم. نتیجه‌ی گزارش‌شده 82.4% (14/17) بود.
**مشکل:** n=17 است، CI بسیار پهن است ([58.8, 100])، و 82.4% بهترینِ ۱۲ پیکربندی است (همین SVC با threshold=0.5 فقط 58.8% می‌دهد). baseline هم ضعیف و ناسازگار است و چند confounder وجود دارد (stills در برابر video، jpg در برابر png، CV-190 در برابر X1). attention rollout هم از نظر پزشک مشاور قانع‌کننده نبود.

**v2 (re-scope، طبق `docs/meeting_notes.md`):** چهار هدف پشت سر هم:
1. **IM vs non-IM** (تشخیص وجود متاپلازی)؛ پیشنهاد: شروع با EndoWLI-NBI (عمومی)
2. **Atrophy** (پیشنهاد: EGID)
3. **Complete vs Incomplete**: فقط GIM-ENDO دیتای endoscopy با برچسب زیرنوع دارد و این هدف pilot باقی می‌ماند.
4. **علائم آندوسکوپی (LBC/WOS/MTB) با localization**: هیچ دیتاست عمومی‌ای box/mask ندارد، پس باید خودمان annotate کنیم (پزشک مشاور، حدود ۱۰۰–۲۰۰ frame).

**Step Zero** (هنوز شروع نشده): baseline منصفانه، مدل فقط تصویری، جدول کامل ۱۲ config، permutation test، Leave-Pair-Out AUC، تحلیل به تفکیک نوع media، و FOV crop.

---

## (b) Repo و تاریخچه

**Remote:** GitHub (`origin`) · `git fetch --all --tags` چیز جدیدی نیاورد.

| ref | commit | تاریخ |
|---|---|---|
| `main`, `origin/main`, `v2-rescope`, `origin/v2-rescope`, tag `v1-pilot` (lightweight) | `6f03d5d` | 2026-09-08 |

- هر پنج ref دقیقاً روی یک commit هستند، پس `git diff --stat` بین v1-pilot، origin/main و v2-rescope **خالی** است. روی v2-rescope هنوز هیچ commit جدیدی نیست.
- **Working tree در برابر HEAD:** `GIMENDO_v2_Metadata.xlsx` تغییر کرده است. diff سلول‌به‌سلول نشان می‌دهد تنها تغییر یک `*` در سلول A42 (ردیف خالی) است و داده‌ها عوض نشده‌اند. `CLAUDE.md` و `docs/` untracked هستند.
- ۱۵ commit داریم که همه از یک نویسنده است. هیچ stash، revert یا branch دیگری وجود ندارد.

**فازهای تاریخچه** (جزئیات: `git_show_stat.txt`):
1. **2026-09-03 · `e534b00`:** یک commit بزرگ شامل همه‌ی اسکریپت‌های phase1–4، README، نتایج و figureهای v1 و **۹۸ تصویر خام بیماران (حدود ۲۳۵MB)**.
2. **2026-09-04 · DINOv2:** `31f6a3a` (README ادعای 82.4% کرد **قبل از** اینکه اسکریپت‌ها commit شوند)، `ac5f1d4` (اسکریپت‌ها و نتایج DINOv2)، `1c3d17f` (.gitignore و CSVها).
3. **2026-09-04 · reproducibility/docs:** `d359b9d` (requirements فقط با `>=`)، `db10a8f` (کار با CSV به‌جای pkl، rollout مقاوم‌تر)، `f87af4c`، `50ecce8`، **`61d90c0` (تصاویر از tracking حذف شدند ولی هنوز در history و روی GitHub هستند)**، `4b67aad`.
4. **2026-09-05 · رفع اشکال pipeline:** `2b1d32f`، `7bc8d26` (NameError)، `4b65635`.
5. **2026-09-05/08 · figureهای مقاله:** `5c5f2be` (`figure2_comprehensive_performance_panel.png` و `patient_level_correction_analysis.csv`، **بدون اسکریپت سازنده**)، `6f03d5d` (برچسب 2602 در rollout از Complete به Incomplete تغییر کرد و figure دوباره ساخته شد).

**کار تکراری:** تصاویر یک بار اضافه و بعد حذف شدند. figure 2602 دو بار ساخته شد.

**فایل‌های tracked** (۶۰ فایل): ۲۴ اسکریپت `.py`، ۲۲ فایل در `data/processed/` (CSV، و همچنین `.npy`، `.pkl` و `.xlsx` که **با وجود .gitignore هنوز tracked هستند**، چون قبل از اضافه شدن قانون commit شده بودند)، ۱۰ PNG در `figures/` (حدود ۲۲MB، که ۴ rollout هر کدام حدود ۵MB است)، به‌علاوه README، requirements، metadata xlsx و .gitignore.

**توضیح .gitignore:**
- خطوط 2–7: محیط‌ها (`.env`، `.venv/`، `venv/`، `env/`، `__pycache__`، `*.pyc`)
- خطوط 10–12: IDE (`.vscode/`، `.idea/`، checkpointها)
- خطوط 15–23: همه‌ی media و آرشیوها (`*.jpg|jpeg|png|mp4|mkv|avi|mov|rar|zip`)
- خطوط 24–26: `data/raw/`، `data/filtered_frames/`، `data/extracted_frames/`
- خطوط 29–30: استثنا، یعنی `figures/*.png|jpg` نگه داشته می‌شوند
- خطوط 33–35: `*.npy`، `*.pkl`، `data/processed/*.xlsx` (اما نسخه‌های قدیمی همچنان tracked هستند)

**فایل‌های ignored روی دیسک:**

| مورد | چیست | اندازه | فقط local؟ |
|---|---|---|---|
| ۱۳۷ فایل media در root (۹۸ تصویر + ۳۹ ویدیو) | داده‌ی خام | حدود ۳.۱GB | روی GitHub نیست. به گفته‌ی README روی Zenodo 20707267 هست (من بررسی نکردم). ۹۸ تصویر در history گیت هم هست. |
| `data/extracted_frames/` | ۱۹۷۷ frame (1fps) | ۵۵۸MB | فقط local (قابل بازتولید) |
| `data/filtered_frames/` | ۲۰۱۶ = ۱۹۷۶ frame ویدیو + ۴۰ still | ۶۰۵MB | فقط local |
| `data/processed/dinov2_frame_embeddings_2016x768.npy`، `dinov2_patient_features_17x768.npy`، `multimodal_biomedclip_dinov2_n17.pkl` | embeddingهای DINOv2 | کوچک | فقط local (معادل CSV آن‌ها tracked است) |
| `.venv/` | Python 3.14, torch 2.13, sklearn 1.9 | ۱.۶GB | فقط local |
| `.vscode/` | تنظیمات IDE | ۱KB | فقط local |
| `models/`، `outputs/plots`، `outputs/tables` | **خالی** | 0 | – |
| (خارج از repo) `~/.cache/huggingface` | وزن‌های BiomedCLIP و DINOv2 | ۱.۴GB | فقط local |

**(e) فایل‌های حساس:**
- هیچ credential، token یا مسیر absolute در فایل‌های tracked پیدا نشد (grep).
- شناسه‌ی بیمار فقط Case Code است (2601–2625)، بدون نام یا کد ملی.
- **ولی:** ۹۸ تصویر بالینی در commit `e534b00` در history روی GitHub می‌ماند. این با قانون CLAUDE.md («Never commit data») تعارض دارد. دیتا public است (CC BY-NC)، اما تصمیم درباره‌ی پاک‌کردن history با شماست.
- بزرگ‌ترین blobها حدود ۵MB هستند. history در مجموع حدود ۲۳۳MB است.

**(f) سازگاری CSVهای tracked با کد:**
- همه‌ی اسکریپت‌های phase3 و `phase4_task1/2` را در کپی scratch اجرا کردم. همه‌ی ۱۶ CSV **دقیقاً یکسان** بازتولید شدند (deterministic در این محیط).
- PNGها هم یکسان بودند (به‌جز `shap_summary_dot_plot` که تفاوت رندر دارد).
- pooling هم چک شد: میانگین embeddingها با CSVهای patient-level تا 1e-8 برابر است، برای هر دو مدل.
- **استثنا:** `figure2_comprehensive_performance_panel.png` و `patient_level_correction_analysis.csv` هیچ اسکریپت سازنده‌ای در repo ندارند و قابل بازتولید نیستند.
- phase1/2 (استخراج frame و embedding) دوباره اجرا نشد، چون سنگین است.

---

## (c) Pipeline قدم‌به‌قدم

| # | اسکریپت | ورودی → عملیات → خروجی | پارامترهای کلیدی |
|---|---|---|---|
| 0 | `phase1_setup_and_verify.py` | ساخت پوشه‌ها و نمایش ستون‌های xlsx | در `run_pipeline` نیست |
| 1 | `phase1_extract_frames.py` | ویدیوها (`.mp4/.mkv/.avi/.mov`) در `data/raw` **یا** `.` → `data/extracted_frames/CASE_xxxx_*_frameNNN.jpg` | هر `int(fps)` frame یکی (:36, :49)، تقریباً 1fps |
| 2 | `phase1_filter_blurry.py` | extracted → `data/filtered_frames` | Laplacian var ≥ `100.0` (:12) |
| 3 | `phase2_task3_fix_missing_frames.py` | stills **فقط** برای ۱۲ case hard-coded (:7) → filtered با پیشوند `CASE_` | همان blur ≥100 |
| 4 | `phase1_process_metadata.py` | `GIMENDO_v2_Metadata.xlsx` (header=1) → `cleaned_metadata_n17.xlsx` | قوانین حذف (:40–57)، mixed→Incomplete (:68–74)، علائم باینری (:86–96)، حذف IRV (:98–100) |
| 5 | `phase2_task2_extract_embeddings.py` | filtered → `frame_embeddings_1976x512.npy` (**در واقع 2016×512**) و `frame_metadata.csv` | BiomedCLIP ViT-B/16، open_clip، batch 32، L2 |
| 6 | `phase2_task3_patient_pooling.py` | → `patient_visual_features_n17.{npy,csv}` | mean + L2 |
| 7 | `phase2_task4_multimodal_fusion.py` | merge با metadata → `multimodal_dataset_n17.{csv,pkl}` (17×533) | fillna(0/"Unknown") (:51–56) |
| 8 | `phase2_dinov2_extract_embeddings.py` | filtered → `dinov2_frame_embeddings_2016x768.npy` | `facebook/dinov2-base`، CLS token، L2 |
| 9 | `phase2_dinov2_patient_pooling_and_fusion.py` | → `dinov2_patient_features_17x768.csv` و `multimodal_biomedclip_dinov2_n17.csv` (17×1301) | mean + L2 |
| 10 | `phase3_task1_lopocv_setup.py` | → `lopocv_splits_n17.pkl` | LeaveOneGroupOut با 17 fold |
| 11 | `phase3_task2_baseline_lr.py` | 7 ویژگی بالینی → `baseline_lr_predictions.csv` | L2-LR، GridSearch روی C∈{0.01, 0.1, 1, 10} با inner StratifiedKFold(3, rs=42)، threshold 0.5 |
| 12 | `phase3_task3_infold_pca.py` | → `infold_pca_summary.csv` | PCA k=5 در برابر ≥90% var (فقط diagnostic، در run_pipeline **نیست**) |
| 13 | `phase3_task4_multimodal_classifiers.py` | clinical + PCA5(BiomedCLIP) → `multimodal_classifier_predictions.csv`، `multimodal_model_comparison.csv` | RF(50, depth 3)، SVC(C=1)، GB(50, lr .05, depth 2)، rs=42، threshold 0.5 |
| 14 | `phase3_task5_olgim_subanalysis.py` | → `olgim_subcohort_analysis.csv` | n=7، در run_pipeline **نیست** |
| 15 | `phase3_task6_statistical_evaluation.py` | → `statistical_evaluation_report.csv` | bootstrap 1000 (rs=42)، McNemar RF در برابر LR |
| 16 | `phase3_dinov2_multimodal_evaluation.py` | clinical + PCA{5,8}(1280-d) → `dinov2_multimodal_evaluation_results.csv` | RF(100, d3)، SVC(rbf، پیش‌فرض)، GB؛ threshold 0.5 و Youden؛ ۱۲ config |
| 17 | `phase3_dinov2_final_stats_and_plots.py` | → `dinov2_final_statistical_report.csv`، ROC، و bar chart | فقط SVC با PCA=8 و Youden؛ LR پیش‌فرض؛ RF روی 512-d BiomedCLIP |
| 18 | `phase4_task1_shap_analysis.py` / `task2` | → `shap_feature_importance.csv`، `shap_values_cache.pkl`، ۳ figure | RF(50, d3) روی clinical + PCA5(**فقط BiomedCLIP**) |
| 19 | `phase4_task3_attention_rollout.py` | اولین frame هر case → `figures/attention_rollout_*.png` | BiomedCLIP ViT، میانگین head، rollout با residual |

**جزئیات threshold** (`find_optimal_threshold`، در `phase3_dinov2_multimodal_evaluation.py:30-43` و `phase3_dinov2_final_stats_and_plots.py:17-29`):
- `np.linspace(0.01, 0.99, 99)` با شرط `j > best_j`، پس **اولین ماکزیمم (کوچک‌ترین threshold)** انتخاب می‌شود.
- J روی احتمالات inner `cross_val_predict` با `StratifiedKFold(4, shuffle, rs=42)` روی **۱۶ بیمار train** حساب می‌شود.

---

## (d) واقعیت‌های داده (از روی فایل‌ها، نه از CLAUDE.md)

- **بیماران:**
  - metadata شامل ۲۴ case است: 2601–2625 بدون 2616 (ردیف ۴۰ یک `*` است).
  - برای هر ۲۴ نفر روی دیسک media وجود دارد.
- **Media:**
  - ۹۸ تصویر: ۴۵ `.jpg` (2601–2612) و ۵۳ `.png` (بخشی از 2610 و 2613–2625).
  - ۳۹ ویدیو: ۳۶ `.mp4` و ۳ `.mkv` (همه‌ی `.mkv`ها مال 2619 هستند).
  - ۱۴ بیمار فقط تصویر دارند (2601–2609 و 2611–2615).
  - ۱۰ بیمار تصویر و ویدیو دارند (2610، 2617–2625).
- **Cohort تحلیلی (n=17)**، تعیین‌شده در `phase1_process_metadata.py:40-57`:
  - **حذف‌شده‌ها (۷ نفر):**
    - 2609، 2617، 2618: «IM Positive» بدون زیرنوع
    - 2621: Negative
    - 2623، 2624: NaN (کنترل)
    - 2612: دیسپلازی LGD
  - **کلاس‌ها:** Incomplete=9 (2601، 2602*، 2605، 2610، 2611*، 2613، 2614، 2615، 2620) و Complete=8 (2603، 2604، 2606، 2607، 2608، 2619، 2622، 2625). موارد * mixed بودند و با قانون substring، Incomplete شدند.
- **منبع تصویری واقعی هر بیمار** (`frames_per_case.csv`):
  - ۱۲ بیمار cohort **فقط still** دارند (۱ تا ۱۰ عدد؛ مثلاً 2601 و 2604 و 2606 هر کدام فقط **۱ تصویر**).
  - ۵ بیمار cohort **فقط frame ویدیو** دارند (100 تا 454 عدد).
  - **stillهای بیماران ویدیودار (۴۵ تصویر، که ۲۶ تای آن‌ها مال بیماران cohort است) هرگز وارد pipeline نشده‌اند.** از ۹۸ تصویر فقط ۴۰ تا استفاده شده است.
  - جدول کلاس در برابر منبع: stills شامل 5C/7I، video شامل 3C/2I.
- **Processor:**
  - در نام فایل فقط 2614 و 2615 «190» هستند. در metadata، CV-190 برای 2614، 2615 و 2621 ثبت شده است (**2621 در نام فایل X1 است: تعارض**).
  - در cohort، CV-190 فقط روی دو Incomplete است (۰ Complete). 5 نفر processor ندارند (NaN).
- **ویدیوها:**
  - fps متفاوت است: 50.02 (int→50)، 30 برای mkv، و 13.73 برای کنترل‌ها (int→13).
  - رزولوشن هم متفاوت است: 1280×720، 1920×1080 (mkv)، و 480×288 (کنترل‌ها).
  - تقریباً همه‌ی کلیپ‌ها حدود ۵۰ ثانیه هستند و حدود ۵۰ frame می‌دهند (`video_frames.csv`).
- **فیلتر blur عملاً کاری نکرده است:** از ۱۹۷۷ frame ویدیو فقط ۱ مورد (2624_2) و از ۴۳ still فقط ۳ مورد (2613) حذف شد.

---

## (e) نتایج موجود (با نام فایل)

| مدل | Acc | F1 | AUC | فایل |
|---|---|---|---|---|
| L2-LR baseline (tuned C) | 0.353 (6/17)، sens .667، spec 0 | 0.522 | 0.500 | `baseline_lr_predictions.csv`، `multimodal_model_comparison.csv` |
| RF clinical+PCA5 BiomedCLIP | 0.647 (11/17) | 0.700 | 0.556 | `multimodal_model_comparison.csv` |
| SVM_RBF (همان ویژگی‌ها) | 0.529 | 0.600 | **0.153** | همان |
| GB | 0.588 | 0.667 | 0.500 | همان |
| RF در برابر LR: McNemar | χ²=3.2، p_corr=0.0736، p_exact=0.0625 (b=0، c=5) | | | `statistical_evaluation_report.csv` |
| **SVC، PCA8، Youden (champion)** | **0.824 (14/17)**، sens .667، spec 1.0؛ CI [0.588, 1.0] | 0.800 [.5, 1] | 0.694 [.414, .933] | `dinov2_final_statistical_report.csv` |
| McNemar SVC در برابر LR پیش‌فرض | p=0.0654 (svc-only=9، lr-only=2) | | | همان |
| همان SVC با threshold 0.5 | 0.588 | 0.588 | 0.694 | `dinov2_multimodal_evaluation_results.csv` |
| ۱۲ config روی هم | میانه حدود 0.47–0.53؛ بازه 0.294 تا 0.824 | | | همان |
| LR پیش‌فرض (C=1) | 0.412 (7/17) | 0.50* | 0.403* | *از re-run من: `per_patient_final_models.csv` |
| RF فقط روی BiomedCLIP 512-d | 0.647 | – | 0.5625* | همان |
| OLGIM sub-cohort (RF) | 5/7 | | | `olgim_subcohort_analysis.csv` |
| SHAP: سهم Visual PCs | حدود 64% (PC2 .071، PC3 .057، LBC .044) | | | `shap_feature_importance.csv` |

**خطاهای champion SVC:** 2601 (p=.136)، 2605 (.280)، 2614 (.488)؛ هر سه Incomplete و هر سه still-only هستند.
احتمال‌های per-patient در `per_patient_final_models.csv` و `patient_level_correction_analysis.csv` هستند.

---

## (f) Code audit

**(i) fit خارج از fold:**
- ✅ Scaler و PCA در همه‌ی مدل‌ها داخل fold هستند. نمونه‌ها: `phase3_task4_multimodal_classifiers.py:109-120`، `phase3_dinov2_multimodal_evaluation.py:91-103`، `phase3_dinov2_final_stats_and_plots.py:99-117`، `phase4_task1_shap_analysis.py:85-96`.
- ✅ C در baseline با inner CV انتخاب می‌شود (`phase3_task2_baseline_lr.py:109-115`).
- ❌ **Model/PCs/threshold-type selection:** ۱۲ config روی کل LOPOCV ارزیابی شد و بهترینش گزارش شد (`phase3_dinov2_multimodal_evaluation.py:67-77`). نتیجه در `phase3_dinov2_final_stats_and_plots.py:87,116` hard-code شده است. این selection bias است و nested CV وجود ندارد.
- ⚠️ `phase3_task1_lopocv_setup.py:74,86`: `Subtype_Binary` (خودِ label) در ماتریس `X` داخل `lopocv_splits_n17.pkl` است و با y برابر است (بررسی شد). هیچ اسکریپت دیگری این pkl را load نمی‌کند (grep)، پس فعلاً leakage فعال نیست، اما یک تله است.
- ⚠️ حذف IRV بر اساس واریانس کل داده بود (:98–100). اثرش ناچیز است.
- ⚠️ `fillna` در `phase2_task4_multimodal_fusion.py:51-56` روی کل جدول اعمال می‌شود، اما ستون‌هایی که پر می‌شوند در مدل استفاده نمی‌شوند.

**(ii) try/except خاموش:**
- `phase3_dinov2_final_stats_and_plots.py:124-128` و `phase3_dinov2_multimodal_evaluation.py:112-117`: اگر inner CV خطا بدهد، threshold روی احتمالات **in-sample** train انتخاب می‌شود.
- در داده‌ی فعلی این fallback **فعال نشد** (probe در `per_patient_final_models.csv`، ستون `fallback`=False برای هر ۱۷ fold). با این حال باید حذف شود.
- موارد دیگر:
  - `phase3_task6_statistical_evaluation.py:94-101`: bootstrap که خطا را قورت می‌دهد
  - `phase2_task2_extract_embeddings.py:108-113`: خطای load که با `continue` رد می‌شود، به‌همراه **باگ هم‌ترازی index** در :126–131 (اگر تصویری load نشود، embeddingها به ردیف اشتباه می‌روند؛ فعلاً errors=0 است)
  - `phase4_task3_attention_rollout.py:78-80`: اگر frame پیدا نشود، با `exit(0)` خارج می‌شود.

**(iii) Threshold:**
- با `np.linspace(0.01, 0.99, 99)` و شرط `>` اولین ماکزیمم انتخاب می‌شود.
- فقط از داده‌ی train (inner CV) استفاده می‌کند ✅.
- J_train بین 0.22 و 0.63 است. thresholdها بین 0.35 و 0.69 متغیرند.
- حساسیت نتیجه: Youden نتیجه را از 10/17 به 14/17 می‌برد. یعنی خود انتخاب «Youden در برابر 0.5» بخشی از همان ۱۲ config است.

**(iv) Pooling قبل یا بعد از split:**
- pooling قبل از split انجام می‌شود (`phase2_task3_patient_pooling.py:78`، `phase2_dinov2_patient_pooling_and_fusion.py:36-39`).
- اما چون عملیات فقط درون هر بیمار است (mean + L2 روی frameهای همان بیمار)، **leakage ندارد** ✅.

**(v) .mkv:**
- ✅ `phase1_extract_frames.py:17` پسوند `.mkv` را دارد. OpenCV هر ۳ فایل را باز می‌کند (30fps، 1530 frame).
- ۱۵۳ frame (51×3) از 2619 در embeddingها هست.
- نکته‌ها:
  - `int(fps)` در :36 (13.73→13).
  - اگر `data/raw` ویدیو داشته باشد، root اسکن نمی‌شود (:20–26). همین الگو در `phase2_task3_fix_missing_frames.py:18-24` هم هست.

**(vi) 2612_8 و 2612_9:**
- SHA256 متفاوت است:
  - `7efa0cd9…d480`
  - `a5e31bff…0086`
- هر دو 923,735 بایت هستند و ۶۴۷ بایت اول (header) یکسان است.
- **2612_8 فایل JPEG خراب و truncated است** (PIL: «image file is truncated»، EOI ندارد). 2612_9 سالم است.
- پس duplicate بایتی نیست، ولی 2612_8 احتمالاً یک کپی خراب است. 2612 در n=17 نیست، اما برای Goal 1/2/4 مهم است.

**(vii) نشت برچسب از نام فایل:**
- نام فایل فقط برای استخراج case code استفاده می‌شود (`phase2_task2_extract_embeddings.py:81-85`، `phase2_dinov2_extract_embeddings.py:52`). embeddingها فقط از pixel ساخته می‌شوند ✅.
- ویژگی‌های بالینی از ستون‌های xlsx در سطح بیمار می‌آیند، نه از نام فایل.
- **ریسک باز:** متن یا overlay روی frame (نام processor، تاریخ، مقیاس) و تفاوت رزولوشن/فرمت (jpg در برابر png، 480p در برابر 1080p) می‌تواند وارد embedding شود. این را بررسی نکردم.

**(viii) ناسازگاری اعداد:**
1. **Baseline دو مدل مختلف است:**
   - task2: 35.3% (6/17)، AUC 0.500، F1 0.522
   - final_stats (`LogisticRegression()` پیش‌فرض، :85): 41.2% (7/17)، AUC 0.403، F1 0.50
   - README سطر baseline را ترکیب کرده است (41.2% همراه با F1 .522 و AUC .500).
   - Table 1 مقاله (sens 33.3، spec 50، F1 .375) با **هیچ‌کدام** نمی‌خواند (LR پیش‌فرض: sens 55.6، spec 25).
2. **Figure2 panel B:** LR AUC=0.500 و BiomedCLIP RF AUC=**0.528**؛ در ROC مستقل: 0.40 و 0.56؛ task4 RF: 0.556. عدد 0.528 در هیچ فایلی نیست، و figure2 اسکریپت ندارد.
3. «BiomedCLIP RF 64.7%» دو مدل متفاوت است: task4 (clinical+PCA5) و final_stats (512-d خام، بدون clinical؛ `phase3_dinov2_final_stats_and_plots.py:105-110`). README می‌گوید «3 PCs» که غلط است.
4. bar chart با اعداد hard-coded `[41.2, 64.7, 82.4]` ساخته می‌شود (`phase3_dinov2_final_stats_and_plots.py:211`). baseline در `phase3_task4_multimodal_classifiers.py:59-63,190` هم hard-coded است. چاپ‌ها در `phase3_dinov2_multimodal_evaluation.py:169-170` هم hard-coded هستند.
5. `patient_level_correction_analysis.csv` احتمالات baseline task2 را دارد (6/17)، اما McNemar با LR پیش‌فرض (7/17) حساب شده است. مقاله می‌گوید «7 بیمار اصلاح شد»، ولی جدول 11 مورد svc-only و 3 مورد lr-only نشان می‌دهد. ستون‌های متنی این جدول ادعای بالینی دارند («eliminated unnecessary aggressive biopsy»).
6. مقاله H. pylori و Sex را covariate می‌داند، اما کد فقط ۶ علامت + Age دارد.
7. مقاله «exclusively EVIS X1» می‌گوید، ولی 2614 و 2615 روی CV-190 هستند.
8. SHAP مقاله «surrogate multimodal RF» نام دارد، در حالی که RF فقط روی BiomedCLIP با PCA5 است، نه dual-vision PCA8. اسم «Visual_PC2» بین foldها هم‌معنا نیست، چون PCA هر fold جدا fit می‌شود (`phase4_task1_shap_analysis.py:94-96,62`).
9. نام فایل `frame_embeddings_1976x512.npy` است ولی shape آن 2016×512 است.
10. تعریف علائم در README غلط است (TVF «Thin Viable Fold»، MLE «Mucosal Lead Elevation»، TM «Tubulovillous Mucosa»). شیت Abbreviations: TVF=Tubulovillous pattern with Fusion، MLE=Map-Like Erythema، TM=Tubular Metaplasia. مقاله تعریف سومی دارد.
11. `requirements.txt` بسته‌های `statsmodels` (`phase3_dinov2_final_stats_and_plots.py:15`) و `tqdm` را ندارد.
12. README به فایل `LICENSE` اشاره می‌کند که وجود ندارد. `run_pipeline.py` اسکریپت‌های task3، task5 و task1_load_model را اجرا نمی‌کند.

**(ix) Seeds و nondeterminism:**
- `random_state=42` همه‌جا هست، به‌علاوه `np.random.seed(42)` برای bootstrap (`phase3_dinov2_final_stats_and_plots.py:37`).
- re-run همه‌ی CSVها را bit-exact بازتولید کرد.
- منابع ریسک:
  - SVC `probability=True` (Platt داخلی) که به seed حساس است
  - GPU inference برای embeddingها که با CPU ممکن است کمی متفاوت باشد (بررسی نشد)
  - نسخه‌ها pin نیستند (فقط `>=`)
  - محیط فعلی Python 3.14 است که با README (3.10–3.12) فرق دارد.

**(x) گوشه‌های سیاه / FOV:**
- هیچ FOV mask یا crop وجود ندارد.
- در `attention_rollout_2602.png` بیشترین attention در گوشه‌ی سیاه بالا-راست است.
- **باگ هم‌ترازی (احتمالی):** `preprocess` در open_clip تصویر را به Resize(224) و سپس CenterCrop(224) می‌برد (`phase4_task3_attention_rollout.py:156`)، ولی heatmap مربع به اندازه‌ی کامل تصویر غیرمربع کشیده می‌شود (:170). پس نقشه نسبت به تصویر جابه‌جا و کشیده است. این را از config پیش‌فرض open_clip استنباط کردم و نیاز به تأیید عددی دارد.
- مشکلات دیگر:
  - colorbar یا legend ندارد (:172–184)؛ این همان بازخورد #2 جلسه است.
  - هر تصویر جداگانه min-max نرمال می‌شود (:168).
  - rollout روی BiomedCLIP است، نه DINOv2 یا SVC.
  - class-agnostic است.
  - frame انتخابی فقط اولین فایل الفبایی است (:56–71).

---

## (g) نامطمئن‌ها / تعارض‌ها / منابع غیرقابل دسترس

- **غیرقابل دسترس:**
  - GitHub issues و PRها: `gh` نصب نیست و بررسی نشد.
  - صفحه‌ی Zenodo (نسخه، لایسنس، تعداد فایل‌ها) را باز نکردم.
  - ایمیل‌ها و PDF ارسالی یکی از همکاران.
  - لینک Wiley (ueg2.12551).
  - Trello/Notion/Drive متصل‌اند، ولی جست‌وجو نکردم.
- **PDF فارسی** `docs/archive_v1/ایده(ویرایش شده).pdf`: متن فارسی با pdftotext استخراج نشد (encoding فونت). فقط بخش‌های انگلیسی‌اش را دیدم. اگر محتوایش مهم است، باید تصویری خوانده شود.
- **DOI:** README badge و مقاله 10.5281/zenodo.20683008 را می‌گویند، ولی لینک record 20707267 است. کاتالوگ هم همین را flag کرده است (v1: 37 ویدیو/99 تصویر، v2: 39/98). فایل‌های روی دیسک با v2 می‌خوانند.
- **شمارش کلاس:** کاتالوگ می‌گوید «8 Complete, 7 Incomplete, 3 Mixed» و کد 8/9 دارد (2612 که mixed است به دلیل LGD حذف شده). قانون «mixed→Incomplete» تصمیم بالینی است و باید با پزشک مشاور تأیید شود.
- **Metadata:**
  - 2621 در xlsx روی CV-190 است، ولی نام فایل‌هایش X1 است.
  - 2610_5 در تصویر «TM_Atr» است و در ویدیو «LBC_TM_IM_Atrophy».
  - 2610_9 در تصویر M-NBI است و در ویدیو NBI.
  - 2607: TVF در Body است ولی histology در Antrum.
  - 2617: IM Positive است، با نام فایل‌های عمدتاً Atrophy.
  - نمی‌دانم برچسب‌های sign در سطح بیمار (xlsx) با نام فایل‌ها (سطح تصویر) چقدر هم‌خوانی دارند.
- **2612_8:** خراب است. آیا نسخه‌ی سالم روی Zenodo هست؟ بررسی نشد.
- **مقیاس اثر confounderها:** still-only (۱ تا ۱۰ تصویر) در برابر video-only (۱۰۰ تا ۴۵۴ frame) و CV-190 که فقط روی Incomplete است. با n=17 قابل تفکیک آماری نیست. Step Zero (تحلیل per-media) باید این را بسنجد.
- **figure2 و correction table:** سازنده‌شان معلوم نیست (احتمالاً notebook یا اسکریپت خارج از repo).

Waiting for approval.
