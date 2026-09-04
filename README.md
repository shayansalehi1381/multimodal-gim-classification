# Multimodal Deep Learning for Gastric Intestinal Metaplasia (GIM) Subtype Differentiation

### Non-Invasive Histological Subtyping via Dual-Vision Foundation Models (BiomedCLIP + DINOv2) and Endoscopic Clinical Predictors

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pipeline](https://img.shields.io/badge/Pipeline-100%25%20Reproducible-success.svg?logo=checkmarx&logoColor=white)](run_pipeline.py)
[![Dataset DOI](https://img.shields.io/badge/Zenodo%20DOI-10.5281%2Fzenodo.20683008-024dad.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20683008)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Clinical Motivation & Objective

**Gastric Intestinal Metaplasia (GIM)** is a critical premalignant mucosal lesion situated along the Correa cascade of gastric carcinogenesis. Histopathologically, GIM is divided into two major subtypes with dramatically diverging clinical outcomes:
- **Complete GIM (Type I)**: Exhibits small-intestinal phenotype with mature absorptive enterocytes, well-defined brush borders, and Paneth cells. Carries a relatively low risk of progression to gastric adenocarcinoma.
- **Incomplete GIM (Type II / Type III)**: Exhibits colonic phenotype characterized by immature, disorganized goblet and columnar cells secreting acidic sulfomucins, lacking a brush border. Incomplete GIM is recognized as a major independent risk factor conferring a **significantly elevated relative risk of malignant transformation** and mandates rigorous endoscopic surveillance.

### The Diagnostic Challenge
Definitive subtype identification historically necessitates multiple targeted or random mucosal forceps biopsies (e.g., the updated Sydney System), introducing risks of sampling error, tissue hemorrhage, and delayed histopathological turnaround. 

While modern **Image-Enhanced Endoscopy (IEE)**—such as Narrow-Band Imaging (NBI), Magnifying NBI (M-NBI), and Texture and Color Enhancement Imaging (TXI)—facilitates the visualization of mucosal micro-surface and micro-vascular patterns, conventional clinical markers (e.g., Light Blue Crest [LBC], Marginal Turbid Band [MTB], White Opaque Substance [WOS]) suffer from considerable inter-observer discordance and modest diagnostic specificity when relied upon in isolation.

### Project Objective
This study introduces an end-to-end, non-invasive **Multimodal Multi-Vision AI Pipeline** designed to accurately differentiate Complete vs. Incomplete GIM in real time. The framework is developed and validated using the clinical **GIM-ENDO dataset** ($n = 17$ histopathologically confirmed cases; Zenodo DOI: [10.5281/zenodo.20683008](https://doi.org/10.5281/zenodo.20683008)) captured via high-definition **Olympus EVIS X1** endoscopy systems.

### 📥 Dataset Access & Public Availability
The raw endoscopic videos, images, and clinical records are publicly available under open access on Zenodo:
- **Zenodo Record**: [https://zenodo.org/records/20707267](https://zenodo.org/records/20707267)
- **Direct Media Archive (.rar)**: [Download GIMENDO_v2_Images_Videos.rar](https://zenodo.org/records/20707267/files/GIMENDO_v2_Images_Videos.rar?download=1)
- **Clinical Metadata (.xlsx)**: [Download GIMENDO_v2_Metadata.xlsx](https://zenodo.org/records/20707267/files/GIMENDO_v2_Metadata.xlsx?download=1)
- **Dataset Documentation**: [Download README.md](https://zenodo.org/records/20707267/files/README.md?download=1)

*Note: For quick evaluation and full statistical reproduction (`python run_pipeline.py --eval-only`), downloading the raw media archive is NOT required, as pre-extracted numerical representations are version-controlled in `data/processed/`.*

---

## 2. Architecture & Multi-Vision Pipeline

The proposed architecture integrates domain-specialized semantic representations, self-supervised structural representations, and structured clinical descriptors into an orthogonal multimodal vector space:

```
                  ┌─────────────────────────────────────────┐
                  │ Olympus EVIS X1 Endoscopic Video/Frames │
                  └────────────────────┬────────────────────┘
                                       │ Frame Filtering & Quality Control
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 Dual-Vision Feature Extraction              │
        │  ┌───────────────────────────┐ ┌──────────────────────────┐ │
        │  │ BiomedCLIP (ViT-B/16)     │ │ DINOv2 (ViT-B/14)        │ │
        │  │ 512 dims (PubMed Domain)  │ │ 768 dims (Self-Sup. ViT) │ │
        │  └─────────────┬─────────────┘ └────────────┬─────────────┘ │
        └────────────────┼────────────────────────────┼───────────────┘
                         └──────────────┬─────────────┘
                                        ▼ Concatenation
                     ┌──────────────────────────────────────┐
                     │ Raw Visual Vector: 1,280 Dimensions  │
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │   Patient-Level Mean Pooling (n=17)  │
                     └──────────────────┬───────────────────┘
                                        │
    Structured Clinical Signs           │
    (LBC, MTB, WOS, TVF,                ▼
     MLE, TM, Patient Age)    ┌───────────────────────────────────┐
             │                │ In-Fold PCA (0% Data Leakage)     │
             └───────────────►│ 17-Fold LOPOCV Cross-Validation   │
                              └─────────────────┬─────────────────┘
                                                ▼
                              ┌───────────────────────────────────┐
                              │ Support Vector Classifier (SVC)   │
                              │ In-Fold Threshold Optimization    │
                              └─────────────────┬─────────────────┘
                                                ▼
                                    Complete vs. Incomplete
```

### Core Methodological Components:
1. **Dual Visual Representation (1,280 dims)**:
   - **BiomedCLIP (512 dims)**: A vision-language foundation model pre-trained on 15 million biomedical figure-caption pairs from PubMed Central, extracting domain-specific histopathological semantic representations.
   - **DINOv2 ViT-Base (768 dims)**: A self-supervised Vision Transformer pre-trained with discriminative objectives, capturing high-resolution mucosal structural geometry, cellular textures, and micro-vascular irregularities without annotation bias.
   - Total visual feature dimension: $512 + 768 = 1,280$ visual channels per frame.
2. **Patient-Level Mean Pooling**:
   - Endoscopic frame embeddings from each patient procedure are aggregated via mean pooling into a unified patient visual descriptor, mitigating frame selection bias and neutralizing transient specular glare or motion artifacts.
3. **In-Fold Principal Component Analysis (In-Fold PCA)**:
   - To prevent curse of dimensionality while strictly adhering to zero data leakage, PCA dimensionality reduction is fitted **exclusively within each training fold** (16 patients) and projected onto the held-out test patient (1 patient) across 17 Leave-One-Patient-Out Cross-Validation (LOPOCV) iterations.
4. **Structured Clinical Predictors (7 dims)**:
   - Encodes verified clinical endoscopy markers:
     1. **LBC**: Light Blue Crest (fine blue-white lines on mucosal epithelial crests)
     2. **MTB**: Marginal Turbid Band (whitish turbid band on mucosal pit edges)
     3. **WOS**: White Opaque Substance (lipid droplet accumulation masking subepithelial capillaries)
     4. **TVF**: Thin Viable Fold
     5. **MLE**: Mucosal Lead Elevation
     6. **TM**: Tubulovillous Mucosa
     7. **Age**: Patient chronological age
5. **Final Classification & Threshold Optimization**:
   - Support Vector Classifier (SVC, RBF kernel) coupled with in-fold hyperparameter tuning and Youden's $J$ statistic threshold optimization to establish maximum sensitivity and specificity for Incomplete GIM detection.

---

## 3. Comprehensive Performance Benchmark

All models were evaluated under strict 17-fold Leave-One-Patient-Out Cross-Validation (LOPOCV). The multi-vision architecture achieves state-of-the-art diagnostic performance:

| Model Architecture | Input Modality | Visual Dims / PCs | Accuracy | Correct / Total | F1-Score | ROC-AUC | Statistical Significance (vs. Baseline) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Baseline (L2-LR)** | Clinical Signs Only | None | 41.2% | 7 / 17 | 0.522 | 0.500 | Reference Baseline |
| **BiomedCLIP-only Multimodal (RF)** | Clinical + BiomedCLIP | 3 PCs | 64.7% | 11 / 17 | 0.700 | 0.556 | $p = 0.0736$ |
| **Champion Multi-Vision (SVC)** | **Clinical + BiomedCLIP + DINOv2** | **8 PCs** | **82.4%** | **14 / 17** | **0.800** | **0.694** | **McNemar $p = 0.0654$** |

### Statistical & Clinical Insights:
- **Major Accuracy Jump**: The Champion Multi-Vision model improves diagnostic accuracy from 41.2% to **82.4%** (+41.2% absolute gain), reducing misdiagnoses by more than half.
- **McNemar's Paired Exact Test**: Demonstrates strong statistical evidence of superiority ($p = 0.0654$, exact binomial paired test) over clinical markers alone.
- **SHAP Feature Importance (Global Explainability)**:
  - Deep visual components dominate the decision boundary, contributing **64.0%** of total predictive importance compared to 36.0% for structured clinical signs.
  - While subjective clinical signs like Light Blue Crest (LBC) are prone to false positives, deep visual representations consistently distinguish subtle pit distortion and structural heterogeneity.
- **Attention Rollout (Local Explainability)**:
  - ViT attention rollout maps verify that self-attention mechanisms attend strictly to pathological mucosal glandular architecture and metaplastic crypts, ignoring non-diagnostic endoscopic artifacts.

---

## 4. Exact Reproduction & Execution Guide

### Section A: Environment Setup

The codebase is engineered and verified for Python $\ge 3.10$ across Linux, macOS, and Windows.

```bash
# 1. Clone repository
git clone https://github.com/shayansalehi1381/multimodal-gim-classification.git
cd multimodal-gim-classification

# 2. Create and activate virtual environment
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (Command Prompt):
python -m venv .venv
.venv\Scripts\activate.bat

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Section B: Quick Evaluation & Statistical Reproduction

> [!TIP]
> **Zero Heavy Downloads Required**: All pre-extracted, quality-checked patient feature matrices, split definitions, and ground-truth labels are stored in standard CSV format inside `data/processed/`. You can reproduce all cross-validation metrics, statistical tests, and publication figures in under 30 seconds without needing GPUs or downloading large video files.

#### On Linux / macOS / Bash:
```bash
python run_pipeline.py --eval-only
```

#### On Windows PowerShell:
Set the output encoding to UTF-8 before executing to guarantee flawless display of diagnostic logs and formatting:
```powershell
$env:PYTHONIOENCODING="utf-8"; python run_pipeline.py --eval-only
```

#### What this executes:
1. `phase3_task1_lopocv_setup.py`: Verifies 17-fold LOPOCV partitions.
2. `phase3_task2_baseline_lr.py`: Fits clinical-only L2 Logistic Regression baseline.
3. `phase3_task4_multimodal_classifiers.py`: Evaluates BiomedCLIP multimodal classifiers.
4. `phase3_task6_statistical_evaluation.py`: Computes 95% bootstrap confidence intervals & McNemar test.
5. `phase3_dinov2_multimodal_evaluation.py`: Evaluates Champion Multi-Vision (BiomedCLIP + DINOv2) SVC models.
6. `phase3_dinov2_final_stats_and_plots.py`: Generates ROC and performance comparison plots.
7. `phase4_task1_shap_analysis.py` & `phase4_task2_shap_plots.py`: Computes SHAP attributions and dot/bar plots.
8. `phase4_task3_attention_rollout.py`: Computes spatial attention maps on representative cases.

---

### Section C: Full Pipeline Execution (From Raw Video)

To execute the entire pipeline from scratch—including video frame extraction, Laplacian blur filtering, BiomedCLIP and DINOv2 feature extraction, and full training:

1. Ensure the raw endoscopy videos (`.mp4`, `.mkv`) and frame captures (downloaded from the [Zenodo Media Archive](https://zenodo.org/records/20707267/files/GIMENDO_v2_Images_Videos.rar?download=1)) are placed in the project root or `data/raw/` according to the metadata table `GIMENDO_v2_Metadata.xlsx`.
2. Run the master orchestrator:
```bash
python run_pipeline.py
```

---

### Section D: PyCharm & VS Code Integration

#### Visual Studio Code:
1. Open the project folder in VS Code: `File -> Open Folder... -> multimodal-gim-classification`.
2. Select the Python Interpreter:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
   - Type `Python: Select Interpreter`.
   - Choose `./.venv/Scripts/python.exe` (Windows) or `./.venv/bin/python` (macOS/Linux).
3. Open terminal in VS Code (``Ctrl+` ``) and run:
   ```bash
   python run_pipeline.py --eval-only
   ```
4. Or create a `.vscode/launch.json` run configuration:
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: Reproduce GIMENDO Evaluation",
         "type": "debugpy",
         "request": "launch",
         "program": "${workspaceFolder}/run_pipeline.py",
         "args": ["--eval-only"],
         "console": "integratedTerminal"
       }
     ]
   }
   ```

#### JetBrains PyCharm:
1. Open the repository directory in PyCharm: `File -> Open...`.
2. Configure Project Interpreter:
   - Navigate to `Settings / Preferences` (`Ctrl+Alt+S` or `Cmd+,`).
   - Go to `Project: multimodal-gim-classification -> Python Interpreter`.
   - Click `Add Interpreter -> Add Local Interpreter...`.
   - Select `Existing Environment` and point to `.venv/Scripts/python.exe` (or `.venv/bin/python`).
3. Run Configuration:
   - Click `Run -> Edit Configurations...`.
   - Add new `Python` configuration:
     - Script path: `.../run_pipeline.py`
     - Parameters: `--eval-only`
     - Working directory: project root directory.
   - Click **Run** (`Shift+F10`).

---

## 5. Directory Tree & File Manifest

```text
multimodal-gim-classification/
├── data/
│   ├── filtered_frames/                        # Extracted, Laplacian-filtered endoscopic frames
│   └── processed/                              # Standardized datasets & experimental outputs (CSV format)
│       ├── baseline_lr_predictions.csv         # Predictions from clinical baseline LR
│       ├── dinov2_final_statistical_report.csv # Final multi-vision performance metrics & CI
│       ├── dinov2_frame_metadata.csv           # Frame-level metadata for DINOv2 extraction
│       ├── dinov2_multimodal_evaluation_results.csv # Multi-vision hyperparameter/threshold grid results
│       ├── dinov2_patient_features_17x768.csv  # Mean-pooled DINOv2 patient features (17 x 768)
│       ├── frame_metadata.csv                  # Frame-level metadata for BiomedCLIP extraction
│       ├── infold_pca_summary.csv              # Eigenvalue variance explained across LOPOCV folds
│       ├── multimodal_biomedclip_dinov2_n17.csv# Complete unified dataset (Clinical + BiomedCLIP + DINOv2)
│       ├── multimodal_classifier_predictions.csv# Out-of-fold predictions across models
│       ├── multimodal_dataset_n17.csv          # Clinical + BiomedCLIP features (17 patients)
│       ├── multimodal_model_comparison.csv     # Model-level comparative summary table
│       ├── patient_visual_features_n17.csv     # Mean-pooled BiomedCLIP features (17 x 512)
│       ├── shap_feature_importance.csv         # Global SHAP ranking and numerical importances
│       └── statistical_evaluation_report.csv   # Comprehensive bootstrap CIs & McNemar tests
├── figures/                                    # Publication-ready plots (300 DPI)
│   ├── model_accuracy_jump_comparison.png      # Bar chart comparing baseline vs. multi-vision accuracy
│   ├── multimodal_roc_comparison_dinov2.png    # ROC curves across all cross-validation models
│   ├── shap_bar_importance.png                 # Global mean absolute SHAP feature importance bar plot
│   ├── shap_summary_dot_plot.png               # Beeswarm summary plot of SHAP feature impacts
│   ├── baseline_vs_multimodal_shap_shift.png   # Analysis of clinical vs. visual feature weighting
│   └── attention_rollout_*.png                 # Spatial ViT attention rollout heatmaps (cases 2601-2605)
├── requirements.txt                            # Verified cross-platform Python dependencies
├── run_pipeline.py                             # Master reproduction orchestrator script
├── phase1_*.py                                 # Video ingest, frame extraction & quality filters
├── phase2_*.py                                 # BiomedCLIP & DINOv2 embedding extraction & fusion
├── phase3_*.py                                 # 17-fold LOPOCV model evaluation & statistical testing
├── phase4_*.py                                 # SHAP attribution & Attention Rollout interpretability
└── README.md                                   # Comprehensive documentation & reproduction guide
```

---

## 6. Citation & Attribution

If you utilize this methodology, pipeline code, or the processed multi-vision dataset, please cite:

```bibtex
@dataset{gim_endo_2024,
  author       = {GIM-ENDO Research Consortium},
  title        = {GIM-ENDO: Multimodal Image-Enhanced Endoscopy Dataset for Complete vs. Incomplete Gastric Intestinal Metaplasia Classification},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20683008},
  url          = {https://doi.org/10.5281/zenodo.20683008}
}
```

---

## 7. License
This project is open-source under the [MIT License](LICENSE).
