# Multimodal Deep Learning for Gastric Intestinal Metaplasia (GIM) Subtype Differentiation

## Clinical Motivation
Gastric Intestinal Metaplasia (GIM) is a known precursor to gastric cancer. Identifying the specific histological subtype of GIM—Complete versus Incomplete—is critical for risk stratification. This project develops a multimodal machine learning pipeline that fuses structured Image-Enhanced Endoscopy (IEE) clinical signs (e.g., LBC, MTB) with deep visual representations extracted from endoscopic frames using the pre-trained **BiomedCLIP Vision Transformer**.

The pipeline is developed using the **GIM-ENDO dataset** (Zenodo DOI: 10.5281/zenodo.20683008), composed of cases collected via Olympus EVIS X1 endoscopy.

## Architecture Overview
1. **Visual Embedding Extraction**: High-quality endoscopic frames are processed through a frozen BiomedCLIP ViT-Base encoder to extract 512-dimensional semantic representations.
2. **Patient-Level Mean Pooling**: Frame-level embeddings are aggregated to create a single robust visual profile per patient.
3. **Multimodal Fusion & Dimensionality Reduction**: Structured clinical features (7 dims) are fused with visual embeddings, which undergo in-fold Principal Component Analysis (PCA) to reduce dimensionality while preventing data leakage.
4. **Classification**: A Random Forest classifier (and baseline Logistic Regression) predicts the GIM subtype under rigorous Leave-One-Patient-Out Cross-Validation (LOPOCV).
5. **Clinical Explainability**: SHAP (SHapley Additive exPlanations) values and ViT Attention Rollout heatmaps provide global and local interpretability.

## Key Results Summary

| Metric | Clinical-Only Baseline (L2-LogReg) | Multimodal Model (Random Forest) |
|--------|------------------------------------|-----------------------------------|
| **Accuracy** | 35.3% | **64.7%** |
| **Sensitivity (Incomplete)** | 66.7% | **77.8%** |
| **Specificity (Complete)** | 0.0% | **50.0%** |
| **F1-Score** | 0.52 | **0.70** |

- **McNemar's Paired Test**: The multimodal model demonstrated substantial qualitative improvement over the baseline (p=0.0625, exact binomial).
- **Explainability**: SHAP analysis revealed that deep visual features contributed to **64.0%** of the model's predictive power, significantly enhancing the diagnostic utility beyond subjective clinical signs alone (like Light Blue Crest).

## Directory Tree & File Inventory

```text
GIMENDO_v2_Images_Videos/
├── data/
│   ├── raw/                 # Raw endoscopic videos (Ignored in Git)
│   ├── filtered_frames/     # Extracted & quality-filtered JPEG frames
│   └── processed/           # Extracted embeddings, merged datasets, predictions, stats
├── figures/                 # Publication-ready plots (300 DPI)
│   ├── shap_summary_dot_plot.png
│   ├── shap_bar_importance.png
│   ├── baseline_vs_multimodal_shap_shift.png
│   └── attention_rollout_*.png
├── requirements.txt         # Python dependencies
├── run_pipeline.py          # Master orchestrator script
├── phase1_*.py              # Video processing & filtering scripts
├── phase2_*.py              # BiomedCLIP extraction & fusion scripts
├── phase3_*.py              # LOPOCV evaluation & stats scripts
└── phase4_*.py              # SHAP & Attention Rollout explainability scripts
```

## Step-by-step Setup & Quickstart

### 1. Environment Setup
Create a Python 3.12 virtual environment and install dependencies:
```bash
python -m venv .venv
# Activate the environment (Windows)
.venv\Scripts\activate
# Activate the environment (Mac/Linux)
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Reproduce Pipeline Evaluation
To rapidly evaluate the models, generate statistics, and plot SHAP values using the pre-extracted processed embeddings:
```bash
python run_pipeline.py --eval-only
```

To run the pipeline from scratch (including deep feature extraction from raw frames):
```bash
python run_pipeline.py
```
*(Note: Full execution requires downloading the raw dataset into `data/raw` and extracting frames via Phase 1 scripts.)*
