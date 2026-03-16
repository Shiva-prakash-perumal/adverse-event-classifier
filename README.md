# Adverse Event Intelligence Pipeline

An end-to-end clinical ML pipeline that classifies adverse event severity from FDA FAERS data using XGBoost, Logistic Regression, and Random Forest, with an LLM extraction layer for unstructured clinical notes, MLflow experiment tracking, a Streamlit prototype UI, and CI/CD via GitHub Actions.

---

## What This Does

Adverse events in clinical trials are frequently reported as unstructured free-text notes. Manual triage of thousands of reports is slow, inconsistent, and not scalable. This pipeline automates severity classification using **919,627 real FDA FAERS patient records**:

1. **Ingests** real FDA FAERS data across four quarters (DEMO, DRUG, REAC, OUTC tables)
2. **Extracts** structured fields from raw clinical notes using Mistral AI with rule-based fallback
3. **Engineers** 14 clinical domain features with MedDRA-aware serious AE classification
4. **Selects** the best features using two-stage Mutual Information + RFE pipeline
5. **Trains** three models with cost-sensitive learning to handle class imbalance
6. **Tunes** hyperparameters across all three models using GridSearchCV
7. **Tracks** every experiment with MLflow
8. **Deploys** a Streamlit prototype demonstrating product integration
9. **Containerizes** everything in Docker with CI/CD via GitHub Actions

---

## Model Performance

Four iterations of progressive improvement on 919,627 real FAERS records:

| Iteration | Change | F1 Macro | Key Learning |
|---|---|---|---|
| 1 — Baseline | Initial training | 0.392 | Moderate class invisible (F1=0.006) |
| 2 — Class Balancing | Cost-sensitive learning | 0.369 | Moderate first detected (F1=0.073) |
| 3 — Hyperparameter Tuning | GridSearchCV all models | 0.406 | Overall accuracy improved |
| 4 — MedDRA Case Fix | Case-insensitive AE matching | **0.420** | `is_serious_ae` became real clinical signal |

### Final Model Comparison

| Model | F1 Macro | F1 Moderate | F1 Severe | AUC-ROC | Status |
|---|---|---|---|---|---|
| Logistic Regression | 0.385 | 0.000 | 0.511 | 0.627 | Baseline |
| XGBoost | 0.414 | 0.011 | 0.579 | 0.696 | Comparison |
| **Random Forest** | **0.420** | **0.022** | **0.574** | **0.701** | **Production** |

**Production model: Random Forest** — selected by F1 Macro and AUC-ROC.

> **Note on Moderate class:** Moderate outcomes (DS=Disability, CA=Congenital Anomaly) represent only 1.7% of FAERS reports. This reflects real-world FDA reporting patterns — not a modeling limitation. Cost-sensitive learning ensures the model does not ignore this minority class.

---

## Architecture

```
FDA FAERS Data (4 quarters: 25Q1–25Q4)
919,627 patient records
         ↓
    Data Ingestion
    (ingestion.py)
    - Load DEMO, DRUG, REAC, OUTC tables
    - Age/weight unit normalization
    - Severity mapping from FDA outcome codes
    - Deduplication across quarters
         ↓
  Feature Engineering
    (features.py)
    - Clean data + encode categoricals
    - Engineer 14 clinical features
    - Case-insensitive MedDRA AE matching
    - Stage 1: Mutual Information (top 12)
    - Stage 2: RFE with saga solver (final 10)
         ↓
    ┌─────────────────────────────────┐
    │  Model Training                 │  ← MLflow tracking
    │  Logistic Regression            │
    │  Random Forest ← production     │
    │  XGBoost                        │
    │  GridSearchCV hyperparameter    │
    │  tuning for all three models    │
    │  Cost-sensitive class weighting │
    └─────────────────────────────────┘
         ↓
    Model Evaluation
    (evaluate.py)
    - Per-class F1 / AUC-ROC
    - Calibration curves (Brier score)
    - Abstain thresholds for human review
         ↓
    LLM Extraction Layer
    (llm_extractor.py)
    - Mistral AI via      
    - Rule-based regex fallback
         ↓
    Streamlit Prototype
    (streamlit_app.py)
         ↓
    Docker + GitHub Actions CI/CD
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/Shiva-prakash-perumal/adverse-event-classifier
cd adverse-event-classifier
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env — Mistral API key optional, pipeline works without it
```

### 3a. Run with real FAERS data (recommended)

Download from [FDA FAERS](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html) and unzip ASCII files into `data/faers/`:

```
data/faers/
    DEMO25Q1.txt  DEMO25Q2.txt  DEMO25Q3.txt  DEMO25Q4.txt
    DRUG25Q1.txt  DRUG25Q2.txt  ...
    REAC25Q1.txt  REAC25Q2.txt  ...
    OUTC25Q1.txt  OUTC25Q2.txt  ...
```

```bash
python src/pipeline.py
```

### 3b. View MLflow experiment results

```bash
mlflow ui
# Open http://localhost:5000
```

### 4. Launch the Streamlit prototype

```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

### 5. Run tests

```bash
pytest tests/test_pipeline.py -v
```

### 6. Run with Docker

```bash
docker build -t adverse-event-classifier .
python src/pipeline.py  # Train first to generate model artifacts
docker run -p 8501:8501 -v $(pwd)/models:/app/models adverse-event-classifier
```

---

## Project Structure

```
adverse-event-classifier/
│
├── data/                          # Data files (gitignored)
│   └── faers/                     # Real FAERS .txt files — download separately
│
├── src/                           # Core pipeline modules
│   ├── ingestion.py               # FAERS loader — all 4 quarters, all 4 tables
│   ├── features.py                # Feature engineering + MI/RFE selection
│   ├── train.py                   # Model training + GridSearchCV + MLflow
│   ├── evaluate.py                # ROC, calibration, abstain threshold analysis
│   ├── llm_extractor.py           # Mistral AI extraction + rule-based fallback
│   ├── pipeline.py                # End-to-end orchestrator
│   └── evaluate_standalone.py    # Run evaluation without full pipeline rerun
│
├── app/
│   └── streamlit_app.py           # Customer-facing prototype UI
│
├── models/                        # Saved model artifacts (gitignored)
│   ├── production_model.pkl       # Best model (Random Forest)
│   ├── feature_names.pkl          # Feature list used at training time
│   ├── model_comparison.csv       # All model metrics across runs
│   └── *.png                      # ROC, calibration, confusion matrix plots
│
├── tests/
│   └── test_pipeline.py           # 23 pytest unit + integration tests
│
├── mlruns/                        # MLflow experiment tracking (gitignored)
│
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI/CD
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Feature Engineering

### Engineered Features (14 total → 10 selected)

| Feature | Clinical Rationale |
|---|---|
| `age`, `weight_kg`, `dosage_mg` | Core patient/drug characteristics from FAERS |
| `age_group` | Pediatric / Young Adult / Adult / Elderly bucketing |
| `elderly_flag` | Binary: age > 65 — higher adverse event risk |
| `high_dose_flag` | Binary: dosage above cohort median |
| `is_serious_ae` | MedDRA-aware: flags cardiac arrest, dyspnoea, hepatic failure, etc. |
| `risk_score` | Composite clinical risk (softened weights to prevent dominance) |
| `gender_encoded`, `route_encoded` | Encoded categorical variables |
| `num_concomitant_drugs` | Polypharmacy — more drugs = higher interaction risk |
| `has_comorbidity`, `has_prior_reaction` | Patient history risk factors |

### Two-Stage Feature Selection

**Stage 1 — Mutual Information:** Scores all 14 features by how much they reduce uncertainty about the target. Captures both linear and non-linear relationships.

**Stage 2 — Recursive Feature Elimination (RFE):** Iteratively removes the least important feature until the optimal 10-feature subset is found. Uses `saga` solver for large-scale stability.

---

## Class Imbalance Handling

FAERS data is severely imbalanced — Moderate outcomes (DS/CA codes) are only 1.7% of reports:

| Class | Count | % |
|---|---|---|
| Mild (OT/RI) | ~451,000 | 49% |
| Severe (DE/LT/HO) | ~447,000 | 49% |
| Moderate (DS/CA) | ~21,000 | 1.7% |

**Solution: Cost-sensitive learning** — adjusts how much the model is penalized for misclassifying each class, without modifying the data:

- **Logistic Regression / Random Forest:** `class_weight="balanced"` — sklearn auto-weights inversely proportional to class frequency
- **XGBoost:** `compute_sample_weight("balanced")` passed to `fit()` — XGBoost does not support `class_weight` natively

SMOTE was not used — synthetic clinical records risk introducing medically unrealistic patient profiles.

---

## LLM Extraction Layer

Used at inference time to extract structured features from free-text clinical notes:

```
"68 year old male reported chest pain after 100mg DrugX via oral route"
                              ↓  Mistral AI
{age: 68, gender: "Male", dosage_mg: 100, route: "Oral",
 adverse_event: "chest pain", has_comorbidity: 0, ...}
                              ↓
                    XGBoost / Random Forest
                              ↓
                    Severity: Severe (87.3%)
```

**Fallback chain:** Mistral AI (     ) → Rule-based regex extractor

Configure in `.env`:
```
MISTRAL_API_KEY=your_key
MISTRAL_BASE_URL=your_api_base_url
MISTRAL_MODEL=your_llm_model_url
MISTRAL_CERT_PATH=/path/to/cert.pem
```

---

## MLflow Tracking

Every training run logs:
- All hyperparameters
- Per-class F1 (Mild / Moderate / Severe)
- AUC-ROC (macro OvR)
- Cross-validation mean and std
- Confusion matrix image
- Feature importance plot
- Full classification report

View results: `mlflow ui` → http://localhost:5000

---

## CI/CD (GitHub Actions)

| Job | Trigger | Action |
|---|---|---|
| Run Tests | Every push to develop/master | Runs 23 pytest unit + integration tests |
| Build Docker | Tests pass | Builds image + smoke test |
| Train & Validate | Push to master only | Trains on synthetic data, fails if F1 < 0.33 |

---

## Tech Stack

| Category | Tools |
|---|---|
| ML & Modeling | scikit-learn, XGBoost, GridSearchCV |
| LLM & AI | Mistral AI (     ), LangChain |
| Data | Pandas, NumPy — 919K real FAERS records |
| Feature Selection | Mutual Information, RFE |
| MLOps | MLflow, Docker, GitHub Actions |
| Prototype | Streamlit |
| Testing | pytest (23 tests) |
| Languages | Python 3.11 |

---

## Results

Model comparison and evaluation plots are available in the
[v1.0.0 release](https://github.com/Shiva-prakash-perumal/adverse-event-classifier/releases/tag/v1.0.0)

---

## Disclaimer

This is a research prototype built to demonstrate end-to-end ML pipeline skills for clinical trial applications. All predictions should be reviewed by qualified clinical staff before any clinical decision is made. Not intended for production clinical use.
