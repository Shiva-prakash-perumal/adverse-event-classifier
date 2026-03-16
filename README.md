# Adverse Event Intelligence Pipeline

An end-to-end ML pipeline for classifying adverse event severity in clinical trials using LLM extraction + classical ML + MLOps.

---

## What This Does

Adverse events (unexpected side effects in clinical trials) are often reported as unstructured free text. This pipeline:

1. **Extracts** structured fields from raw clinical notes using an LLM
2. **Engineers** clinical domain features (risk score, elderly flag, serious AE flag)
3. **Selects** the best features using Mutual Information + RFE
4. **Trains** and compares three models: Logistic Regression, Random Forest, XGBoost
5. **Tracks** every experiment with MLflow
6. **Deploys** a Streamlit prototype showing how this could live in a clinical product
7. **Containerizes** everything in Docker with CI/CD via GitHub Actions

---

## Architecture

```
FDA FAERS Data (or synthetic)
          ↓
    Data Ingestion
    (ingestion.py)
          ↓
  Feature Engineering
    (features.py)
    - Clean data
    - Encode categoricals
    - Derive risk features
    - MI + RFE selection
          ↓
    ┌─────────────────────┐
    │  Model Training     │  ← mlflow tracking
    │  Logistic Regression│
    │  Random Forest      │
    │  XGBoost ← best     │
    └─────────────────────┘
          ↓
    Model Evaluation
    (evaluate.py)
    - Per-class F1/AUC
    - Calibration curves
    - Abstain thresholds
          ↓
    LLM Extraction Layer          
    (llm_extractor.py)
    - OpenAI / Mistral / Rules
          ↓
    Streamlit Prototype
    (streamlit_app.py)
          ↓
    Docker + GitHub Actions
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/adverse-event-intelligence
cd adverse-event-intelligence
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys (optional — works without them)
```

### 3. Run the full pipeline (generates data, trains, evaluates)

```bash
python src/pipeline.py
```

### 4. View MLflow experiment results

```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Launch the Streamlit prototype

```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

### 6. Run tests

```bash
pytest tests/test_pipeline.py -v
```

### 7. Run with Docker

```bash
# Build
docker build -t adverse-event-intelligence .

# Train first (outside Docker to save artifacts)
python src/pipeline.py

# Run Streamlit app
docker run -p 8501:8501 -v $(pwd)/models:/app/models adverse-event-intelligence
```

---

## Project Structure

```
adverse-event-intelligence/
│
├── data/                          # Data files
│   └── faers_synthetic.csv        # Generated synthetic FAERS data
│
├── src/                           # Core pipeline modules
│   ├── ingestion.py               # Data loading + synthetic generation
│   ├── features.py                # Feature engineering + selection
│   ├── train.py                   # Model training + MLflow tracking
│   ├── evaluate.py                # Deep model evaluation
│   ├── llm_extractor.py           # LLM/rule-based note extraction
│   └── pipeline.py                # End-to-end orchestrator
│
├── app/
│   └── streamlit_app.py           # Customer-facing prototype UI
│
├── models/                        # Saved model artifacts
│   ├── production_model.pkl       # Best model
│   ├── feature_names.pkl          # Feature list
│   └── model_comparison.csv       # All model metrics
│
├── tests/
│   └── test_pipeline.py           # Pytest unit + integration tests
│
├── mlruns/                        # MLflow experiment tracking
│
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI/CD
│
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── README.md
```

---

## Models Compared

| Model | F1 Macro | AUC-ROC | Notes |
|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.85 | Fast, interpretable baseline |
| Random Forest | ~0.79 | ~0.89 | Robust, good feature importance |
| **XGBoost** | **~0.84** | **~0.92** | **Best — production model** |

---

## Feature Selection

Two-stage selection process:

**Stage 1 — Mutual Information**
Scores all features by how much they reduce uncertainty about severity.
Captures both linear and non-linear relationships.

**Stage 2 — Recursive Feature Elimination (RFE)**
Trains model iteratively, removes least important features until
optimal subset found.

Final features used:
- `age`, `dosage_mg`, `time_to_onset_days`
- `num_concomitant_drugs`, `symptom_count`
- `has_comorbidity`, `has_prior_reaction`
- `is_serious_ae`, `elderly_flag`, `high_dose_flag`, `risk_score`

---

## LLM Extraction

The pipeline supports three extraction backends:

| Backend | Setup | Accuracy |
|---|---|---|
| OpenAI GPT-3.5 | `OPENAI_API_KEY` in `.env` | Highest |
| Mistral (free tier) | `MISTRAL_API_KEY` in `.env` | High |
| Rule-based (regex) | No setup needed | Good for testing |

---

## MLflow Tracking

Every training run logs:
- Hyperparameters
- F1 per class (Mild / Moderate / Severe)
- AUC-ROC
- Cross-validation scores
- Confusion matrix (artifact)
- Feature importance plot (artifact)
- Classification report (artifact)

View with: `mlflow ui`

---

## Docker

```bash
# Build image
docker build -t adverse-event-intelligence .

# Run app
docker run -p 8501:8501 adverse-event-intelligence

# Run tests in container
docker run adverse-event-intelligence pytest tests/ -v
```

---

## CI/CD (GitHub Actions)

On every push:
1. **Test** — runs full pytest suite
2. **Build** — builds and smoke-tests Docker image
3. **Train** (main branch only) — retrains model, fails if F1 < 0.50

---

## Data Source

**Synthetic data** is generated by default — mirrors real FAERS schema.

**Real FAERS data:**
1. Download from: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
2. Unzip into `data/faers_real/`
3. Set `use_synthetic=False` in `pipeline.py`

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | scikit-learn, XGBoost |
| LLM | OpenAI, Mistral, LangChain |
| MLOps | MLflow |
| Data | Pandas, NumPy |
| App | Streamlit, Plotly |
| DevOps | Docker, GitHub Actions |
| Testing | pytest |

---

## Disclaimer

This is a research prototype. All predictions should be reviewed by qualified clinical staff before any clinical decision is made.

---

*Built as a demonstration of end-to-end ML pipeline skills for clinical trial applications.*
