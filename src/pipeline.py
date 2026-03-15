"""
pipeline.py
-----------
End-to-end pipeline orchestrator.
Ties together: ingestion → features → training → evaluation

Run this to train and evaluate all models in one command:
    python src/pipeline.py
"""

import sys
import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))

from ingestion import load_data
from features import get_features_and_target
from train import train_all_models
from evaluate import full_evaluation, load_production_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_pipeline(use_synthetic: bool = True):
    """
    Run the complete ML pipeline from data ingestion to evaluation.

    Steps:
    1. Load data
    2. Feature engineering + selection
    3. Train all models with MLflow tracking
    4. Evaluate best model
    5. Save artifacts

    Parameters:
        use_synthetic: Use synthetic data (True) or real FAERS (False)
    """
    logger.info("="*60)
    logger.info("ADVERSE EVENT INTELLIGENCE PIPELINE STARTING")
    logger.info("="*60)

    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} records")

    # Step 2: Feature engineering + selection
    logger.info("\nStep 2: Feature engineering and selection...")
    X, y, feature_names = get_features_and_target(df, run_selection=True)
    logger.info(f"Final features: {feature_names}")

    # Step 3: Train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Train all models
    logger.info("\nStep 3: Training all models...")
    results_df = train_all_models(X, y, feature_names)
    logger.info(f"\nModel Comparison:\n{results_df}")

    # Step 5: Evaluate production model
    logger.info("\nStep 4: Evaluating production model...")
    model, feature_names_saved, model_name = load_production_model()
    eval_results = full_evaluation(model, X_test, y_test, model_name)

    # Save feature names for Streamlit app
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Best model: {model_name}")
    logger.info(f"Model artifacts saved to: {MODELS_DIR}")
    logger.info(f"MLflow runs saved to: mlruns/")
    logger.info("Run 'mlflow ui' to view experiment results")
    logger.info("="*60)

    return results_df, eval_results


def predict_single(note: str) -> dict:
    """
    Predict severity for a single clinical note.
    Used by the Streamlit app.

    Parameters:
        note: Raw clinical note text

    Returns:
        Dict with prediction, confidence, extracted fields
    """
    from llm_extractor import note_to_features, fill_defaults
    from features import encode_categoricals, engineer_features, SERIOUS_AES
    import numpy as np

    # Step 1: Extract structured fields from note
    extracted = note_to_features(note)

    # Step 2: Convert to DataFrame
    row = {
        "report_id": 0,
        "age": extracted.get("age", 55),
        "gender": extracted.get("gender", "Unknown"),
        "weight_kg": extracted.get("weight_kg", 75.0),
        "drug_name": extracted.get("drug_name", "Unknown"),
        "dosage_mg": extracted.get("dosage_mg", 50.0),
        "route": extracted.get("route", "Unknown"),
        "adverse_event": extracted.get("adverse_event", "Unknown"),
        "time_to_onset_days": extracted.get("time_to_onset_days", 3.0),
        "num_concomitant_drugs": extracted.get("num_concomitant_drugs", 0),
        "symptom_count": extracted.get("symptom_count", 1),
        "has_comorbidity": extracted.get("has_comorbidity", 0),
        "has_prior_reaction": extracted.get("has_prior_reaction", 0),
        "severity": "Mild"  # placeholder — overwritten by prediction
    }

    df_single = pd.DataFrame([row])

    # Step 3: Feature engineering
    df_single = encode_categoricals(df_single)
    df_single = engineer_features(df_single)

    # Step 4: Load model and feature names
    model = joblib.load(MODELS_DIR / "production_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

    # Step 5: Align features
    for col in feature_names:
        if col not in df_single.columns:
            df_single[col] = 0
    X_single = df_single[feature_names]

    # Step 6: Predict
    prediction_encoded = model.predict(X_single)[0]
    probas = model.predict_proba(X_single)[0]

    label_map = {0: "Mild", 1: "Moderate", 2: "Severe"}
    prediction_label = label_map[prediction_encoded]
    confidence = probas.max()

    return {
        "prediction": prediction_label,
        "confidence": round(float(confidence) * 100, 1),
        "prob_mild": round(float(probas[0]) * 100, 1),
        "prob_moderate": round(float(probas[1]) * 100, 1),
        "prob_severe": round(float(probas[2]) * 100, 1),
        "needs_review": confidence < 0.60,
        "extracted_fields": extracted
    }


if __name__ == "__main__":
    results_df, eval_results = run_pipeline(use_synthetic=True)
