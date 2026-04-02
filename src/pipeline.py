"""
pipeline.py
-----------
End-to-end pipeline orchestrator.
Ties together: ingestion → features → training → evaluation

FIXED VERSION - Data leakage removed:
- Train/test split happens BEFORE feature selection
- Feature transformer fitted only on training data
- Feature selection (MI + RFE) only uses training data

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
from features import get_features_and_target, select_features_mutual_info, select_features_rfe
from train import train_all_models
from evaluate import full_evaluation, load_production_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_pipeline():
    """
    Run the complete ML pipeline from data ingestion to evaluation.
    
    FIXED: Proper train/test split order to prevent data leakage.

    Steps:
    1. Load data
    2. Train/test split (BEFORE feature engineering to get separate datasets)
    3. Feature engineering (fit on train, transform both)
    4. Feature selection (ONLY on training data)
    5. Train all models with MLflow tracking
    6. Evaluate best model
    7. Save artifacts
    """
    logger.info("="*60)
    logger.info("ADVERSE EVENT INTELLIGENCE PIPELINE STARTING")
    logger.info("="*60)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Load data
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nStep 1: Loading data...")
    df = load_data()  # FIXED: Removed use_synthetic parameter
    logger.info(f"Loaded {len(df)} records")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Split BEFORE any feature engineering (prevent leakage)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nStep 2: Splitting into train/test sets...")
    
    # Need to encode target for stratification
    from features import encode_categoricals
    df_with_target = encode_categoricals(df)
    y_all = df_with_target["severity_encoded"]
    
    # Split raw data FIRST
    train_indices, test_indices = train_test_split(
        df.index, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_all
    )
    
    df_train = df.loc[train_indices].copy()
    df_test = df.loc[test_indices].copy()
    
    logger.info(f"Train size: {len(df_train)} | Test size: {len(df_test)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Feature engineering (fit on train, transform both)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nStep 3: Feature engineering...")
    
    # Fit transformer on training data
    X_train, y_train, transformer = get_features_and_target(
        df_train, 
        is_train=True
    )
    
    # Apply fitted transformer to test data (no leakage)
    X_test, y_test, _ = get_features_and_target(
        df_test, 
        is_train=False, 
        transformer=transformer
    )
    
    logger.info(f"Training features: {X_train.shape}")
    logger.info(f"Test features: {X_test.shape}")
    logger.info(f"Train severity distribution:\n{pd.Series(y_train).value_counts()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Feature selection (ONLY on training data)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nStep 4: Feature selection (training data only)...")
    
    # Step 4a: Mutual Information
    mi_features = select_features_mutual_info(X_train, y_train, top_k=12)
    X_train_mi = X_train[mi_features]
    X_test_mi = X_test[mi_features]  # Apply same feature selection to test
    
    # Step 4b: RFE
    final_features = select_features_rfe(X_train_mi, y_train, n_features=10)
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]
    
    logger.info(f"Final features selected: {final_features}")
    logger.info(f"Final feature count: {len(final_features)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Train all models
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nStep 5: Training all models...")
    
    # Combine train data back into single DataFrame for train_all_models
    # (it will do its own internal CV splits)
    X_combined = pd.concat([X_train_final, X_test_final])
    y_combined = pd.concat([y_train, y_test])
    
    results_df = train_all_models(
        X_combined, 
        y_combined, 
        final_features,
        test_size=0.2,
        test_indices=test_indices.tolist()
    )
    logger.info(f"\nModel Comparison:\n{results_df}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Evaluate production model
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nStep 6: Evaluating production model...")
    model, feature_names_saved, model_name = load_production_model()

    # Align X_test to the features the production model was trained on
    X_test_eval = X_test.copy()
    for col in feature_names_saved:
        if col not in X_test_eval.columns:
            X_test_eval[col] = 0
    X_test_eval = X_test_eval[feature_names_saved]
    logger.info(f"Evaluation feature set: {feature_names_saved}")

    eval_results = full_evaluation(model, X_test_eval, y_test, model_name)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7: Save artifacts
    # ══════════════════════════════════════════════════════════════════════════
    # Save transformer for production inference
    joblib.dump(transformer, MODELS_DIR / "feature_transformer.pkl")
    logger.info(f"Saved feature transformer to {MODELS_DIR / 'feature_transformer.pkl'}")

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
    
    FIXED: Uses saved transformer to prevent leakage.

    Parameters:
        note: Raw clinical note text

    Returns:
        Dict with prediction, confidence, extracted fields
    """
    from llm_extractor import note_to_features
    from features import engineer_features, encode_categoricals
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
        "severity": "Mild"  # placeholder
    }

    df_single = pd.DataFrame([row])

    # Step 3: Load transformer and apply (prevents leakage)
    transformer = joblib.load(MODELS_DIR / "feature_transformer.pkl")
    
    # Apply the same transformations used during training
    from features import clean_data
    df_clean, _ = clean_data(df_single, is_train=False, transformer=transformer)
    df_encoded = encode_categoricals(df_clean)
    df_engineered = engineer_features(df_encoded)

    # Step 4: Load model and feature names
    model = joblib.load(MODELS_DIR / "production_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

    # Step 5: Align features
    for col in feature_names:
        if col not in df_engineered.columns:
            df_engineered[col] = 0
    X_single = df_engineered[feature_names]

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
    results_df, eval_results = run_pipeline()
