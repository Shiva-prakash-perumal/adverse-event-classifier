"""
features.py
-----------
Feature engineering and feature selection for the adverse event classifier.

Steps:
1. Clean raw data
2. Encode categorical variables
3. Engineer derived features
4. Select best features using Mutual Information + RFE
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Features to use for modeling
FEATURE_COLS = [
    "age",
    "weight_kg",
    "dosage_mg",
    "num_concomitant_drugs",
    "symptom_count",
    "has_comorbidity",
    "has_prior_reaction",
    "gender_encoded",
    "route_encoded",
    "is_serious_ae",
    "age_group",
    "high_dose_flag",
    "elderly_flag",
    "risk_score"
]

TARGET_COL = "severity_encoded"

# Adverse events considered clinically serious
SERIOUS_AES = [
    "Chest Pain", "Anaphylaxis", "Cardiac Arrest",
    "Liver Failure", "Dyspnea"
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data — handle missing values, fix types, remove duplicates.
    """
    logger.info("Cleaning data...")
    df = df.copy()

    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=["report_id"])
    logger.info(f"Removed {initial_len - len(df)} duplicates")

    # Handle missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    df["dosage_mg"] = df["dosage_mg"].fillna(df["dosage_mg"].median())
    df["gender"] = df["gender"].fillna("Unknown")
    df["route"] = df["route"].fillna("Unknown")

    # Remove rows with missing target
    df = df.dropna(subset=["severity"])

    # Clip outliers
    df["age"] = df["age"].clip(0, 120)
    df["weight_kg"] = df["weight_kg"].clip(20, 300)
    df["dosage_mg"] = df["dosage_mg"].clip(0, 10000)

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables into numeric form for ML models.
    """
    logger.info("Encoding categorical variables...")
    df = df.copy()

    # Gender encoding
    gender_map = {"Male": 0, "Female": 1, "Unknown": 2}
    df["gender_encoded"] = df["gender"].map(gender_map).fillna(2)

    # Route of administration encoding
    route_map = {"Oral": 0, "Intravenous": 1, "Subcutaneous": 2, "Topical": 3, "Unknown": 4}
    df["route_encoded"] = df["route"].map(route_map).fillna(4)

    # Target encoding: Mild=0, Moderate=1, Severe=2
    severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
    df["severity_encoded"] = df["severity"].map(severity_map)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture clinical domain knowledge.

    Domain knowledge applied:
    - Elderly patients (>65) have higher adverse event risk
    - High doses increase severity risk
    - Serious adverse events (chest pain, anaphylaxis etc.) skew severe
    - Composite risk score combines multiple factors
    """
    logger.info("Engineering features...")
    df = df.copy()

    # Flag: Is the adverse event clinically serious?
    df["is_serious_ae"] = df["adverse_event"].isin(SERIOUS_AES).astype(int)

    # Age groups (clinical standard bucketing)
    # Use nullable Int64 + fillna before final astype(int)
    # Real FAERS data has null ages — pd.cut returns NaN for those rows
    # which crashes astype(int) directly
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 65, 120],
        labels=[0, 1, 2, 3]  # Pediatric, Young Adult, Adult, Elderly
    ).astype("Int64").fillna(1).astype(int)

    # High dose flag — fillna(0): unknown dosage treated as not high dose
    median_dose = df["dosage_mg"].median()
    df["high_dose_flag"] = (df["dosage_mg"].fillna(0) > median_dose).astype(int)

    # Elderly flag — fillna(0): unknown age treated as not elderly
    df["elderly_flag"] = (df["age"].fillna(0) > 65).astype(int)

    # Composite risk score — softened weights so individual features
    # can still contribute independently to the model.
    # Previous weights were too aggressive causing risk_score to dominate
    # at 42% feature importance, drowning out individual clinical signals.
    #
    # Old weights: elderly=2.0, high_dose=1.5, serious_ae=3.0
    # New weights: elderly=0.5, high_dose=0.4, serious_ae=0.8
    df["risk_score"] = (
        df["elderly_flag"] * 0.5
        + df["high_dose_flag"] * 0.4
        + df["is_serious_ae"] * 0.8
        + df["has_comorbidity"] * 0.4
        + df["has_prior_reaction"] * 0.3
        + df["num_concomitant_drugs"] * 0.1
    )

    logger.info(f"Features after engineering: {df.shape[1]} columns")
    return df


def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 12
) -> list:
    """
    Step 1 of feature selection: Mutual Information.

    Mutual Information measures how much knowing a feature reduces
    uncertainty about the target — captures both linear AND non-linear
    relationships (better than simple correlation for clinical data).

    Parameters:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to keep

    Returns:
        List of selected feature names
    """
    logger.info("Running Mutual Information feature selection...")

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False)

    logger.info(f"\nMutual Information Scores:\n{mi_df.to_string()}")

    # Keep top k features
    selected = mi_df.head(top_k)["feature"].tolist()
    logger.info(f"\nTop {top_k} features selected by MI: {selected}")

    return selected


def select_features_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10
) -> list:
    """
    Step 2 of feature selection: Recursive Feature Elimination (RFE).

    RFE trains a model, ranks features by importance, removes the least
    important, retrains, repeats — finds the optimal subset.

    Parameters:
        X: Feature DataFrame (pre-filtered by MI)
        y: Target Series
        n_features: Final number of features to keep

    Returns:
        List of selected feature names
    """
    logger.info("Running RFE feature selection...")

    # Scale features first — Logistic Regression converges much faster
    # on standardized data especially at 900k+ rows
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = LogisticRegression(
        max_iter=5000,      # increased from 1000
        random_state=42,
        solver="saga",      # saga handles large datasets better than lbfgs
        n_jobs=-1           # use all CPU cores
    )
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_scaled, y)

    selected = X.columns[rfe.support_].tolist()
    logger.info(f"RFE selected {len(selected)} features: {selected}")

    return selected


def get_features_and_target(
    df: pd.DataFrame,
    run_selection: bool = True
) -> tuple:
    """
    Full feature pipeline: clean → encode → engineer → select.

    Parameters:
        df: Raw DataFrame from ingestion
        run_selection: Whether to run MI + RFE selection

    Returns:
        X (features), y (target), selected_features (list)
    """
    df = clean_data(df)
    df = encode_categoricals(df)
    df = engineer_features(df)

    # Start with all engineered features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    logger.info(f"Starting with {len(available_features)} features")

    if run_selection:
        # Step 1: Mutual Information — broad filter
        mi_features = select_features_mutual_info(X, y, top_k=12)
        X_mi = X[mi_features]

        # Step 2: RFE — fine-tune the final subset
        final_features = select_features_rfe(X_mi, y, n_features=10)
        X_final = X[final_features]

        logger.info(f"Final feature count after selection: {len(final_features)}")
        return X_final, y, final_features
    else:
        return X, y, available_features


def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit and return a standard scaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return scaler


if __name__ == "__main__":
    from ingestion import load_data

    df = load_data(use_synthetic=True)
    X, y, features = get_features_and_target(df)

    print(f"\nFinal feature set: {features}")
    print(f"X shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
    print(f"\nSample features:\n{X.head()}")
    