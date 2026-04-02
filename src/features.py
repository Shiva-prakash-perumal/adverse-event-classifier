"""
features.py
-----------
Feature engineering and feature selection for the adverse event classifier.

FIXED VERSION - Data leakage removed:
- Feature selection now happens only on training data
- Imputation values (median) fitted only on training data
- high_dose_flag median calculated only on training data

Steps:
1. Clean raw data (without leaking statistics)
2. Encode categorical variables
3. Engineer derived features (separate fit/transform pattern)
4. Select best features using Mutual Information + RFE (training data only)
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
from ingestion import load_data
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
# Includes both synthetic data terms AND real FAERS MedDRA preferred terms
# MedDRA terms use British spelling and standardized nomenclature
SERIOUS_AES = [
    # Synthetic data terms (title case)
    "Chest Pain", "Anaphylaxis", "Cardiac Arrest",
    "Liver Failure", "Dyspnea",

    # Real FAERS MedDRA preferred terms (lowercase)
    "chest pain", "anaphylactic reaction", "anaphylactic shock",
    "cardiac arrest", "cardio-respiratory arrest",
    "hepatic failure", "acute hepatic failure",
    "dyspnoea", "acute respiratory failure", "respiratory failure",
    "respiratory arrest", "death", "sudden death",
    "ventricular fibrillation", "ventricular tachycardia",
    "pulmonary embolism", "cerebrovascular accident",
    "myocardial infarction", "acute myocardial infarction",
    "septic shock", "sepsis", "multi-organ failure",
    "loss of consciousness", "coma", "convulsion"
]


class FeatureTransformer:
    """
    Stateful transformer for feature engineering.
    Learns statistics from training data, applies to test data.
    Prevents data leakage.
    """
    def __init__(self):
        self.age_median = None
        self.weight_median = None
        self.dosage_median = None
        self.high_dose_threshold = None
        
    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        """Learn statistics from training data only."""
        self.age_median = df["age"].median()
        self.weight_median = df["weight_kg"].median()
        self.dosage_median = df["dosage_mg"].median()
        
        # Calculate high dose threshold (median) from training data only
        self.high_dose_threshold = df["dosage_mg"].median()
        
        logger.info(f"Fitted transformer: age_median={self.age_median:.1f}, "
                   f"weight_median={self.weight_median:.1f}, "
                   f"dosage_median={self.dosage_median:.1f}")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations using fitted statistics."""
        if self.age_median is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Handle missing values using fitted statistics
        df["age"] = df["age"].fillna(self.age_median)
        df["weight_kg"] = df["weight_kg"].fillna(self.weight_median)
        df["dosage_mg"] = df["dosage_mg"].fillna(self.dosage_median)
        
        # High dose flag using fitted threshold
        df["high_dose_flag"] = (df["dosage_mg"].fillna(0) > self.high_dose_threshold).astype(int)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (for training data)."""
        self.fit(df)
        return self.transform(df)


def clean_data(df: pd.DataFrame, is_train: bool = True, transformer: FeatureTransformer = None) -> tuple:
    """
    Clean raw data — handle missing values, fix types, remove duplicates.
    
    Parameters:
        df: Raw DataFrame
        is_train: If True, fit transformer. If False, use provided transformer.
        transformer: Fitted transformer (required if is_train=False)
    
    Returns:
        (cleaned_df, transformer)
    """
    logger.info("Cleaning data...")
    df = df.copy()

    # Remove rows with missing target
    df = df.dropna(subset=["severity"])

    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=["report_id"])
    logger.info(f"Removed {initial_len - len(df)} duplicates")

    # Fill categorical missing values (no leakage risk)
    df["gender"] = df["gender"].fillna("Unknown")
    df["route"] = df["route"].fillna("Unknown")

    # Clip outliers (no leakage - uses domain knowledge, not data statistics)
    df["age"] = df["age"].clip(0, 120)
    df["weight_kg"] = df["weight_kg"].clip(20, 300)
    df["dosage_mg"] = df["dosage_mg"].clip(0, 10000)

    # Handle numeric missing values with transformer (prevents leakage)
    if is_train:
        transformer = FeatureTransformer()
        df = transformer.fit_transform(df)
    else:
        if transformer is None:
            raise ValueError("transformer required when is_train=False")
        df = transformer.transform(df)

    logger.info(f"Cleaned data shape: {df.shape}")
    return df, transformer


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables into numeric form for ML models.
    No data leakage - uses fixed mappings.
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
    
    NOTE: high_dose_flag is now created in the FeatureTransformer to prevent leakage.
    
    Domain knowledge applied:
    - Elderly patients (>65) have higher adverse event risk
    - Serious adverse events (chest pain, anaphylaxis etc.) skew severe
    - Composite risk score combines multiple factors
    """
    logger.info("Engineering features...")
    df = df.copy()

    # Flag: Is the adverse event clinically serious?
    serious_lower = [ae.lower() for ae in SERIOUS_AES]
    df["is_serious_ae"] = df["adverse_event"].str.lower().isin(serious_lower).astype(int)

    # Age groups (clinical standard bucketing)
    # Handle null ages by assigning to "Unknown" group (3)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 65, 120],
        labels=[0, 1, 2, 3]
    ).astype("Int64")
    
    # Fill NaN age groups with a separate category instead of assuming young adult
    df["age_group"] = df["age_group"].fillna(4).astype(int)  # 4 = unknown age group

    # Elderly flag
    df["elderly_flag"] = (df["age"].fillna(0) > 65).astype(int)

    # Composite risk score
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
    
    Only call this on TRAINING data to prevent leakage.

    Mutual Information measures how much knowing a feature reduces
    uncertainty about the target — captures both linear AND non-linear
    relationships (better than simple correlation for clinical data).

    Parameters:
        X: Feature DataFrame (TRAINING DATA ONLY)
        y: Target Series (TRAINING DATA ONLY)
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
    
    Only call this on TRAINING data to prevent leakage.

    RFE trains a model, ranks features by importance, removes the least
    important, retrains, repeats — finds the optimal subset.

    Parameters:
        X: Feature DataFrame (pre-filtered by MI, TRAINING DATA ONLY)
        y: Target Series (TRAINING DATA ONLY)
        n_features: Final number of features to keep

    Returns:
        List of selected feature names
    """
    logger.info("Running RFE feature selection...")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = LogisticRegression(
        max_iter=5000,
        random_state=42,
        solver="saga",
        n_jobs=-1
    )
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_scaled, y)

    selected = X.columns[rfe.support_].tolist()
    logger.info(f"RFE selected {len(selected)} features: {selected}")

    return selected


def get_features_and_target(
    df: pd.DataFrame,
    is_train: bool = True,
    transformer: FeatureTransformer = None
) -> tuple:
    """
    Full feature pipeline: clean → encode → engineer.
    
    CHANGE: Feature selection removed from here.
    Feature selection must happen AFTER train/test split to prevent leakage.
    
    Parameters:
        df: Raw DataFrame from ingestion
        is_train: If True, fit transformer. If False, use provided transformer.
        transformer: Fitted transformer (required if is_train=False)

    Returns:
        X (features), y (target), transformer
    """
    df_clean, transformer = clean_data(df, is_train=is_train, transformer=transformer)
    df_encoded = encode_categoricals(df_clean)
    df_engineered = engineer_features(df_encoded)

    # Extract all engineered features
    available_features = [f for f in FEATURE_COLS if f in df_engineered.columns]
    X = df_engineered[available_features].copy()
    y = df_engineered[TARGET_COL].copy()

    logger.info(f"Extracted {len(available_features)} features")
    
    return X, y, transformer


def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit and return a standard scaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return scaler


if __name__ == "__main__":
    

    df = load_data()
    
    X, y, transformer = get_features_and_target(df, is_train=True)

    print(f"\nFeature set: {X.columns.tolist()}")
    print(f"X shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
    print(f"\nSample features:\n{X.head()}")