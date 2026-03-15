"""
test_pipeline.py
----------------
Unit and integration tests for the adverse event pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Load a small synthetic dataset for testing."""
    from ingestion import generate_synthetic_faers
    return generate_synthetic_faers(n_samples=200, save=False)


@pytest.fixture
def sample_features(sample_df):
    """Get features from sample data."""
    from features import get_features_and_target
    X, y, feature_names = get_features_and_target(sample_df, run_selection=False)
    return X, y, feature_names


# ─── Ingestion Tests ──────────────────────────────────────────────────────────
class TestIngestion:

    def test_synthetic_data_shape(self, sample_df):
        """Synthetic data should have the right number of rows and columns."""
        assert len(sample_df) == 200
        assert "severity" in sample_df.columns
        assert "clinical_note" in sample_df.columns

    def test_severity_distribution(self, sample_df):
        """All three severity classes should be present."""
        severities = sample_df["severity"].unique()
        assert "Mild" in severities
        assert "Moderate" in severities
        assert "Severe" in severities

    def test_no_null_report_ids(self, sample_df):
        """Report IDs must be unique and non-null."""
        assert sample_df["report_id"].isnull().sum() == 0
        assert sample_df["report_id"].nunique() == len(sample_df)

    def test_age_range(self, sample_df):
        """Ages should be within realistic range."""
        assert sample_df["age"].min() >= 18
        assert sample_df["age"].max() <= 90

    def test_clinical_notes_not_empty(self, sample_df):
        """All rows should have a clinical note."""
        assert sample_df["clinical_note"].isnull().sum() == 0
        assert (sample_df["clinical_note"].str.len() > 0).all()


# ─── Feature Engineering Tests ────────────────────────────────────────────────
class TestFeatures:

    def test_clean_data_removes_duplicates(self, sample_df):
        """Duplicate report IDs should be removed."""
        from features import clean_data
        # Add a duplicate
        df_with_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        cleaned = clean_data(df_with_dup)
        assert len(cleaned) == len(sample_df)

    def test_encode_categoricals_no_nulls(self, sample_df):
        """Encoded columns should have no nulls."""
        from features import clean_data, encode_categoricals
        df = clean_data(sample_df)
        df = encode_categoricals(df)
        assert df["gender_encoded"].isnull().sum() == 0
        assert df["severity_encoded"].isnull().sum() == 0

    def test_engineer_features_adds_columns(self, sample_df):
        """Feature engineering should add expected derived columns."""
        from features import clean_data, encode_categoricals, engineer_features
        df = clean_data(sample_df)
        df = encode_categoricals(df)
        df = engineer_features(df)
        for col in ["is_serious_ae", "elderly_flag", "high_dose_flag", "risk_score", "age_group"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_risk_score_is_positive(self, sample_df):
        """Risk scores should be non-negative."""
        from features import clean_data, encode_categoricals, engineer_features
        df = clean_data(sample_df)
        df = encode_categoricals(df)
        df = engineer_features(df)
        assert (df["risk_score"] >= 0).all()

    def test_feature_matrix_no_nulls(self, sample_features):
        """Feature matrix should have no nulls."""
        X, y, _ = sample_features
        assert X.isnull().sum().sum() == 0

    def test_target_values(self, sample_features):
        """Target should only contain 0, 1, 2."""
        _, y, _ = sample_features
        assert set(y.unique()).issubset({0, 1, 2})


# ─── LLM Extractor Tests ──────────────────────────────────────────────────────
class TestLLMExtractor:

    def test_rule_based_extracts_age(self):
        """Rule-based extractor should find age from text."""
        from llm_extractor import extract_with_rules
        note = "68 year old male patient reported chest pain"
        result = extract_with_rules(note)
        assert result["age"] == 68

    def test_rule_based_extracts_gender(self):
        """Rule-based extractor should detect gender."""
        from llm_extractor import extract_with_rules

        male_note = "45 year old male patient"
        female_note = "35 year old female patient"

        assert extract_with_rules(male_note)["gender"] == "Male"
        assert extract_with_rules(female_note)["gender"] == "Female"

    def test_rule_based_extracts_dosage(self):
        """Rule-based extractor should find dosage."""
        from llm_extractor import extract_with_rules
        note = "patient took 100mg of DrugX"
        result = extract_with_rules(note)
        assert result["dosage_mg"] == 100.0

    def test_fill_defaults_completes_missing(self):
        """fill_defaults should fill all missing fields."""
        from llm_extractor import fill_defaults
        partial = {"age": 55, "gender": "Male"}
        filled = fill_defaults(partial)
        assert "dosage_mg" in filled
        assert "route" in filled
        assert "symptom_count" in filled

    def test_note_to_features_returns_dict(self):
        """note_to_features should always return a dict."""
        from llm_extractor import note_to_features
        note = "Patient experienced severe nausea after taking medication"
        result = note_to_features(note)
        assert isinstance(result, dict)
        assert "age" in result
        assert "severity_indicators" in result

    def test_serious_ae_flagged(self):
        """Serious adverse events should be flagged."""
        from llm_extractor import note_to_features
        note = "Patient experienced chest pain and anaphylaxis after dosing"
        result = note_to_features(note)
        assert result["is_serious_ae"] == 1

    def test_mild_ae_not_flagged(self):
        """Non-serious adverse events should not be flagged."""
        from llm_extractor import note_to_features
        note = "Patient reported mild headache after taking medication"
        result = note_to_features(note)
        assert result["is_serious_ae"] == 0


# ─── Model Tests ──────────────────────────────────────────────────────────────
class TestModels:

    def test_logistic_regression_trains(self, sample_features):
        """Logistic regression should train and predict."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1, 2})

    def test_xgboost_trains(self, sample_features):
        """XGBoost should train and predict."""
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split

        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBClassifier(
            n_estimators=50, eval_metric="mlogloss", random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_model_probabilities_sum_to_one(self, sample_features):
        """Predicted probabilities should sum to 1."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_f1_score_above_baseline(self, sample_features):
        """XGBoost F1 should be above random baseline (0.33)."""
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBClassifier(n_estimators=50, eval_metric="mlogloss", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        assert f1 > 0.33, f"F1 {f1:.3f} should be above random baseline 0.33"


# ─── Integration Test ─────────────────────────────────────────────────────────
class TestIntegration:

    def test_full_pipeline_runs(self):
        """Full pipeline should run without errors on small dataset."""
        from ingestion import generate_synthetic_faers
        from features import get_features_and_target
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score

        # Generate small dataset
        df = generate_synthetic_faers(n_samples=300, save=False)
        X, y, feature_names = get_features_and_target(df, run_selection=False)

        # Quick train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        assert f1 > 0.0
        assert len(preds) == len(y_test)
