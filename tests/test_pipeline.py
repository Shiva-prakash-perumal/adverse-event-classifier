"""
test_pipeline.py
----------------
Unit and integration tests for the adverse event pipeline.
Uses inline synthetic data generation — no real FAERS data needed for CI.

Run with: pytest tests/test_pipeline.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))


# =============================================================================
# SYNTHETIC DATA GENERATOR — inline for CI/CD
# Copied here so tests work without real FAERS data
# =============================================================================

def generate_synthetic_faers(n_samples: int = 200, save: bool = False) -> pd.DataFrame:
    """Generates synthetic FAERS-like data for testing."""
    np.random.seed(42)

    ages = np.random.normal(55, 18, n_samples).clip(18, 90).astype(int)
    genders = np.random.choice(["Male", "Female", "Unknown"], n_samples, p=[0.45, 0.45, 0.10])
    weights = np.random.normal(75, 18, n_samples).clip(40, 150).round(1)
    drug_names = np.random.choice(["DrugA", "DrugB", "DrugC", "DrugD", "DrugE"], n_samples)
    dosages = np.random.choice([10, 25, 50, 100, 200, 500], n_samples)
    routes = np.random.choice(
        ["Oral", "Intravenous", "Subcutaneous", "Topical"],
        n_samples, p=[0.60, 0.25, 0.10, 0.05]
    )
    adverse_events = np.random.choice(
        ["Nausea", "Headache", "Chest Pain", "Dyspnea", "Rash",
         "Fatigue", "Dizziness", "Vomiting", "Anaphylaxis",
         "Cardiac Arrest", "Liver Failure", "Hypertension", "Hypotension", "Fever"],
        n_samples
    )
    time_to_onset = np.random.exponential(scale=5, size=n_samples).clip(0.1, 90).round(1)
    num_concomitant_drugs = np.random.poisson(lam=2, size=n_samples).clip(0, 10)
    symptom_count = np.random.randint(1, 6, n_samples)
    has_comorbidity = np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    has_prior_reaction = np.random.choice([0, 1], n_samples, p=[0.80, 0.20])

    severity_score = (
        (ages > 65).astype(int) * 1.5
        + (dosages > 100).astype(int) * 1.2
        + (np.isin(adverse_events, ["Chest Pain", "Anaphylaxis", "Cardiac Arrest",
                                     "Liver Failure", "Dyspnea"])).astype(int) * 2.5
        + has_comorbidity * 1.0
        + has_prior_reaction * 0.8
        + np.random.normal(0, 1, n_samples)
    )
    severity = pd.cut(
        severity_score,
        bins=[-np.inf, 2.5, 5.0, np.inf],
        labels=["Mild", "Moderate", "Severe"]
    )
    notes = [
        f"{ages[i]} year old {genders[i].lower()} patient reported "
        f"{adverse_events[i].lower()} approximately {time_to_onset[i]} days "
        f"after taking {dosages[i]}mg of {drug_names[i]} via {routes[i].lower()} route."
        for i in range(n_samples)
    ]

    return pd.DataFrame({
        "report_id":             range(1, n_samples + 1),
        "age":                   ages,
        "gender":                genders,
        "weight_kg":             weights,
        "drug_name":             drug_names,
        "dosage_mg":             dosages,
        "route":                 routes,
        "adverse_event":         adverse_events,
        "time_to_onset_days":    time_to_onset,
        "num_concomitant_drugs": num_concomitant_drugs,
        "symptom_count":         symptom_count,
        "has_comorbidity":       has_comorbidity,
        "has_prior_reaction":    has_prior_reaction,
        "severity":              severity,
        "clinical_note":         notes
    })


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    return generate_synthetic_faers(n_samples=200, save=False)


@pytest.fixture
def sample_features(sample_df):
    from features import get_features_and_target
    X, y, feature_names = get_features_and_target(sample_df, run_selection=False)
    return X, y, feature_names


# ─── Ingestion Tests ──────────────────────────────────────────────────────────
class TestIngestion:

    def test_synthetic_data_shape(self, sample_df):
        assert len(sample_df) == 200
        assert "severity" in sample_df.columns
        assert "clinical_note" in sample_df.columns

    def test_severity_distribution(self, sample_df):
        severities = sample_df["severity"].unique()
        assert "Mild" in severities
        assert "Moderate" in severities
        assert "Severe" in severities

    def test_no_null_report_ids(self, sample_df):
        assert sample_df["report_id"].isnull().sum() == 0
        assert sample_df["report_id"].nunique() == len(sample_df)

    def test_age_range(self, sample_df):
        assert sample_df["age"].min() >= 18
        assert sample_df["age"].max() <= 90

    def test_clinical_notes_not_empty(self, sample_df):
        assert sample_df["clinical_note"].isnull().sum() == 0
        assert (sample_df["clinical_note"].str.len() > 0).all()


# ─── Feature Engineering Tests ────────────────────────────────────────────────
class TestFeatures:

    def test_clean_data_removes_duplicates(self, sample_df):
        from features import clean_data
        df_with_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        cleaned = clean_data(df_with_dup)
        assert len(cleaned) == len(sample_df)

    def test_encode_categoricals_no_nulls(self, sample_df):
        from features import clean_data, encode_categoricals
        df = clean_data(sample_df)
        df = encode_categoricals(df)
        assert df["gender_encoded"].isnull().sum() == 0
        assert df["severity_encoded"].isnull().sum() == 0

    def test_engineer_features_adds_columns(self, sample_df):
        from features import clean_data, encode_categoricals, engineer_features
        df = clean_data(sample_df)
        df = encode_categoricals(df)
        df = engineer_features(df)
        for col in ["is_serious_ae", "elderly_flag", "high_dose_flag", "risk_score", "age_group"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_risk_score_is_positive(self, sample_df):
        from features import clean_data, encode_categoricals, engineer_features
        df = clean_data(sample_df)
        df = encode_categoricals(df)
        df = engineer_features(df)
        assert (df["risk_score"] >= 0).all()

    def test_feature_matrix_no_nulls(self, sample_features):
        X, y, _ = sample_features
        assert X.isnull().sum().sum() == 0

    def test_target_values(self, sample_features):
        _, y, _ = sample_features
        assert set(y.unique()).issubset({0, 1, 2})


# ─── LLM Extractor Tests ──────────────────────────────────────────────────────
class TestLLMExtractor:

    def test_rule_based_extracts_age(self):
        from llm_extractor import extract_with_rules
        note = "68 year old male patient reported chest pain"
        result = extract_with_rules(note)
        assert result["age"] == 68

    def test_rule_based_extracts_gender(self):
        from llm_extractor import extract_with_rules
        assert extract_with_rules("45 year old male patient")["gender"] == "Male"
        assert extract_with_rules("35 year old female patient")["gender"] == "Female"

    def test_rule_based_extracts_dosage(self):
        from llm_extractor import extract_with_rules
        result = extract_with_rules("patient took 100mg of DrugX")
        assert result["dosage_mg"] == 100.0

    def test_fill_defaults_completes_missing(self):
        from llm_extractor import fill_defaults
        partial = {"age": 55, "gender": "Male"}
        filled = fill_defaults(partial)
        assert "dosage_mg" in filled
        assert "route" in filled
        assert "symptom_count" in filled

    def test_note_to_features_returns_dict(self):
        from llm_extractor import note_to_features
        note = "Patient experienced severe nausea after taking medication"
        result = note_to_features(note)
        assert isinstance(result, dict)
        assert "age" in result
        assert "severity_indicators" in result

    def test_serious_ae_flagged(self):
        from llm_extractor import note_to_features
        note = "Patient experienced chest pain and anaphylaxis after dosing"
        result = note_to_features(note)
        assert result["is_serious_ae"] == 1

    def test_mild_ae_not_flagged(self):
        from llm_extractor import note_to_features
        note = "Patient reported mild headache after taking medication"
        result = note_to_features(note)
        assert result["is_serious_ae"] == 0


# ─── Model Tests ──────────────────────────────────────────────────────────────
class TestModels:

    def test_logistic_regression_trains(self, sample_features):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1, 2})

    def test_xgboost_trains(self, sample_features):
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=50, eval_metric="mlogloss", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_model_probabilities_sum_to_one(self, sample_features):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_f1_score_above_baseline(self, sample_features):
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        X, y, _ = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=50, eval_metric="mlogloss", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        assert f1 > 0.33, f"F1 {f1:.3f} should be above random baseline 0.33"


# ─── Integration Test ─────────────────────────────────────────────────────────
class TestIntegration:

    def test_full_pipeline_runs(self):
        from features import get_features_and_target
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score

        df = generate_synthetic_faers(n_samples=300, save=False)
        X, y, feature_names = get_features_and_target(df, run_selection=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        assert f1 > 0.0
        assert len(preds) == len(y_test)