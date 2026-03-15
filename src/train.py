"""
train.py
--------
Trains three classifiers with class balancing and tracks all experiments with MLflow.

Class Imbalance Fix:
- Logistic Regression: class_weight="balanced"
- Random Forest:       class_weight="balanced"
- XGBoost:             sample_weight=compute_sample_weight("balanced")

Models:
1. Logistic Regression — interpretable baseline
2. Random Forest       — robust, handles non-linearity
3. XGBoost             — best accuracy, production-grade
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "adverse_event_classifier")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def get_models() -> dict:
    """
    Define the three models to train and compare.

    Class imbalance fix applied to each:

    Logistic Regression → class_weight="balanced"
        sklearn automatically adjusts weights inversely proportional
        to class frequency: weight = n_samples / (n_classes * n_samples_per_class)

    Random Forest → class_weight="balanced"
        Same automatic balancing at each tree level

    XGBoost → sample_weight passed at fit() time
        XGBoost doesn't have class_weight param — instead we pass
        per-sample weights computed by compute_sample_weight("balanced")
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=5000,
                C=1.0,
                multi_class="multinomial",
                solver="saga",
                class_weight="balanced",   # ← class imbalance fix
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "C": 1.0,
                "solver": "saga",
                "class_weight": "balanced"
            },
            "scale": True,
            "use_sample_weight": False
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",   # ← class imbalance fix
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "class_weight": "balanced"
            },
            "scale": False,
            "use_sample_weight": False
        },
        "xgboost": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
                
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "sample_weight": "balanced"  # logged for reference
            },
            "scale": False,
            "use_sample_weight": True   # ← signals fit() to use sample weights
        }
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: list = ["Mild", "Moderate", "Severe"]
) -> str:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()
    path = MODELS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str
) -> str:
    """Plot and save feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()
    path = MODELS_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    model_config: dict,
    feature_names: list
) -> dict:
    """
    Train a single model with class balancing, evaluate it,
    and log everything to MLflow.
    """
    model = model_config["model"]
    params = model_config["params"]
    needs_scaling = model_config["scale"]
    use_sample_weight = model_config["use_sample_weight"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):

        # Scale features if needed (Logistic Regression)
        if needs_scaling:
            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train)
            X_test_final = scaler.transform(X_test)
            joblib.dump(scaler, MODELS_DIR / f"scaler_{model_name}.pkl")
        else:
            X_train_final = X_train.values
            X_test_final = X_test.values

        # Compute sample weights for XGBoost
        # (Logistic Regression and Random Forest use class_weight="balanced" internally)
        if use_sample_weight:
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )
            logger.info(
                f"Sample weights computed for {model_name}:\n"
                f"  Class 0 (Mild) weight:     {sample_weights[y_train == 0].mean():.4f}\n"
                f"  Class 1 (Moderate) weight: {sample_weights[y_train == 1].mean():.4f}\n"
                f"  Class 2 (Severe) weight:   {sample_weights[y_train == 2].mean():.4f}"
            )

        # Train
        logger.info(f"Training {model_name} with class balancing...")
        if use_sample_weight:
            model.fit(X_train_final, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_final, y_train)

        # Predict
        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final) if hasattr(model, "predict_proba") else None

        # Metrics
        f1_macro     = f1_score(y_test, y_pred, average="macro")
        f1_weighted  = f1_score(y_test, y_pred, average="weighted")
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro    = recall_score(y_test, y_pred, average="macro", zero_division=0)

        # Per-class F1 — critical for checking Moderate is no longer 0
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        f1_mild, f1_moderate, f1_severe = f1_per_class

        logger.info(
            f"\n{model_name} Per-Class F1:\n"
            f"  Mild:     {f1_mild:.4f}\n"
            f"  Moderate: {f1_moderate:.4f}  ← was 0.0 before balancing\n"
            f"  Severe:   {f1_severe:.4f}"
        )

        # AUC-ROC
        auc = None
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(
                    y_test, y_pred_proba,
                    multi_class="ovr", average="macro"
                )
            except Exception:
                auc = None

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train_final, y_train,
            cv=cv, scoring="f1_macro"
        )

        metrics = {
            "f1_macro":        round(f1_macro, 4),
            "f1_weighted":     round(f1_weighted, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro":    round(recall_macro, 4),
            "f1_mild":         round(f1_mild, 4),
            "f1_moderate":     round(f1_moderate, 4),
            "f1_severe":       round(f1_severe, 4),
            "cv_f1_mean":      round(cv_scores.mean(), 4),
            "cv_f1_std":       round(cv_scores.std(), 4),
        }
        if auc:
            metrics["auc_roc"] = round(auc, 4)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Log artifacts
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        fi_path = plot_feature_importance(model, feature_names, model_name)
        if fi_path:
            mlflow.log_artifact(fi_path)

        report = classification_report(
            y_test, y_pred,
            target_names=["Mild", "Moderate", "Severe"]
        )
        logger.info(f"\n{model_name} Classification Report:\n{report}")
        mlflow.log_text(report, "classification_report.txt")

        # Save model locally
        joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")
        logger.info(f"Saved {model_name} to {MODELS_DIR}")

    return metrics


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    test_size: float = 0.2
) -> pd.DataFrame:
    """
    Train all three models with class balancing,
    compare results, save the best one.
    """
    # Stratified split — preserves class distribution in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    logger.info(f"Class distribution in train:\n{pd.Series(y_train).value_counts()}")

    models = get_models()
    results = {}

    for model_name, model_config in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name.upper()}")
        logger.info(f"{'='*50}")
        metrics = train_model(
            X_train, X_test, y_train, y_test,
            model_name, model_config, feature_names
        )
        results[model_name] = metrics

    # Compare all models
    results_df = pd.DataFrame(results).T
    logger.info(f"\n{'='*50}")
    logger.info("MODEL COMPARISON:")
    logger.info(
        f"\n{results_df[['f1_macro', 'f1_mild', 'f1_moderate', 'f1_severe', 'auc_roc']].to_string()}"
    )

    # Save best model
    best_model_name = results_df["f1_macro"].idxmax()
    logger.info(
        f"\nBest model: {best_model_name} "
        f"(F1 Macro: {results_df.loc[best_model_name, 'f1_macro']:.4f})"
    )

    import shutil
    shutil.copy(
        MODELS_DIR / f"{best_model_name}.pkl",
        MODELS_DIR / "production_model.pkl"
    )

    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(best_model_name, MODELS_DIR / "best_model_name.pkl")

    results_df.to_csv(MODELS_DIR / "model_comparison.csv")
    logger.info(f"Model comparison saved to {MODELS_DIR / 'model_comparison.csv'}")

    return results_df


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    from ingestion import load_data
    from features import get_features_and_target

    df = load_data()
    X, y, feature_names = get_features_and_target(df)
    results = train_all_models(X, y, feature_names)
    print(f"\nFinal Results:\n{results}")