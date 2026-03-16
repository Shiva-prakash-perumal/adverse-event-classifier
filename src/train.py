"""
train.py
--------
Trains three classifiers and tracks all experiments with MLflow.

Models:
1. Logistic Regression — interpretable baseline
2. Random Forest       — robust, handles non-linearity
3. XGBoost             — best accuracy, production-grade

Every run is tracked in MLflow:
- Parameters (hyperparameters)
- Metrics (F1, AUC, precision, recall)
- Artifacts (model files, confusion matrices)
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
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

    Why these three?
    - Logistic Regression: Fast, interpretable, good baseline.
                           Easy to explain to clinical stakeholders.
    - Random Forest: Handles non-linearity, robust to outliers,
                     built-in feature importance.
    - XGBoost: State-of-the-art on tabular data, handles missing values,
               best accuracy. Production model of choice.
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=1000,
                C=1.0,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42
            ),
            "params": {"C": 1.0, "solver": "lbfgs", "multi_class": "multinomial"},
            "scale": True  # Logistic Regression needs scaled features
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5
            },
            "scale": False
        },
        "xgboost": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8
            },
            "scale": False
        }
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: list = ["Mild", "Moderate", "Severe"]
) -> str:
    """Plot and save confusion matrix, return file path."""
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
    Train a single model, evaluate it, log everything to MLflow.

    Parameters:
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
        model_name: Name for logging
        model_config: Dict with model, params, scale flag
        feature_names: List of feature column names

    Returns:
        Dict of evaluation metrics
    """
    model = model_config["model"]
    params = model_config["params"]
    needs_scaling = model_config["scale"]

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

        # Train
        logger.info(f"Training {model_name}...")
        model.fit(X_train_final, y_train)

        # Predict
        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final) if hasattr(model, "predict_proba") else None

        # Metrics
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

        # Per-class F1 (important — don't just look at macro average)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        f1_mild, f1_moderate, f1_severe = f1_per_class

        # AUC-ROC (multiclass OvR)
        auc = None
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
            except Exception:
                auc = None

        # Cross-validation for robustness estimate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_final, y_train, cv=cv, scoring="f1_macro")

        metrics = {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_mild": round(f1_mild, 4),
            "f1_moderate": round(f1_moderate, 4),
            "f1_severe": round(f1_severe, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }
        if auc:
            metrics["auc_roc"] = round(auc, 4)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model artifact
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        # Log feature importance
        fi_path = plot_feature_importance(model, feature_names, model_name)
        if fi_path:
            mlflow.log_artifact(fi_path)

        # Log classification report as text
        report = classification_report(
            y_test, y_pred,
            target_names=["Mild", "Moderate", "Severe"]
        )
        logger.info(f"\n{model_name} Classification Report:\n{report}")
        mlflow.log_text(report, "classification_report.txt")

        # Save model locally
        joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")
        logger.info(f"Saved {model_name} to {MODELS_DIR}")

        logger.info(f"{model_name} | F1 Macro: {f1_macro:.4f} | AUC: {auc}")

    return metrics


def run_grid_search(
    model,
    param_grid: dict,
    X_train,
    y_train,
    model_name: str,
    use_sample_weight: bool = False
) -> dict:
    """
    Generic GridSearchCV runner used by all three tuning functions.

    Parameters:
        model:              base estimator
        param_grid:         hyperparameter grid to search
        X_train:            training features
        y_train:            training labels
        model_name:         name for logging and saving results
        use_sample_weight:  True for XGBoost (doesn't support class_weight param)

    Returns:
        best_params dict
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",   # optimize for all 3 classes equally
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    if use_sample_weight:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search.fit(X_train, y_train)

    logger.info(f"{model_name} best params: {grid_search.best_params_}")
    logger.info(f"{model_name} best CV F1 Macro: {grid_search.best_score_:.4f}")

    # Save full results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("mean_test_score", ascending=False)
    results_df.to_csv(MODELS_DIR / f"{model_name}_tuning_results.csv", index=False)
    logger.info(f"Tuning results saved to models/{model_name}_tuning_results.csv")

    return grid_search.best_params_


def tune_logistic_regression(X_train, y_train) -> dict:
    """
    Tune Logistic Regression hyperparameters.

    Key params:
    - C:       inverse regularization strength (lower = more regularization)
    - solver:  algorithm for optimization (saga best for large imbalanced data)
    - max_iter: increase for convergence on large datasets
    """
    logger.info("\nTuning Logistic Regression...")

    param_grid = {
        "C":        [0.01, 0.1, 1.0, 10.0],
        "solver":   ["saga"],
        "max_iter": [1000, 3000, 5000],
    }

    # Scale features first — LR needs standardized data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    base_model = LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return run_grid_search(
        base_model, param_grid,
        X_scaled, y_train,
        model_name="logistic_regression",
        use_sample_weight=False
    )


def tune_random_forest(X_train, y_train) -> dict:
    """
    Tune Random Forest hyperparameters.

    Key params:
    - n_estimators:    more trees = more stable but slower
    - max_depth:       controls overfitting
    - min_samples_leaf: higher = smoother model, better on rare classes
    - max_features:    fraction of features per split
    """
    logger.info("\nTuning Random Forest...")

    # param_grid = {
    #     "n_estimators":    [100, 200, 300],
    #     "max_depth":       [8, 10, 15, None],
    #     "min_samples_leaf": [1, 5, 10],   # key for Moderate class
    #     "max_features":    ["sqrt", "log2"],
    # }

    param_grid = {
        "n_estimators":     [100, 200],       # removed 300
        "max_depth":        [10, 15],          # removed 8 and None
        "min_samples_leaf": [1, 5],            # removed 10
        "max_features":     ["sqrt"],          # removed log2
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return run_grid_search(
        base_model, param_grid,
        X_train.values, y_train,
        model_name="random_forest",
        use_sample_weight=False
    )


def tune_xgboost(X_train, y_train) -> dict:
    """
    Tune XGBoost hyperparameters.

    Key params:
    - min_child_weight: higher = harder to split rare Moderate nodes
    - gamma:            min loss reduction required to make a split
    - max_depth:        tree complexity
    - learning_rate:    step size shrinkage
    - n_estimators:     number of trees
    """
    logger.info("\nTuning XGBoost... (may take 30-60 mins on large data)")

    param_grid = {
        "n_estimators":     [300, 500],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.01, 0.05],
        "min_child_weight": [1, 5, 10],
        "gamma":            [0, 0.5, 1.0],
        "subsample":        [0.8, 0.9],
    }

    base_model = XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    return run_grid_search(
        base_model, param_grid,
        X_train.values, y_train,
        model_name="xgboost",
        use_sample_weight=True   # XGBoost needs sample_weight not class_weight
    )


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    test_size: float = 0.2
) -> pd.DataFrame:
    """
    Train all three models, compare results, save the best one.

    Parameters:
        X: Features
        y: Target
        feature_names: Feature column names
        test_size: Test split fraction

    Returns:
        DataFrame comparing all model metrics
    """
    # Train/test split — stratified to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    logger.info(f"Train severity distribution:\n{pd.Series(y_train).value_counts()}")

    # ── Tune all three models before full training ───────────────────────────
    logger.info("\n" + "="*50)
    logger.info("HYPERPARAMETER TUNING — ALL MODELS")
    logger.info("="*50)

    logger.info("\nStep 1/3: Tuning Logistic Regression...")
    best_lr_params = tune_logistic_regression(X_train, y_train)

    logger.info("\nStep 2/3: Tuning Random Forest...")
    best_rf_params = tune_random_forest(X_train, y_train)

    logger.info("\nStep 3/3: Tuning XGBoost...")
    best_xgb_params = tune_xgboost(X_train, y_train)

    logger.info("\nAll tuning complete. Best params found:")
    logger.info(f"  Logistic Regression: {best_lr_params}")
    logger.info(f"  Random Forest:       {best_rf_params}")
    logger.info(f"  XGBoost:             {best_xgb_params}")

    # Apply best params to each model before training
    models = get_models()
    models["logistic_regression"]["model"].set_params(**best_lr_params)
    models["logistic_regression"]["params"].update(best_lr_params)
    models["random_forest"]["model"].set_params(**best_rf_params)
    models["random_forest"]["params"].update(best_rf_params)
    models["xgboost"]["model"].set_params(**best_xgb_params)
    models["xgboost"]["params"].update(best_xgb_params)

    # Override min_child_weight — tuning found 1 is best overall F1
    # but min_child_weight=1 hurts Moderate class detection
    # (too small leaf nodes → unstable splits on rare class)
    # min_child_weight=5 forces more stable splits → better Moderate recall
    models["xgboost"]["model"].set_params(min_child_weight=5)
    models["xgboost"]["params"]["min_child_weight"] = 5
    logger.info("Overriding min_child_weight=5 to improve Moderate class detection")

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
    logger.info(f"\n{results_df[['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']].to_string()}")

    # Save best model as "production" model
    best_model_name = results_df["f1_macro"].idxmax()
    logger.info(f"\nBest model: {best_model_name} (F1 Macro: {results_df.loc[best_model_name, 'f1_macro']:.4f})")

    # Copy best model to production path
    import shutil
    best_model_path = MODELS_DIR / f"{best_model_name}.pkl"
    prod_model_path = MODELS_DIR / "production_model.pkl"
    shutil.copy(best_model_path, prod_model_path)

    # Save feature names for inference
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

    df = load_data(use_synthetic=True)
    X, y, feature_names = get_features_and_target(df)

    results = train_all_models(X, y, feature_names)
    print(f"\nFinal Results:\n{results}")
