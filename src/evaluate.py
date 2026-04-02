"""
evaluate.py - FIXED VERSION
-----------
Deep evaluation of trained models beyond basic accuracy.

Includes:
- Per-class precision/recall/F1
- Calibration curves (is the model's confidence trustworthy?)
- Abstain thresholds (when to send to human review)
- AUC-ROC curves
"""

import numpy as np
import pandas as pd
import joblib
import logging
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    brier_score_loss
)
from ingestion import load_data
from features import get_features_and_target
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
LABEL_MAP = {0: "Mild", 1: "Moderate", 2: "Severe"}
REVERSE_LABEL_MAP = {"Mild": 0, "Moderate": 1, "Severe": 2}


def load_production_model():
    """Load the best production model and associated artifacts."""
    model = joblib.load(MODELS_DIR / "production_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    best_model_name = joblib.load(MODELS_DIR / "best_model_name.pkl")
    logger.info(f"Loaded production model: {best_model_name}")
    return model, feature_names, best_model_name


def predict_with_confidence(
    model,
    X: pd.DataFrame,
    abstain_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Make predictions with confidence scores and abstain logic.

    Abstain threshold:
    If the model's max class probability is below the threshold,
    mark the prediction as "Review" — send to human reviewer.

    This is important in clinical settings where a wrong prediction
    has real consequences for patient safety.

    Parameters:
        model: Trained classifier
        X: Feature DataFrame
        abstain_threshold: Min confidence to make a prediction

    Returns:
        DataFrame with prediction, confidence, and review flag
    """
    probas = model.predict_proba(X)
    predictions = model.predict(X)
    max_confidence = probas.max(axis=1)

    results = pd.DataFrame({
        "prediction_encoded": predictions,
        "prediction_label": [LABEL_MAP[p] for p in predictions],
        "confidence": max_confidence.round(4),
        "prob_mild": probas[:, 0].round(4),
        "prob_moderate": probas[:, 1].round(4),
        "prob_severe": probas[:, 2].round(4),
        "needs_review": max_confidence < abstain_threshold
    })

    n_abstain = results["needs_review"].sum()
    logger.info(
        f"Predictions: {len(results)} total | "
        f"{n_abstain} flagged for human review ({n_abstain/len(results)*100:.1f}%)"
    )

    return results


def plot_roc_curves(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model"
) -> None:
    """
    Plot ROC curves for each class (One vs Rest).

    AUC-ROC tells you: if you randomly pick one positive
    and one negative example, how often does the model rank
    the positive higher? 1.0 = perfect, 0.5 = random.
    """
    probas = model.predict_proba(X_test)
    class_names = ["Mild", "Moderate", "Severe"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, class_name in enumerate(class_names):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, probas[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name.replace('_', ' ').title()}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"roc_curves_{model_name}.png", dpi=150)
    plt.close()
    logger.info(f"ROC curves saved")


def plot_calibration_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model"
) -> None:
    """
    Plot calibration curves.

    Calibration answers: when the model says 80% confidence,
    is it actually right 80% of the time?
    - Perfectly calibrated = diagonal line
    - Above diagonal = underconfident
    - Below diagonal = overconfident (dangerous in clinical settings!)
    """
    probas = model.predict_proba(X_test)
    class_names = ["Mild", "Moderate", "Severe"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        y_binary = (y_test == i).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_binary, probas[:, i], n_bins=10, strategy="uniform"
        )
        brier = brier_score_loss(y_binary, probas[:, i])

        ax.plot(prob_pred, prob_true, "s-", label=f"Model (Brier={brier:.3f})", color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration — {class_name}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Calibration Curves — {model_name.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"calibration_{model_name}.png", dpi=150)
    plt.close()
    logger.info("Calibration curves saved")


def evaluate_abstain_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate performance at different abstain thresholds.

    Shows the tradeoff: higher threshold = fewer predictions
    but higher accuracy on what IS predicted.
    Helps clinical teams choose the right operating point.
    """
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    probas = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    max_confidence = probas.max(axis=1)

    for threshold in thresholds:
        mask = max_confidence >= threshold
        n_predicted = mask.sum()
        n_abstained = (~mask).sum()

        if n_predicted > 0:
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test[mask], predictions[mask], average="macro")
        else:
            f1 = 0.0

        results.append({
            "threshold": threshold,
            "n_predicted": n_predicted,
            "n_abstained": n_abstained,
            "coverage_pct": round(n_predicted / len(y_test) * 100, 1),
            "f1_macro": round(f1, 4)
        })

    results_df = pd.DataFrame(results)
    logger.info(f"\nAbstain Threshold Analysis:\n{results_df.to_string()}")
    return results_df


def full_evaluation(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "production"
) -> dict:
    """Run the complete evaluation suite."""
    logger.info(f"\nRunning full evaluation for {model_name}...")

    # Basic predictions
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["Mild", "Moderate", "Severe"],
        output_dict=True
    )
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(y_test, y_pred, target_names=['Mild', 'Moderate', 'Severe'])}")

    # Plots
    plot_roc_curves(model, X_test, y_test, model_name)
    plot_calibration_curve(model, X_test, y_test, model_name)

    # Abstain analysis
    abstain_df = evaluate_abstain_thresholds(model, X_test, y_test)

    return {
        "classification_report": report,
        "abstain_analysis": abstain_df
    }


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))

    df = load_data()
    X, y, transformer = get_features_and_target(df, is_train=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model, feature_names, model_name = load_production_model()
    results = full_evaluation(model, X_test, y_test, model_name)
    print("Evaluation complete!")
