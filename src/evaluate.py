from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model: Any, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a fitted classification model on test data.

    Parameters:
        model: Trained model that supports predict and optionally predict_proba.
        X_test: Test feature matrix.
        y_test: Ground-truth test labels.

    Returns:
        Dictionary containing accuracy, balanced_accuracy, precision, recall, f1,
        roc_auc, and confusion matrix counts.
    """
    predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics
