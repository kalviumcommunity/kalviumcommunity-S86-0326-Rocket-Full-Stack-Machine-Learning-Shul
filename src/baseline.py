from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


def create_baseline(strategy: str = "most_frequent") -> DummyClassifier:
    """Create a baseline classifier using a simple heuristic strategy.

    Parameters:
        strategy: Baseline strategy. Options:
            - "most_frequent": Always predict the most common class (default)
            - "stratified": Predict randomly respecting training class distribution
            - "uniform": Predict all classes with equal probability

    Returns:
        An unfitted DummyClassifier instance ready for fit_transform.
    """
    return DummyClassifier(strategy=strategy, random_state=42)


def evaluate_baseline(
    baseline: DummyClassifier,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Fit and evaluate a baseline classifier on test data.

    Parameters:
        baseline: Unfitted DummyClassifier instance.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target labels.
        y_test: Ground-truth test labels.

    Returns:
        Dictionary of baseline metrics (accuracy, precision, recall, f1, roc_auc).
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    # Fit baseline on training data only (no leakage)
    baseline.fit(X_train, y_train)

    # Predict on test data
    predictions = baseline.predict(X_test)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    # Most baseline strategies expose class probabilities, enabling ROC-AUC comparison.
    if hasattr(baseline, "predict_proba"):
        probabilities = baseline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics
