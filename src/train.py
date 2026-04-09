from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int,
    max_iter: int,
    c: float,
    class_weight: str | None,
) -> LogisticRegression:
    """Train and return a Logistic Regression classifier.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Seed for reproducible model behavior.
        max_iter: Maximum number of optimization iterations.
        c: Inverse of regularization strength.
        class_weight: Class weighting strategy for imbalanced data.

    Returns:
        A fitted LogisticRegression instance.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        C=c,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model
