from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
) -> RandomForestClassifier:
    """Train and return a RandomForest classifier.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Seed for reproducible model behavior.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth. Use None for unconstrained depth.

    Returns:
        A fitted RandomForestClassifier instance.
    """
    model = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return model
