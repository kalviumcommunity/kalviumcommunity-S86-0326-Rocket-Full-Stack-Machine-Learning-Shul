from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
) -> ColumnTransformer:
    """Build a preprocessing pipeline for categorical and numerical features.

    Parameters:
        categorical_cols: Column names to one-hot encode.
        numerical_cols: Column names to standardize.

    Returns:
        A ColumnTransformer ready for fit_transform and transform calls.
    """
    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    numerical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(categorical_cols)),
            ("numerical", numerical_pipeline, list(numerical_cols)),
        ],
        remainder="drop",
    )
    return preprocessing_pipeline
