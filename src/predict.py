from __future__ import annotations

from typing import Any

import joblib
import pandas as pd


def load_artifacts(model_path: str, pipeline_path: str) -> tuple[Any, Any]:
    """Load serialized model and preprocessing pipeline artifacts.

    Parameters:
        model_path: Path to the saved model artifact.
        pipeline_path: Path to the saved preprocessing pipeline artifact.

    Returns:
        Tuple of (model, pipeline) loaded from disk.
    """
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


def predict_new_data(new_data: pd.DataFrame, model: Any, pipeline: Any) -> pd.DataFrame:
    """Generate predictions for new observations using fitted artifacts.

    Parameters:
        new_data: New input rows for inference.
        model: Fitted model artifact.
        pipeline: Fitted preprocessing pipeline artifact.

    Returns:
        DataFrame containing original columns and prediction outputs.
    """
    transformed = pipeline.transform(new_data)
    predicted_class = model.predict(transformed)

    result = new_data.copy()
    result["prediction"] = predicted_class

    if hasattr(model, "predict_proba"):
        result["prediction_probability"] = model.predict_proba(transformed)[:, 1]

    return result
