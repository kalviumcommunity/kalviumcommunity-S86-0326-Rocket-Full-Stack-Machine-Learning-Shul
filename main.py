from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import build_preprocessing_pipeline
from src.predict import predict_new_data
from src.train import train_model


def run_training_and_prediction(config: Config) -> dict[str, float]:
    """Run the end-to-end ML workflow from ingestion to sample predictions.

    Parameters:
        config: Centralized configuration object containing paths and model settings.

    Returns:
        Dictionary of evaluation metrics computed on the test split.
    """
    raw_df = load_data(config.DATA_PATH)
    cleaned_df = clean_data(
        raw_df,
        numeric_columns=config.NUMERICAL_COLUMNS,
        categorical_columns=config.CATEGORICAL_COLUMNS,
        target_column=config.TARGET_COLUMN,
    )

    Path(config.PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(config.PROCESSED_DATA_PATH, index=False)

    X_train, X_test, y_train, y_test = split_data(
        cleaned_df,
        target_column=config.TARGET_COLUMN,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=True,
    )

    preprocessing_pipeline = build_preprocessing_pipeline(
        categorical_cols=config.CATEGORICAL_COLUMNS,
        numerical_cols=config.NUMERICAL_COLUMNS,
    )
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    model = train_model(
        X_train=X_train_processed,
        y_train=y_train,
        random_state=config.RANDOM_STATE,
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
    )

    metrics = evaluate_model(model, X_test_processed, y_test)

    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(preprocessing_pipeline, config.PIPELINE_PATH)

    Path(config.METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    sample_predictions = predict_new_data(X_test.head(5), model=model, pipeline=preprocessing_pipeline)
    sample_predictions.to_csv(config.PREDICTIONS_PATH, index=False)

    return metrics


if __name__ == "__main__":
    metrics_result = run_training_and_prediction(Config())
    print("Training complete. Metrics:", metrics_result)
