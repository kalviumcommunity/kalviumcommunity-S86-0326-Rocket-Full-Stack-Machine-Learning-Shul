from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.baseline import create_baseline, evaluate_baseline
from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import build_preprocessing_pipeline
from src.predict import predict_new_data
from src.train import train_model


def run_training_and_prediction(config: Config) -> dict[str, dict[str, float]]:
    """Run the end-to-end ML workflow from ingestion to sample predictions.

    Includes baseline comparison to validate that the model adds real value over naive strategies.

    Parameters:
        config: Centralized configuration object containing paths and model settings.

    Returns:
        Dictionary containing both baseline and model metrics computed on the test split.
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

    # --- Establish Baseline ---
    baseline = create_baseline(strategy="most_frequent")
    baseline_metrics = evaluate_baseline(baseline, X_train, X_test, y_train, y_test)

    # --- Train Real Model ---
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

    model_metrics = evaluate_model(model, X_test_processed, y_test)

    # --- Compare Results ---
    results = {
        "baseline": baseline_metrics,
        "model": model_metrics,
        "improvement": {
            "accuracy": round(model_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "precision": round(model_metrics["precision"] - baseline_metrics["precision"], 4),
            "recall": round(model_metrics["recall"] - baseline_metrics["recall"], 4),
            "f1": round(model_metrics["f1"] - baseline_metrics["f1"], 4),
        }
    }

    Path(config.METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(preprocessing_pipeline, config.PIPELINE_PATH)

    with open(config.METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    sample_predictions = predict_new_data(X_test.head(5), model=model, pipeline=preprocessing_pipeline)
    sample_predictions.to_csv(config.PREDICTIONS_PATH, index=False)

    return results


if __name__ == "__main__":
    results = run_training_and_prediction(Config())

    # Pretty-print results with baseline comparison
    print("\n" + "=" * 80)
    print("BASELINE vs MODEL COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Baseline':<20} {'Model':<20} {'Improvement':<15}")
    print("-" * 80)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        baseline_val = results["baseline"][metric]
        model_val = results["model"][metric]
        improvement = results["improvement"][metric]

        print(f"{metric:<20} {baseline_val:<20.4f} {model_val:<20.4f} {improvement:+.4f}")

    print("=" * 80 + "\n")

