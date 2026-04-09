from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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
        max_iter=config.MAX_ITER,
        c=config.C,
        class_weight=config.CLASS_WEIGHT,
    )

    model_metrics = evaluate_model(model, X_test_processed, y_test)

    # Run cross-validation on the raw training split via a full pipeline to avoid leakage.
    cv_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessing_pipeline(
                    categorical_cols=config.CATEGORICAL_COLUMNS,
                    numerical_cols=config.NUMERICAL_COLUMNS,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    random_state=config.RANDOM_STATE,
                    max_iter=config.MAX_ITER,
                    C=config.C,
                    class_weight=config.CLASS_WEIGHT,
                ),
            ),
        ]
    )

    cv_auc = cross_val_score(cv_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    cv_f1 = cross_val_score(cv_pipeline, X_train, y_train, cv=5, scoring="f1")

    feature_names = preprocessing_pipeline.get_feature_names_out()
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)

    # --- Compare Results ---
    results = {
        "baseline": baseline_metrics,
        "model": model_metrics,
        "cross_validation": {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "cv_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_std": round(float(cv_f1.std()), 4),
        },
        "improvement": {
            "accuracy": round(model_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "balanced_accuracy": round(
                model_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                4,
            ),
            "precision": round(model_metrics["precision"] - baseline_metrics["precision"], 4),
            "recall": round(model_metrics["recall"] - baseline_metrics["recall"], 4),
            "f1": round(model_metrics["f1"] - baseline_metrics["f1"], 4),
            "roc_auc": round(model_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4),
        }
    }

    Path(config.METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(preprocessing_pipeline, config.PIPELINE_PATH)

    with open(config.METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    coef_df.to_csv(config.COEFFICIENTS_PATH, index=False)

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

    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
        baseline_val = results["baseline"][metric]
        model_val = results["model"][metric]
        improvement = results["improvement"][metric]

        print(f"{metric:<20} {baseline_val:<20.4f} {model_val:<20.4f} {improvement:+.4f}")

    print("\nCross-Validation (train split, 5-fold)")
    print("-" * 80)
    print(
        "ROC-AUC: "
        f"{results['cross_validation']['cv_roc_auc_mean']:.4f} "
        f"+/- {results['cross_validation']['cv_roc_auc_std']:.4f}"
    )
    print(
        "F1:      "
        f"{results['cross_validation']['cv_f1_mean']:.4f} "
        f"+/- {results['cross_validation']['cv_f1_std']:.4f}"
    )

    print("\nConfusion Matrix Counts (TN, FP, FN, TP)")
    print("-" * 80)
    print(
        "Baseline: "
        f"({int(results['baseline']['tn'])}, {int(results['baseline']['fp'])}, "
        f"{int(results['baseline']['fn'])}, {int(results['baseline']['tp'])})"
    )
    print(
        "Model:    "
        f"({int(results['model']['tn'])}, {int(results['model']['fp'])}, "
        f"{int(results['model']['fn'])}, {int(results['model']['tp'])})"
    )

    print("=" * 80 + "\n")

