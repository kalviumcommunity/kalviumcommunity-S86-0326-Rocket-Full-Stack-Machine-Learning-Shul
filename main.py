from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
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

    X_train_full, X_test, y_train_full, y_test = split_data(
        cleaned_df,
        target_column=config.TARGET_COLUMN,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_train_full,
    )

    # --- Establish Baseline ---
    baseline = create_baseline(strategy="most_frequent")
    baseline_metrics = evaluate_baseline(baseline, X_train_full, X_test, y_train_full, y_test)

    # --- Train Real Model ---
    preprocessing_pipeline = build_preprocessing_pipeline(
        categorical_cols=config.CATEGORICAL_COLUMNS,
        numerical_cols=config.NUMERICAL_COLUMNS,
    )
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    model = train_model(
        X_train=X_train_processed,
        y_train=y_train,
        random_state=config.RANDOM_STATE,
        max_iter=config.MAX_ITER,
        c=config.C,
        class_weight=config.CLASS_WEIGHT,
    )

    model_metrics_default = evaluate_model(model, X_test_processed, y_test)

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

    cv_auc = cross_val_score(cv_pipeline, X_train_full, y_train_full, cv=5, scoring="roc_auc")
    cv_precision = cross_val_score(cv_pipeline, X_train_full, y_train_full, cv=5, scoring="precision")
    cv_recall = cross_val_score(cv_pipeline, X_train_full, y_train_full, cv=5, scoring="recall")
    cv_f1 = cross_val_score(cv_pipeline, X_train_full, y_train_full, cv=5, scoring="f1")

    val_prob = model.predict_proba(X_val_processed)[:, 1]
    threshold_f1: dict[float, float] = {}
    for threshold in config.F1_THRESHOLD_GRID:
        val_pred = (val_prob >= threshold).astype(int)
        threshold_f1[threshold] = float(f1_score(y_val, val_pred, zero_division=0))

    best_f1_threshold = max(threshold_f1, key=threshold_f1.get)
    best_val_f1 = threshold_f1[best_f1_threshold]

    test_prob = model.predict_proba(X_test_processed)[:, 1]
    test_pred_default = (test_prob >= 0.5).astype(int)
    test_pred_optimized = (test_prob >= best_f1_threshold).astype(int)

    model_metrics_optimized = _compute_binary_metrics(y_test, test_pred_optimized)
    model_metrics_optimized["roc_auc"] = float(roc_auc_score(y_test, test_prob))

    threshold_metrics: dict[str, dict[str, float]] = {}
    threshold_confusion_rows: list[dict[str, float | str]] = []
    for threshold in config.THRESHOLD_CANDIDATES:
        y_custom = (test_prob >= threshold).astype(int)
        threshold_confusion = _compute_confusion_summary(y_test, y_custom)
        threshold_metrics[f"{threshold:.2f}"] = {
            "precision": round(float(precision_score(y_test, y_custom, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_custom, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_custom, zero_division=0)), 4),
        }
        threshold_confusion_rows.append(
            {
                "threshold": f"{threshold:.2f}",
                "tn": int(threshold_confusion["tn"]),
                "fp": int(threshold_confusion["fp"]),
                "fn": int(threshold_confusion["fn"]),
                "tp": int(threshold_confusion["tp"]),
                "tn_rate": round(float(threshold_confusion["tn_rate"]), 4),
                "fp_rate": round(float(threshold_confusion["fp_rate"]), 4),
                "fn_rate": round(float(threshold_confusion["fn_rate"]), 4),
                "tp_rate": round(float(threshold_confusion["tp_rate"]), 4),
            }
        )

    precisions, recalls, thresholds = precision_recall_curve(y_test, test_prob)
    # precision_recall_curve returns one more precision/recall point than thresholds.
    pr_curve_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precisions[:-1],
            "recall": recalls[:-1],
        }
    )

    valid_idx = np.where(recalls[:-1] >= config.TARGET_RECALL)[0]
    best_threshold_summary: dict[str, float | None]
    if len(valid_idx) > 0:
        best_idx = int(valid_idx[np.argmax(precisions[:-1][valid_idx])])
        best_threshold_summary = {
            "target_recall": round(config.TARGET_RECALL, 4),
            "threshold": round(float(thresholds[best_idx]), 4),
            "precision": round(float(precisions[best_idx]), 4),
            "recall": round(float(recalls[best_idx]), 4),
        }
    else:
        best_threshold_summary = {
            "target_recall": round(config.TARGET_RECALL, 4),
            "threshold": None,
            "precision": None,
            "recall": None,
        }

    threshold_tuning = {
        "optimized_for": "f1",
        "optimized_on": "validation",
        "evaluation_on": "test",
        "default_threshold": 0.5,
        "best_threshold": round(float(best_f1_threshold), 4),
        "best_validation_f1": round(float(best_val_f1), 4),
        "test_f1_default": round(float(f1_score(y_test, test_pred_default, zero_division=0)), 4),
        "test_f1_optimized": round(float(model_metrics_optimized["f1"]), 4),
    }

    baseline_confusion = _compute_confusion_summary(y_test, baseline.predict(X_test))
    default_confusion = _compute_confusion_summary(y_test, test_pred_default)
    optimized_confusion = _compute_confusion_summary(y_test, test_pred_optimized)

    confusion_matrix_summary = {
        "baseline": {
            "layout": [["tn", "fp"], ["fn", "tp"]],
            "counts": {
                "tn": int(baseline_confusion["tn"]),
                "fp": int(baseline_confusion["fp"]),
                "fn": int(baseline_confusion["fn"]),
                "tp": int(baseline_confusion["tp"]),
            },
            "normalized_true": {
                "tn_rate": round(float(baseline_confusion["tn_rate"]), 4),
                "fp_rate": round(float(baseline_confusion["fp_rate"]), 4),
                "fn_rate": round(float(baseline_confusion["fn_rate"]), 4),
                "tp_rate": round(float(baseline_confusion["tp_rate"]), 4),
            },
        },
        "model_default_threshold": {
            "threshold": 0.5,
            "layout": [["tn", "fp"], ["fn", "tp"]],
            "counts": {
                "tn": int(default_confusion["tn"]),
                "fp": int(default_confusion["fp"]),
                "fn": int(default_confusion["fn"]),
                "tp": int(default_confusion["tp"]),
            },
            "normalized_true": {
                "tn_rate": round(float(default_confusion["tn_rate"]), 4),
                "fp_rate": round(float(default_confusion["fp_rate"]), 4),
                "fn_rate": round(float(default_confusion["fn_rate"]), 4),
                "tp_rate": round(float(default_confusion["tp_rate"]), 4),
            },
        },
        "model_optimized_threshold": {
            "threshold": round(float(best_f1_threshold), 4),
            "layout": [["tn", "fp"], ["fn", "tp"]],
            "counts": {
                "tn": int(optimized_confusion["tn"]),
                "fp": int(optimized_confusion["fp"]),
                "fn": int(optimized_confusion["fn"]),
                "tp": int(optimized_confusion["tp"]),
            },
            "normalized_true": {
                "tn_rate": round(float(optimized_confusion["tn_rate"]), 4),
                "fp_rate": round(float(optimized_confusion["fp_rate"]), 4),
                "fn_rate": round(float(optimized_confusion["fn_rate"]), 4),
                "tp_rate": round(float(optimized_confusion["tp_rate"]), 4),
            },
        },
    }

    final_pipeline = build_preprocessing_pipeline(
        categorical_cols=config.CATEGORICAL_COLUMNS,
        numerical_cols=config.NUMERICAL_COLUMNS,
    )
    X_train_full_processed = final_pipeline.fit_transform(X_train_full)
    final_model = train_model(
        X_train=X_train_full_processed,
        y_train=y_train_full,
        random_state=config.RANDOM_STATE,
        max_iter=config.MAX_ITER,
        c=config.C,
        class_weight=config.CLASS_WEIGHT,
    )

    feature_names = final_pipeline.get_feature_names_out()
    coefficients = final_model.coef_[0]
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
        "model": {
            "default_threshold": model_metrics_default,
            "optimized_threshold": model_metrics_optimized,
        },
        "cross_validation": {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "cv_precision_mean": round(float(cv_precision.mean()), 4),
            "cv_precision_std": round(float(cv_precision.std()), 4),
            "cv_recall_mean": round(float(cv_recall.mean()), 4),
            "cv_recall_std": round(float(cv_recall.std()), 4),
            "cv_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_std": round(float(cv_f1.std()), 4),
        },
        "threshold_analysis": {
            "candidate_thresholds": threshold_metrics,
            "best_for_target_recall": best_threshold_summary,
            "f1_threshold_tuning": threshold_tuning,
        },
        "confusion_matrix": confusion_matrix_summary,
        "improvement": {
            "accuracy": round(model_metrics_optimized["accuracy"] - baseline_metrics["accuracy"], 4),
            "balanced_accuracy": round(
                model_metrics_optimized["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                4,
            ),
            "precision": round(model_metrics_optimized["precision"] - baseline_metrics["precision"], 4),
            "recall": round(model_metrics_optimized["recall"] - baseline_metrics["recall"], 4),
            "f1": round(model_metrics_optimized["f1"] - baseline_metrics["f1"], 4),
            "roc_auc": round(model_metrics_optimized["roc_auc"] - baseline_metrics["roc_auc"], 4),
        }
    }

    Path(config.METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.MODEL_PATH)
    joblib.dump(final_pipeline, config.PIPELINE_PATH)

    with open(config.METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    coef_df.to_csv(config.COEFFICIENTS_PATH, index=False)
    pr_curve_df.to_csv(config.PR_CURVE_PATH, index=False)
    pd.DataFrame(
        [
            {
                "model_variant": "baseline",
                "threshold": "n/a",
                "tn": int(baseline_confusion["tn"]),
                "fp": int(baseline_confusion["fp"]),
                "fn": int(baseline_confusion["fn"]),
                "tp": int(baseline_confusion["tp"]),
                "tn_rate": round(float(baseline_confusion["tn_rate"]), 4),
                "fp_rate": round(float(baseline_confusion["fp_rate"]), 4),
                "fn_rate": round(float(baseline_confusion["fn_rate"]), 4),
                "tp_rate": round(float(baseline_confusion["tp_rate"]), 4),
            },
            {
                "model_variant": "model_default_threshold",
                "threshold": "0.50",
                "tn": int(default_confusion["tn"]),
                "fp": int(default_confusion["fp"]),
                "fn": int(default_confusion["fn"]),
                "tp": int(default_confusion["tp"]),
                "tn_rate": round(float(default_confusion["tn_rate"]), 4),
                "fp_rate": round(float(default_confusion["fp_rate"]), 4),
                "fn_rate": round(float(default_confusion["fn_rate"]), 4),
                "tp_rate": round(float(default_confusion["tp_rate"]), 4),
            },
            {
                "model_variant": "model_optimized_threshold",
                "threshold": f"{best_f1_threshold:.2f}",
                "tn": int(optimized_confusion["tn"]),
                "fp": int(optimized_confusion["fp"]),
                "fn": int(optimized_confusion["fn"]),
                "tp": int(optimized_confusion["tp"]),
                "tn_rate": round(float(optimized_confusion["tn_rate"]), 4),
                "fp_rate": round(float(optimized_confusion["fp_rate"]), 4),
                "fn_rate": round(float(optimized_confusion["fn_rate"]), 4),
                "tp_rate": round(float(optimized_confusion["tp_rate"]), 4),
            },
        ]
    ).to_csv(config.CONFUSION_MATRIX_COUNTS_PATH, index=False)
    pd.DataFrame(threshold_confusion_rows).to_csv(config.THRESHOLD_CONFUSION_MATRICES_PATH, index=False)

    sample_predictions = predict_new_data(X_test.head(5), model=final_model, pipeline=final_pipeline)
    sample_predictions.to_csv(config.PREDICTIONS_PATH, index=False)

    return results


def _compute_binary_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics from labels only."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _compute_confusion_summary(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return confusion matrix counts and normalized row-wise rates."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    actual_negative = tn + fp
    actual_positive = fn + tp
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "tn_rate": float(tn / actual_negative) if actual_negative > 0 else 0.0,
        "fp_rate": float(fp / actual_negative) if actual_negative > 0 else 0.0,
        "fn_rate": float(fn / actual_positive) if actual_positive > 0 else 0.0,
        "tp_rate": float(tp / actual_positive) if actual_positive > 0 else 0.0,
    }


if __name__ == "__main__":
    results = run_training_and_prediction(Config())

    # Pretty-print results with baseline comparison
    print("\n" + "=" * 80)
    print("BASELINE vs MODEL COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Baseline':<20} {'Model (opt.)':<20} {'Improvement':<15}")
    print("-" * 80)

    model_optimized = results["model"]["optimized_threshold"]
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
        baseline_val = results["baseline"][metric]
        model_val = model_optimized[metric]
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
    print(
        "Precision: "
        f"{results['cross_validation']['cv_precision_mean']:.4f} "
        f"+/- {results['cross_validation']['cv_precision_std']:.4f}"
    )
    print(
        "Recall:    "
        f"{results['cross_validation']['cv_recall_mean']:.4f} "
        f"+/- {results['cross_validation']['cv_recall_std']:.4f}"
    )

    print("\nThreshold Analysis (Precision/Recall/F1)")
    print("-" * 80)
    for threshold, metrics in results["threshold_analysis"]["candidate_thresholds"].items():
        print(
            f"threshold={threshold} | precision={metrics['precision']:.4f} "
            f"| recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f}"
        )

    tuning = results["threshold_analysis"]["f1_threshold_tuning"]
    print(
        "best threshold for F1 on validation: "
        f"{tuning['best_threshold']:.4f} "
        f"(val_f1={tuning['best_validation_f1']:.4f})"
    )
    print(
        "test F1 default vs optimized: "
        f"{tuning['test_f1_default']:.4f} -> {tuning['test_f1_optimized']:.4f}"
    )

    best = results["threshold_analysis"]["best_for_target_recall"]
    if best["threshold"] is not None:
        print(
            "best threshold for target recall "
            f"{best['target_recall']:.2f}: {best['threshold']:.4f} "
            f"(precision={best['precision']:.4f}, recall={best['recall']:.4f})"
        )
    else:
        print(
            "No threshold met target recall "
            f"{best['target_recall']:.2f}; consider retraining or feature improvements."
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
        f"({int(results['model']['optimized_threshold']['tn'])}, "
        f"{int(results['model']['optimized_threshold']['fp'])}, "
        f"{int(results['model']['optimized_threshold']['fn'])}, "
        f"{int(results['model']['optimized_threshold']['tp'])})"
    )

    print("=" * 80 + "\n")

