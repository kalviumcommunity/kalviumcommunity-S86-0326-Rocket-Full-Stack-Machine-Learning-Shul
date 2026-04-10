from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Centralized project configuration for paths, schema, and model settings."""

    DATA_PATH: str = "data/raw/telco_churn.csv"
    PROCESSED_DATA_PATH: str = "data/processed/cleaned_telco_churn.csv"
    MODEL_PATH: str = "models/logistic_regression_model.pkl"
    PIPELINE_PATH: str = "models/preprocessing_pipeline.pkl"
    METRICS_PATH: str = "reports/metrics.json"
    PREDICTIONS_PATH: str = "reports/predictions.csv"
    COEFFICIENTS_PATH: str = "reports/logistic_coefficients.csv"
    PR_CURVE_PATH: str = "reports/precision_recall_curve.csv"
    CONFUSION_MATRIX_COUNTS_PATH: str = "reports/confusion_matrix_counts.csv"
    THRESHOLD_CONFUSION_MATRICES_PATH: str = "reports/threshold_confusion_matrices.csv"

    TARGET_COLUMN: str = "Churn"
    CATEGORICAL_COLUMNS: tuple[str, ...] = ("Contract", "PaymentMethod")
    NUMERICAL_COLUMNS: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")

    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    MAX_ITER: int = 1000
    C: float = 1.0
    CLASS_WEIGHT: str | None = None

    TARGET_RECALL: float = 0.8
    THRESHOLD_CANDIDATES: tuple[float, ...] = (0.3, 0.5, 0.7)
    F1_THRESHOLD_GRID: tuple[float, ...] = (
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
    )
