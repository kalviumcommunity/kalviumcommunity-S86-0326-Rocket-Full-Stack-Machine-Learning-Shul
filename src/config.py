from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Centralized project configuration for paths, schema, and model settings."""

    DATA_PATH: str = "data/raw/telco_churn.csv"
    PROCESSED_DATA_PATH: str = "data/processed/cleaned_telco_churn.csv"
    MODEL_PATH: str = "models/random_forest_model.pkl"
    PIPELINE_PATH: str = "models/preprocessing_pipeline.pkl"
    METRICS_PATH: str = "reports/metrics.json"
    PREDICTIONS_PATH: str = "reports/predictions.csv"

    TARGET_COLUMN: str = "Churn"
    CATEGORICAL_COLUMNS: tuple[str, ...] = ("Contract", "PaymentMethod")
    NUMERICAL_COLUMNS: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    N_ESTIMATORS: int = 300
    MAX_DEPTH: int | None = 8
