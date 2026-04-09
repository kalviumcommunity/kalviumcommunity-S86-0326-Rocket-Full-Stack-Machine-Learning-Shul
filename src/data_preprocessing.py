from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load tabular source data from CSV.

    Parameters:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing raw source rows.
    """
    return pd.read_csv(filepath)


def clean_data(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    """Clean missing values and normalize field formats for downstream ML steps.

    Parameters:
        df: Raw input DataFrame.
        numeric_columns: Columns expected to be numeric and median-imputed.
        categorical_columns: Columns expected to be categorical and mode-imputed.
        target_column: Name of the target column. Rows missing target are dropped.

    Returns:
        A cleaned DataFrame with consistent dtypes and missing values handled.
    """
    cleaned = df.copy()

    missing_columns = [
        col
        for col in [*numeric_columns, *categorical_columns, target_column]
        if col not in cleaned.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for col in numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()
        mode_value = cleaned[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        cleaned[col] = cleaned[col].fillna(fill_value)

    cleaned[target_column] = cleaned[target_column].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=[target_column])

    return cleaned


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cleaned data into training and test feature/target sets.

    Parameters:
        df: Cleaned DataFrame that includes feature and target columns.
        target_column: Name of the target column.
        test_size: Fraction reserved for test data.
        random_state: Seed used for deterministic splitting.
        stratify: Whether to stratify split by target labels.

    Returns:
        Tuple ordered as (X_train, X_test, y_train, y_test).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column].map({"Yes": 1, "No": 0})
    if y.isna().any():
        invalid_labels = sorted(df.loc[y.isna(), target_column].astype(str).unique().tolist())
        raise ValueError(f"Unsupported target labels found: {invalid_labels}. Expected only 'Yes'/'No'.")
    y = y.astype(int)

    stratify_labels = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return X_train, X_test, y_train, y_test
