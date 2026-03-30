# [Refactor] Modular ML Pipeline Into Reusable Functions

This repository demonstrates a clean machine learning workflow built with modular Python functions and explicit imports.

## Project Structure

```text
project_root/
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## Module Responsibilities

- `src/config.py`: Centralized configuration (paths, schema, randomness, model hyperparameters).
- `src/data_preprocessing.py`: Data loading, cleaning, and train/test splitting utilities.
- `src/feature_engineering.py`: Encodes categorical features and scales numerical features.
- `src/train.py`: Trains and returns a fitted model.
- `src/evaluate.py`: Computes and returns evaluation metrics as a dictionary.
- `src/predict.py`: Loads saved artifacts and generates predictions for new data.
- `main.py`: Orchestrates the full train/evaluate/save/predict sequence.

## Design Principles Applied

- Single-responsibility functions.
- Explicit input/output contracts with type hints.
- Clear docstrings for purpose, parameters, and return values.
- No hidden global state inside core functions.
- Reproducibility via explicit `random_state` usage.
- Imports are explicit and modular (no wildcard imports).

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run End-to-End Workflow

```bash
python main.py
```

Outputs created after run:

- `data/processed/cleaned_telco_churn.csv`
- `models/random_forest_model.pkl`
- `models/preprocessing_pipeline.pkl`
- `reports/metrics.json`
- `reports/predictions.csv`

## Pull Request Details (Assignment Ready)

- **PR Title:** `[Refactor] Modularize ML pipeline into reusable functions`
- **Suggested PR Description:**

```text
### What changed
- Refactored ML workflow into modular functions under `src/`
- Added centralized configuration via `Config`
- Separated data preprocessing, feature engineering, training, evaluation, and prediction concerns
- Added orchestration script (`main.py`) to run full pipeline in sequence
- Added typed function signatures and docstrings for all core functions
- Added sample dataset and requirements for reproducibility

### Why this change
- Improves maintainability, testability, and reuse
- Prevents notebook execution-order dependency issues
- Makes training/inference boundaries explicit and safer
```