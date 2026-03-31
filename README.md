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

## How This Code Implements ML Engineering Principles

### 1. **Single Responsibility**
Each module has one clear purpose:
- `data_preprocessing.py` handles only data loading, cleaning, and splitting
- `feature_engineering.py` builds and applies transformations
- `train.py` only trains and returns the model
- `evaluate.py` only computes metrics (returns dict, does not print)
- `predict.py` only generates predictions from saved artifacts

### 2. **Clear Input/Output Contracts**
All functions use type hints and explicit signatures:
```python
def train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
) -> RandomForestClassifier:
```
Every function clearly documents what it expects and what it returns.

### 3. **No Hidden Global State**
Configuration is passed explicitly through the `Config` object:
```python
from src.config import Config

config = Config()  # All settings are visible
model = train_model(..., random_state=config.RANDOM_STATE)
```
Functions never depend on globals; configuration flows downward through parameters.

### 4. **Reproducibility**
Every stochastic operation exposes `random_state`:
```python
X_train, X_test, y_train, y_test = split_data(
    df_clean,
    target_column=config.TARGET_COLUMN,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,  # Explicit seed
    stratify=True,
)
```
Run the pipeline again and get identical results.

### 5. **Training/Inference Separation**
Training logic is completely isolated from prediction:
- `train_model()` takes raw training data and returns a fitted artifact
- `predict_new_data()` takes only a fitted model and pipeline (no training code)
- The prediction module never calls fit or fit_transform, preventing data leakage

### 6. **Functions Return Data, Not Print**
Core logic functions return values for composition and logging:
```python
metrics = evaluate_model(model, X_test_processed, y_test)
# Calling code decides whether to print, save, or compose into larger pipelines
```

### 7. **Orchestration Layer**
`main.py` keeps the workflow readable and sequences all steps:
```python
raw_df = load_data(config.DATA_PATH)
cleaned_df = clean_data(raw_df, ...)
X_train, X_test, y_train, y_test = split_data(cleaned_df, ...)
# ... continues in clear sequence
```

## Example: How Modularity Prevents Common Failures

**Without modularity:** A single notebook cell contains loading, cleaning, encoding, scaling, training, and evaluation. To predict on new data, you copy the cell and change variable names. The new copy has a subtle bug (missing a preprocessing step), but the original still works. Two versions diverge. Impossible to debug.

**With modularity:** 
- New data calls `clean_data()` using the same function as training
- New data passes through the same `preprocessing_pipeline` (already fitted)
- No copy-paste. One source of truth. Bug fix in any function fixes both training and prediction.

## Outputs After Running

```bash
python main.py
```

Creates:
- `data/processed/cleaned_telco_churn.csv` — cleaned training data
- `models/random_forest_model.pkl` — fitted model artifact
- `models/preprocessing_pipeline.pkl` — fitted transformer pipeline
- `reports/metrics.json` — test set evaluation metrics
- `reports/predictions.csv` — sample predictions on test data

## Pull Request Details (Assignment Ready)

- **PR Title:** `[Refactor] Modularize ML pipeline into reusable functions`
- **Suggested PR Description:**

```text
### What changed
- Refactored ML workflow into modular functions across `src/` modules
- Added centralized `Config` class for paths, schema, and hyperparameters
- Separated concerns: data loading → cleaning → splitting → encoding/scaling → training → evaluation → prediction
- All functions use type hints, docstrings, and explicit parameter passing
- Added orchestration script (`main.py`) that sequences the full pipeline
- Enforced training/inference separation (prediction never refits transformations)

### Why this change
**Maintainability:** One source of truth for each operation. Bug fixes in preprocessing apply to both training and prediction.

**Reproducibility:** Explicit random_state parameters ensure identical results across runs.

**Reusability:** Functions can be called independently. Preprocessing generalizes to any new dataset.

**Safety:** Training and inference are hardened boundaries. No data leakage possible.

**Collaboration:** Clear module responsibilities mean teammate can understand and modify code without breaking others' work.

### Principles Demonstrated
- ✅ Single Responsibility: Each function does one thing
- ✅ Clear Contracts: Type hints + docstrings for all functions
- ✅ No Hidden State: Config passed explicitly, no globals
- ✅ Determinism: All randomness is seeded
- ✅ Separation of Concerns: Training logic never mixed with inference
- ✅ Return Data: Functions return values, orchestration layer handles printing/saving
```