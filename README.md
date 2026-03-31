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

### 1. Separation of Concerns
Each module handles one stage of the ML workflow:
- **data_preprocessing.py** — How do we load and prepare raw data?
- **feature_engineering.py** — How do we transform cleaned data into model-ready features?
- **train.py** — How do we fit a model on prepared features?
- **evaluate.py** — How do we measure model performance?
- **predict.py** — How do we generate predictions using saved artifacts?

This separation prevents code duplication, makes testing easier, and ensures that changes to one stage don't accidentally break another.

### 2. Single Responsibility Principle
Each function does exactly one thing:
- `load_data()` → loads CSV
- `clean_data()` → handles missing values
- `split_data()` → splits into train/test
- `build_preprocessing_pipeline()` → creates transformers
- `train_model()` → trains and returns fitted model
- `evaluate_model()` → computes metrics, returns dict (never prints)
- `predict_new_data()` → generates predictions only

No function loads AND cleans AND trains. This isolation makes functions independently testable and reusable.

### 3. Centralized Configuration
All file paths, random seeds, hyperparameters, and column names live in `src/config.py`:
- Change the random seed in one place, it updates everywhere
- Change a file path once, not across multiple modules
- Makes reproducibility explicit and verifiable
- Eliminates magic numbers scattered through code

### 4. Clear Input/Output Contracts
Every function has:
- **Type hints** on parameters and return values (e.g., `X_train: pd.DataFrame → RandomForestClassifier`)
- **Docstrings** explaining what it does, what it expects, what it returns
- **Explicit parameters** (no reliance on global variables)

This makes functions impossible to misuse and easy for others to understand.

### 5. Training/Inference Separation (Prevents Data Leakage)
```python
# Training fits on training data
X_train_processed = pipeline.fit_transform(X_train)

# Prediction only transforms, never refits
X_new_processed = pipeline.transform(new_data)  # NOT fit_transform
```
Training and prediction are completely separate. Prediction loads already-fitted artifacts and uses `transform()`, never `fit()`. This architectural boundary prevents accidental data leakage.

### 6. Reproducibility via Explicit Randomness
Every operation that involves randomness exposes `random_state`:
```python
X_train, X_test, y_train, y_test = split_data(df, ..., random_state=42)
model = train_model(..., random_state=42)
```
Same seed → identical results every time. No surprises, no debugging "why was it better yesterday?"

### 7. Functions Return Data, Not Print
Core logic functions return values. The orchestration layer (main.py) decides what to do:
```python
metrics = evaluate_model(model, X_test, y_test)  # Returns dict
# Orchestration decides: save it, print it, log it, email it
```
No printing inside core functions. This makes functions composable and testable.

### 8. Explicit Imports (No Wildcards)
```python
# Good - clear where each function comes from
from src.data_preprocessing import load_data, clean_data, split_data
from src.train import train_model

# Avoid - obscures dependencies
from src.data_preprocessing import *
```

### Why This Structure Matters
Without structure:
- ❌ Preprocessing logic copy-pasted between training and prediction → diverges over time
- ❌ Randomness not seeded → results change every run
- ❌ Training/prediction mixed → data leakage possible
- ❌ All logic in one file → can't test components independently
- ❌ Hardcoded paths → doesn't work on other machines

With structure:
- ✅ One source of truth for preprocessing (called by both train and predict)
- ✅ Deterministic results (explicit random seed everywhere)
- ✅ Impossible to leak data (prediction module can't call fit())
- ✅ Each function independently testable
- ✅ Paths centralized in config.py

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

- **PR Description (copy to GitHub):**

```markdown
## What Changed
Refactored ML workflow into modular, maintainable Python functions following production-grade engineering practices.

### Files and Changes
- `src/config.py` — Centralized all configuration (file paths, random seeds, hyperparameters, column names)
- `src/data_preprocessing.py` — Data loading, cleaning, and train/test splitting
- `src/feature_engineering.py` — Preprocessing pipeline (OneHotEncoder + StandardScaler)
- `src/train.py` — Model training function with explicit parameters and artifact return
- `src/evaluate.py` — Evaluation function that returns metrics dict (never prints)
- `src/predict.py` — Prediction function that loads artifacts and transforms new data (never refits)
- `main.py` — Orchestration script that sequences all steps
- `requirements.txt` — Pinned dependencies for reproducibility
- `README.md` — Updated with design principles and architectural rationale

## Why This Matters

### Problem Solved
In unstructured projects:
- Preprocessing code lives in notebook cells and training scripts (hard to reuse)
- Prediction script accidentally copy-pastes preprocessing (diverges from training)
- Random operations aren't seeded (results change every run)
- Training and prediction mixed together (data leakage possible)
- No way to test individual components independently

### Solution
This restructuring creates:
- **One source of truth** for each operation (load data once, use everywhere)
- **Deterministic results** (explicit random_state in all stochastic operations)
- **Safe predictions** (prediction module never refits transformations)
- **Testable components** (each function independently verifiable)
- **Clear ownership** (clear which module is responsible for what)

## Design Principles Demonstrated

✅ **Separation of Concerns** — Each module answers one question
- data_preprocessing.py: "How do we load and prepare raw data?"
- feature_engineering.py: "How do we transform data into features?"
- train.py: "How do we fit a model?"
- evaluate.py: "How do we measure performance?"
- predict.py: "How do we make predictions on new data?"

✅ **Single Responsibility** — Each function does exactly one thing
- load_data() only loads, doesn't clean
- clean_data() only cleans, doesn't split
- train_model() only trains, doesn't evaluate
- evaluate_model() only evaluates, doesn't save
- Can change any function without affecting others

✅ **Clear Contracts** — Type hints + docstrings on every function
```python
def train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
) -> RandomForestClassifier:
```
No ambiguity about what goes in and what comes out.

✅ **No Hidden State** — All config passed explicitly
```python
from src.config import Config
config = Config()
model = train_model(..., random_state=config.RANDOM_STATE)
```
Dependencies are visible. Functions can be called with different configs for experimentation.

✅ **Training/Inference Separation** — Prevents data leakage
- Training: `pipeline.fit_transform(X_train)` on training data only
- Prediction: `pipeline.transform(new_data)` using already-fitted pipeline, never fits
- Prediction module doesn't import training logic; boundaries are enforced

✅ **Reproducibility** — Explicit random seeds everywhere
```python
split_data(..., random_state=42)  # Same seed → same split every time
train_model(..., random_state=42)  # Same seed → same model every time
```

✅ **Pure Functions** — Return data, never print
```python
metrics = evaluate_model(model, X_test, y_test)  # Returns dict
# Orchestration layer decides whether to print, save, or log
```
Core functions are testable and composable.

✅ **Explicit Imports** — Clear dependencies
```python
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
```
No wildcard imports. Instantly clear what's being used.

## How to Verify

```bash
# Install dependencies
pip install -r requirements.txt

# Run end-to-end pipeline
python main.py

# Expected output
Training complete. Metrics: {'accuracy': 0.5, 'precision': 0.5, 'recall': 1.0, 'f1': 0.67, 'roc_auc': 0.5}
```

Artifacts created:
- `data/processed/cleaned_telco_churn.csv` — Clean training data
- `models/random_forest_model.pkl` — Fitted model
- `models/preprocessing_pipeline.pkl` — Fitted transformers
- `reports/metrics.json` — Evaluation metrics
- `reports/predictions.csv` — Sample predictions

## Real-World Impact

This structure allows you to:
- **Train once, deploy anywhere** — Save fitted model and pipeline, load in any environment
- **Reuse preprocessing safely** — Same clean_data() and transformations used in training and prediction
- **Experiment confidently** — Change one component without breaking others
- **Debug efficiently** — When something breaks, know exactly which module is responsible
- **Collaborate effectively** — Teammate can understand module responsibilities without asking
- **Hand off the project** — Another engineer can read the structure and maintain it

## Next: How This Generalizes

This structure works for:
- Different datasets (swap data path in config)
- Different models (swap RandomForest for XGBoost in train.py)
- Different preprocessing (update feature_engineering.py)
- Different evaluation metrics (update evaluate.py)
- Production deployment (load artifacts, call predict.py)

All without rewriting the entire project. This is what separates experiments from systems.
```