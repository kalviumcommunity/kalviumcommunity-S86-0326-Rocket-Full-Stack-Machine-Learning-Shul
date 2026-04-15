# Lesson 18: Scikit-Learn Pipelines - Building Reproducible, Production-Ready ML Workflows

## Overview

As ML workflows grow more complex, the risk of silent failures increases. **Scikit-Learn Pipelines** transform fragile scripts into production-ready structure that guarantees correctness at every step.

### The Core Problem

Manual preprocessing introduces multiple failure modes:

1. **Data Leakage** — Fitting preprocessing on the full dataset, including validation folds
2. **Inconsistent Transformations** — Different transformations applied to train and test data
3. **Fragile Code** — Manual transformation sequences that break on data changes
4. **Invalid Cross-Validation** — Inflated CV scores from preprocessing that sees too much data

**Each produces models that work during development but fail in production.**

### The Solution: Pipelines

A Pipeline bundles preprocessing and modeling into a single object that guarantees:

✅ **Correct Order** — Transformations happen before modeling, always  
✅ **Training Data Only** — Preprocessing refits from scratch inside every CV fold  
✅ **Reproducibility** — Same pipeline produces same transformations every time  
✅ **Production Compatibility** — Save once, deploy with identical preprocessing  

---

## Learning Objectives

After completing this lesson, you will understand:

1. **What a Pipeline is** and how it works mechanically
2. **Why pipelines prevent data leakage** at the CV level (the mechanism, not just the rule)
3. **How to build pipelines** step-by-step from simple to complex
4. **ColumnTransformer** for mixed data types with nested pipelines
5. **Handling missing values** cleanly inside the pipeline
6. **Inspecting pipelines** to verify correctness and debug issues
7. **Integration with cross-validation and GridSearchCV** (complete end-to-end)
8. **Saving and loading pipelines** for deployment (single artifact)
9. **Common mistakes** and how each one causes silent failures
10. **Production-grade pattern** — nested pipelines, feature groups, complete workflow

---

## Notebook Contents

### Sections

1. **What Is a Pipeline?**
   - Mechanics and interface
   - Execution flow for fit() and predict()
   - Why transformers and estimators are different

2. **Why Pipelines Prevent Data Leakage (The Mechanism)**
   - Problem: Preprocessing before CV introduces leakage
   - Visual demonstration: leaked vs. clean CV scores
   - How the pipeline refits preprocessing inside each fold
   - Impact: actual difference in reported metrics

3. **Simple Pipeline (Scaling + Logistic Regression)**
   - Basic pattern: StandardScaler → LogisticRegression
   - Building and evaluating the pipeline
   - Using cross_val_score with the pipeline
   - Comparing against manual preprocessing

4. **Mixed Data Types with ColumnTransformer**
   - Creating a mixed-type dataset (numerical + categorical)
   - ColumnTransformer pattern for simultaneous different transformations
   - Why `handle_unknown="ignore"` matters for production
   - Why `remainder="drop"` prevents accidental column passthrough
   - Practical example with Gender, City, ContractType

5. **Handling Missing Values with Nested Pipelines**
   - Creating dataset with realistic missing values
   - Sub-pipeline pattern: imputation + scaling for numerical
   - Sub-pipeline pattern: imputation + encoding for categorical
   - Why median imputation (robust to outliers) vs. mean
   - Complete nested pipeline architecture
   - Verification that imputation values are learned from training only

6. **Inspecting and Debugging Pipelines**
   - Accessing individual steps by name
   - Verifying structure and learned parameters
   - Inspecting StandardScaler's learned means and scales
   - Viewing imputation values and feature names after transformation
   - Debugging: manual verification that imputation matches training stats

7. **Pipelines with GridSearchCV**
   - Double-underscore notation for tuning any hyperparameter
   - Tuning model hyperparameters through the pipeline
   - Tuning preprocessing hyperparameters alongside model
   - Example: GridSearchCV tuning max_depth, min_samples_leaf, n_estimators
   - Verification that all combinations are correctly cross-validated

8. **Saving and Loading Pipelines for Deployment**
   - Saving trained pipeline with joblib.dump
   - Loading pipeline for inference
   - Making predictions on new raw data (no manual preprocessing)
   - Why this eliminates preprocessing mismatch in production
   - Self-contained artifact advantage

9. **Common Mistakes to Avoid**
   - ❌ Scaling before train/test split
   - ❌ Manual preprocessing before cross_val_score
   - ❌ Forgetting handle_unknown on OneHotEncoder
   - ❌ Saving only the model, not the pipeline
   - ❌ Using remainder="passthrough" without intention
   - ❌ Tuning hyperparameters outside the pipeline
   - All with ✅ correct versions

10. **Practical Checklist Before Finalizing**
    - Data preparation checks
    - Pipeline construction checks
    - Model development checks
    - Evaluation and deployment checks

11. **When Pipelines Are Essential vs. Optional**
    - When to always use (production, CV, complex preprocessing)
    - When optional but still recommended

---

## Key Concepts Explained

### Data Leakage Mechanism

**Without Pipeline (❌ WRONG):**
```
scaler FITS on ALL training data [samples 0-999]
↓
cross_val_score splits:
  Fold 1: Train [200-999]                    Validate [0-199]  ← SCALER ALREADY KNOWS THESE
  Fold 2: Train [0-199, 400-999]             Validate [200-399] ← SCALER ALREADY KNOWS THESE
  ...
Result: CV scores are optimistically inflated (scaler statistics include fold samples)
```

**With Pipeline (✅ CORRECT):**
```
cross_val_score splits:
  Fold 1: Scaler FITS on [200-999] only → Transforms [0-199] with fold-only statistics
  Fold 2: Scaler REFITS on [0-199, 400-999] → Transforms [200-399] with new statistics
  ...
Result: CV scores are honest (each fold's validation data is truly unseen by preprocessing)
```

### Three-Tier Pipeline Architecture

**Simple Pipeline (1 level):**
```
StandardScaler → LogisticRegression
```

**Intermediate (1 level with ColumnTransformer):**
```
ColumnTransformer([
    ("num", StandardScaler(), ["Age", "Income"]),
    ("cat", OneHotEncoder(), ["Gender", "City"])
]) → RandomForestClassifier
```

**Production-Grade (2 levels with nested pipelines):**
```
ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), ["Age", "Income"]),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ]), ["Gender", "City"])
]) → RandomForestClassifier
```

### Double-Underscore Notation for Tuning

Access any parameter in the pipeline hierarchy:

```python
param_grid = {
    "model__max_depth": [4, 6, 8],                    # Model parameter
    "model__min_samples_leaf": [1, 5],               # Model parameter
    "preprocessor__num__imputer__strategy": ["mean", "median"]  # Preprocessing parameter
}
```

The nested structure flattens with `__`:
```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num (Pipeline)
│   │   ├── imputer (SimpleImputer) ← "preprocessor__num__imputer__strategy"
│   │   └── scaler (StandardScaler) ← "preprocessor__num__scaler__..."
│   └── cat (Pipeline)
│       ├── imputer (SimpleImputer) ← "preprocessor__cat__imputer__strategy"
│       └── encoder (OneHotEncoder) ← "preprocessor__cat__encoder__..."
└── model (RandomForestClassifier) ← "model__max_depth", etc.
```

---

## Practical Workflow

```
1. Raw Data
       ↓
2. Train/Test Split (✓ test set locked away)
       ↓
3. Define Pipelines
   ├── Numerical sub-pipeline: Imputer → Scaler
   ├── Categorical sub-pipeline: Imputer → Encoder
   └── Combined: ColumnTransformer + Model
       ↓
4. Cross-Validation
   └── Pipeline.fit(X_train, y_train, cv=5) ✓ no leakage
       ↓
5. GridSearchCV on Pipeline
   └── GridSearchCV(pipeline, param_grid, cv=5) ✓ correct tuning
       ↓
6. Test Evaluation
   └── best_estimator_.predict(X_test) ✓ exactly once
       ↓
7. Save Pipeline
   └── joblib.dump(best_estimator_, "pipeline.pkl")
       ↓
8. Deployment
   └── loaded_pipeline.predict(new_raw_data) ✓ reproducible
```

---

## Complete Production-Grade Example

```python
# 1. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 2. Define feature groups
num_features = ["Age", "Income", "Tenure"]
cat_features = ["Gender", "ContractType", "PaymentMethod"]

# 3. Sub-pipelines
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 4. Combine
preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# 5. Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# 6. Tune
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# 7. Evaluate on test
y_pred = grid.best_estimator_.predict(X_test)
test_score = accuracy_score(y_test, y_pred)

# 8. Save for deployment
joblib.dump(grid.best_estimator_, "production_pipeline.pkl")
```

---

## Common Mistakes Summary

| Mistake | Problem | Solution |
|---|---|---|
| Scale before train/test split | Test statistics influence scaler | Split first, then pipeline |
| Manual preprocessing before CV | CV-level leakage | Use pipeline with cross_val_score |
| Missing `handle_unknown` | Runtime crash on new categories | `OneHotEncoder(handle_unknown="ignore")` |
| Save model only, not pipeline | Preprocessing not reproducible | Save full pipeline with joblib |
| `remainder="passthrough"` | Unwanted columns reach model | Use `remainder="drop"` |
| Tune model outside pipeline | Reintroduces leakage | Run GridSearchCV on full pipeline |

---

## When Pipelines Are Mandatory

✅ **Always use when:**
- You have preprocessing that learns from data (scaling, imputation, encoding)
- You perform cross-validation or hyperparameter tuning
- You're deploying to production
- Working in a team (reproducibility)

⚠️ **Consider using even when:**
- Rapid exploratory analysis (wrapping costs almost nothing)
- Single-step preprocessing (still prevents refactoring later)

❌ **Only skip when:**
- Pure exploration with no CV or deployment, and you'll never touch the code again
- (Even then, adding a pipeline is a one-line refactor)

---

## Best Practices Checklist

**Before Finalizing Any Pipeline Project:**

□ Train/test split done BEFORE any pipeline fitting  
□ All preprocessing inside pipeline  
□ ColumnTransformer used for mixed types  
□ `handle_unknown="ignore"` on OneHotEncoder  
□ `remainder="drop"` explicit  
□ Numerical: median imputation (not mean)  
□ Categorical: most_frequent imputation  
□ Cross-validation on full pipeline  
□ GridSearchCV on full pipeline  
□ Full pipeline saved (preprocessing + model)  
□ Test set evaluated once, on raw data  
□ Pipeline loaded and tested before deployment  

---

## Key Takeaways

1. **Pipelines prevent leakage** — preprocessing refits inside each CV fold
2. **Pipelines ensure reproducibility** — same pipeline = same transformations
3. **Pipelines simplify deployment** — single artifact, no preprocessing mismatch
4. **Pipelines are composable** — work with cross_val_score, GridSearchCV, serialization
5. **Pipelines are self-documenting** — the pipeline shows the exact preprocessing workflow
6. **Pipelines catch mistakes** — common errors like premature scaling become obvious
7. **Pipelines are not optional** — professional ML requires them

**A model without a pipeline might be correct. A model inside a pipeline is provably correct — for leakage, at least.**

---

## References

- **Notebook:** `18_scikit_learn_pipelines_for_reproducible_ml.ipynb`
- **Scikit-learn Docs:** Pipeline API Reference
- **Scikit-learn Docs:** ColumnTransformer API Reference
- **User Guide:** Pipelines and Composite Estimators

---

## Related Lessons

- **Lesson 16:** GridSearchCV - Exhaustive grid search (use this with pipelines!)
- **Lesson 17:** RandomizedSearchCV - Efficient random search (use this with pipelines!)
- **Lesson 8–15:** Classification and regression evaluation
- **Lesson 1–7:** Raw data preparation and feature engineering

---

**Professional ML is built on pipelines. Start using them today.**
