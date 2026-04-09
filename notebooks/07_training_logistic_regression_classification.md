# Training a Logistic Regression Classification Model

After regression, classification predicts discrete classes (for example: churn vs no churn).
This lesson introduces Logistic Regression as the first strong, interpretable classification benchmark.

## Why Logistic Regression

Logistic Regression predicts:
- a class label (0 or 1)
- and the probability of class 1

It starts with a linear score:

z = w1*x1 + w2*x2 + ... + wn*xn + b

Then converts the score to probability with the sigmoid:

p = 1 / (1 + exp(-z))

This keeps outputs in [0, 1], unlike linear regression.

## Why Not Linear Regression for Classification

Linear Regression is a poor fit for binary targets because:
- predictions can be below 0 or above 1
- it uses MSE, which is not designed for probabilistic binary outcomes
- it does not model class probabilities with the right shape near the decision boundary

## Training Objective: Log Loss

Logistic Regression minimizes binary cross-entropy (log loss):

loss = -(y * log(p) + (1 - y) * log(1 - p))

Key intuition:
- confidently correct predictions get low loss
- confidently wrong predictions get very high loss

## Practical Workflow (scikit-learn)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1) Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:, 1]

# 3) Logistic Regression with scaling
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)
model_pred = pipeline.predict(X_test)
model_prob = pipeline.predict_proba(X_test)[:, 1]

# 4) Evaluate
print("Baseline AUC:", roc_auc_score(y_test, baseline_prob))
print("Model Accuracy:", accuracy_score(y_test, model_pred))
print("Model AUC:", roc_auc_score(y_test, model_prob))
print(classification_report(y_test, model_pred))

# 5) Cross-validation stability
cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print("CV AUC mean/std:", cv_auc.mean(), cv_auc.std())
```

## Coefficients and Odds Ratios

Logistic coefficients are in log-odds space. Convert to odds ratios with exp(coef):

```python
coef = pipeline.named_steps["model"].coef_[0]
odds_ratio = np.exp(coef)
```

Interpretation example:
- coefficient +0.69 -> odds ratio about 2.0 (odds double)
- coefficient -0.69 -> odds ratio about 0.5 (odds halve)

## Checklist Before Declaring Success

- Model beats majority baseline on accuracy and ROC-AUC
- Precision/recall trade-off is acceptable for the business goal
- Stratified split is used
- Scaling is inside a pipeline (no leakage)
- `max_iter` is high enough (no convergence warnings)
- Cross-validation mean and standard deviation are reported
- Coefficients are directionally sensible

## Notes on This Repository

This repository implements the same workflow in the production code path:
- baseline comparison
- Logistic Regression model training
- ROC-AUC and classification metrics
- cross-validation stability tracking
- coefficient and odds-ratio export
