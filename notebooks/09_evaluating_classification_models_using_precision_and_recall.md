# Evaluating Classification Models Using Precision and Recall

Accuracy tells you overall correctness, but it hides error type. In real systems, false positives and false negatives often have very different costs.

Precision and Recall make those costs visible.

## Confusion Matrix Refresher

For binary classification:

- TP: predicted positive, actually positive
- FP: predicted positive, actually negative
- FN: predicted negative, actually positive
- TN: predicted negative, actually negative

## Precision

Precision answers:

Of all predicted positives, how many were truly positive?

precision = TP / (TP + FP)

Use precision when false alarms are expensive (for example, legal actions, moderation over-blocking, low-trust alerts).

## Recall

Recall answers:

Of all actual positives, how many did we catch?

recall = TP / (TP + FN)

Use recall when missed positives are expensive (for example, fraud misses, disease misses, security misses).

## Precision-Recall Trade-Off

Most classifiers output probabilities. A threshold converts probability into class:

- Higher threshold -> fewer positive predictions -> precision usually increases, recall usually decreases
- Lower threshold -> more positive predictions -> recall usually increases, precision usually decreases

Default threshold 0.5 is not sacred. Choose a threshold that matches business risk.

## Compute in scikit-learn

```python
from sklearn.metrics import precision_score, recall_score, classification_report

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
```

## Baseline Comparison Is Mandatory

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("Baseline Precision:", precision_score(y_test, baseline_pred, zero_division=0))
print("Baseline Recall:   ", recall_score(y_test, baseline_pred, zero_division=0))
```

A majority baseline often has near-zero recall for the minority class. Any deployable model should beat that floor clearly.

## Threshold Tuning

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_custom = (y_prob >= threshold).astype(int)

print(f"Threshold: {threshold}")
print(f"Precision: {precision_score(y_test, y_custom, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_custom, zero_division=0):.3f}")
```

## Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
```

Use this curve to pick the threshold that satisfies a target recall while maximizing precision.

## Cross-Validation Stability

```python
from sklearn.model_selection import cross_val_score

cv_precision = cross_val_score(model, X_train, y_train, cv=5, scoring="precision")
cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")

print(f"CV Precision: {cv_precision.mean():.3f} +/- {cv_precision.std():.3f}")
print(f"CV Recall:    {cv_recall.mean():.3f} +/- {cv_recall.std():.3f}")
```

## Practical Checklist

- Report precision and recall together
- Inspect confusion matrix counts, not just summary metrics
- Compare against majority baseline
- Tune threshold based on error costs
- Validate stability with cross-validation

## Repository Alignment

This repository now includes:
- precision/recall metrics in baseline and model evaluation
- threshold analysis from model probabilities
- best-threshold search for a target recall
- cross-validation summary for precision and recall
- precision-recall curve data export to reports
