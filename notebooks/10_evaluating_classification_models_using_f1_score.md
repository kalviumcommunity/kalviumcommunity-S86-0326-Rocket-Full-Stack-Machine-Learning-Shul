# Evaluating Classification Models Using F1-Score

After learning Precision and Recall, the natural question is:

What if we care about both?

In many real-world classification systems, optimizing only one of these metrics can fail in production. A model with very high precision but very low recall misses too many real positives. A model with very high recall but very low precision overwhelms operations with false alarms.

F1-Score exists to enforce balance.

## 1. Definition of F1-Score

F1 is the harmonic mean of Precision and Recall:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

Substituting metric definitions:

F1 = 2*TP / (2*TP + FP + FN)

Important consequence:

- F1 depends on TP, FP, and FN
- TN does not affect F1
- F1 is focused on positive-class performance

## 2. Why Harmonic Mean, Not Arithmetic Mean

For two numbers a and b:

- Arithmetic mean: (a + b) / 2
- Harmonic mean: 2ab / (a + b)

The harmonic mean punishes imbalance more strongly.

Example:

- Precision = 0.90
- Recall = 0.10

Then:

- Arithmetic mean = 0.50 (looks moderate)
- F1 (harmonic mean) = 0.18 (near-failure)

That behavior is desirable because one weak component should not be hidden by one strong component.

## 3. Intuition

- Precision: When we predict positive, how often are we right?
- Recall: Of all actual positives, how many did we catch?
- F1: Are we balancing both correctness and coverage?

F1 drops rapidly when either Precision or Recall collapses.

## 4. Worked Example

If:

- Precision = 0.80
- Recall = 0.60

Then:

F1 = 2 * 0.8 * 0.6 / (0.8 + 0.6) = 0.686

Approximate interpretation: moderate balance, not yet strong in both dimensions.

## 5. Why Accuracy Can Mislead

On imbalanced data, accuracy can be high while minority detection is poor.

Example:

- 95% negatives, 5% positives
- Model predicts all negatives

Metrics:

- Accuracy = 0.95
- Precision = 0
- Recall = 0
- F1 = 0

F1 correctly reports failure on the positive class.

## 6. When F1 Is the Right Primary Metric

Use F1 when:

- dataset is imbalanced
- both FP and FN matter
- you need one selection metric for model comparison and tuning
- minority-class detection is operationally important

Common domains:

- fraud detection
- spam filtering
- disease screening
- intrusion detection
- churn prediction

## 7. When F1 Is Not Ideal

Prefer other metrics when:

- FP and FN costs are very asymmetric (use F-beta)
- TN matters strongly (consider MCC or balanced accuracy)
- probability calibration is required (Brier score, calibration curves)
- threshold-independent ranking is needed (PR-AUC, ROC-AUC)

## 8. Computing F1 in scikit-learn

```python
from sklearn.metrics import f1_score, classification_report

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")

print(classification_report(y_test, y_pred, digits=3))
```

Always report Precision and Recall with F1.

## 9. Baseline Comparison

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("=== Baseline ===")
print(f"F1-Score: {f1_score(y_test, baseline_pred, zero_division=0):.3f}")
print(classification_report(y_test, baseline_pred, zero_division=0))

print("=== Model ===")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

Baseline gives the minimum acceptable floor.

## 10. Multi-Class F1: Macro, Micro, Weighted

### Macro F1

Average per-class F1 equally:

```python
f1_macro = f1_score(y_test, y_pred, average="macro")
```

Use when every class matters equally.

### Micro F1

Aggregate TP/FP/FN globally, then compute F1:

```python
f1_micro = f1_score(y_test, y_pred, average="micro")
```

Use when overall volume matters most. In standard single-label classification, micro F1 equals accuracy.

### Weighted F1

Average per-class F1 weighted by class support:

```python
f1_weighted = f1_score(y_test, y_pred, average="weighted")
```

Useful as context, but can hide minority-class failure if majority support is large.

## 11. Threshold Optimization for F1

Default threshold 0.5 is not always best for F1.

```python
import numpy as np
from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 0.9, 0.05)
val_prob = model.predict_proba(X_val)[:, 1]

val_f1s = [f1_score(y_val, (val_prob >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(val_f1s)]

# Final evaluation only on test

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_t).astype(int)
print(f"Best threshold: {best_t:.2f}")
print(f"Test F1: {f1_score(y_test, test_pred):.3f}")
```

Critical rule:

- tune threshold on validation
- evaluate once on test
- never optimize on test

## 12. Cross-Validation with F1

```python
from sklearn.model_selection import cross_val_score

cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"Mean CV F1: {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

cv_f1_macro = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
print(f"Mean CV Macro F1: {cv_f1_macro.mean():.3f} +/- {cv_f1_macro.std():.3f}")
```

High fold variance indicates unstable minority-class behavior.

## 13. Common Mistakes

- reporting F1 without Precision/Recall
- using weighted F1 alone on imbalanced data
- not specifying averaging in multi-class settings
- tuning threshold on test data
- reporting only training F1
- ignoring baseline F1

## 14. Practical Checklist Before Reporting F1

- F1 computed on test (or cross-validation), not training
- baseline F1 included
- Precision and Recall reported
- full classification report inspected
- averaging method specified and justified
- threshold tuned on validation (if tuned)
- CV mean and std reported
- interpretation tied to business cost and impact

## 15. Repository Alignment

This repository now supports leakage-safe F1 evaluation:

- baseline and model F1 are reported
- 5-fold CV F1 summary is reported
- threshold search for F1 is performed on validation data
- optimized threshold is applied once to test data
- default-threshold and optimized-threshold test metrics are both saved

## 16. Final Takeaway

Precision captures trust.
Recall captures coverage.
F1 captures the balance required for real operational usefulness.

Use F1 thoughtfully, compare against baseline, and always validate threshold decisions without test leakage.
