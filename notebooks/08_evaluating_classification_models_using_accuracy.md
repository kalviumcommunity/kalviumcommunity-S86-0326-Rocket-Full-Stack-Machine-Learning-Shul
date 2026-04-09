# Evaluating Classification Models Using Accuracy

After training a classifier, the first question is often: how often is it correct?
Accuracy answers that directly.

## What Accuracy Means

Accuracy is the proportion of correct predictions:

accuracy = (number of correct predictions) / (total predictions)

For binary classification, using confusion matrix terms:

accuracy = (TP + TN) / (TP + TN + FP + FN)

Where:
- TP: predicted 1, actual 1
- TN: predicted 0, actual 0
- FP: predicted 1, actual 0
- FN: predicted 0, actual 1

## Why Accuracy Can Mislead

Accuracy is reliable when:
- classes are reasonably balanced
- false positives and false negatives have similar cost

Accuracy is misleading when:
- classes are imbalanced
- minority-class detection is critical (fraud, churn, disease)

A model that predicts only the majority class can have high accuracy and zero business value.

## Compute Accuracy in scikit-learn

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy:          {acc:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(classification_report(y_test, y_pred))
```

## Always Compare to Baseline

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

baseline_acc = accuracy_score(y_test, baseline_pred)
model_acc = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {baseline_acc:.3f}")
print(f"Model Accuracy:    {model_acc:.3f}")
print(f"Improvement:       {model_acc - baseline_acc:+.3f}")
```

If baseline and model accuracy are close, inspect recall, F1, ROC-AUC, and confusion matrix before concluding improvement.

## Accuracy and Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
```

Accuracy uses only total correct predictions (TP + TN). It does not reveal the FP/FN trade-off.

## Balanced Accuracy

Balanced Accuracy averages recall across classes:

balanced_accuracy = (recall_class_0 + recall_class_1) / 2

This prevents majority-class dominance and is a better default on imbalanced datasets.

## Cross-Validation with Accuracy

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
cv_bal_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="balanced_accuracy")

print(f"CV Accuracy:          {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
print(f"CV Balanced Accuracy: {cv_bal_acc.mean():.3f} +/- {cv_bal_acc.std():.3f}")
```

## Practical Checklist

- Accuracy computed on test data, not training data
- Baseline accuracy reported
- Balanced accuracy included for imbalanced data
- Confusion matrix inspected
- Classification report reviewed
- Cross-validation mean and std reported

## Repository Alignment

In this repository:
- evaluation now reports both accuracy and balanced accuracy
- confusion matrix counts (TN, FP, FN, TP) are included in metrics output
- baseline and model are compared side-by-side

Use accuracy as a starting point, not the only decision metric.
