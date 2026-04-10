# Creating and Interpreting a Confusion Matrix

After training a classifier and reporting Accuracy, Precision, Recall, and F1-score, one tool still gives the most complete picture of model behavior:

The confusion matrix.

Aggregate metrics compress behavior into one number. That is useful for ranking models, but it can hide where errors occur. The confusion matrix shows exactly how predictions are distributed across actual vs predicted classes.

## 1. What a Confusion Matrix Is

A confusion matrix is a table of actual labels vs predicted labels.

For binary classification, it has four cells:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | TP               | FN                 |
| Actual Negative | FP               | TN                 |

Definitions:

- TP (True Positive): predicted positive, actually positive
- TN (True Negative): predicted negative, actually negative
- FP (False Positive): predicted positive, actually negative (false alarm)
- FN (False Negative): predicted negative, actually positive (missed case)

The matrix answers the most important question:

Where exactly is the model correct, and where exactly is it wrong?

## 2. Interpreting the Four Components

### True Positives (TP)

Correct positive detections. Increase TP when the goal is catching real positives.

### True Negatives (TN)

Correct negative rejections. TN is often large in imbalanced datasets and can inflate accuracy.

### False Positives (FP)

False alarms. Typical costs are operational and reputational.

Examples:

- valid transaction blocked
- legitimate email marked spam
- unnecessary manual review

### False Negatives (FN)

Missed positives. Typical costs are severe or irreversible.

Examples:

- missed fraud
- missed disease case
- missed security intrusion

In many high-stakes use cases, FN and FP are not equally costly. The confusion matrix makes that asymmetry explicit.

## 3. Connection to Classification Metrics

All major classification metrics come from TP, TN, FP, FN.

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2TP / (2TP + FP + FN)

Notice F1 does not include TN. That is one reason F1 is often preferred for imbalanced positive-class tasks.

## 4. Worked Example

Suppose the matrix is:

|               | Predicted Fraud | Predicted Legitimate |
|---------------|-----------------|----------------------|
| Actual Fraud  | 40              | 10                   |
| Actual Legitimate | 5          | 145                  |

Then:

- TP = 40
- FN = 10
- FP = 5
- TN = 145

Interpretation:

- false alarms are low (good precision)
- some fraud is still missed (recall not perfect)
- threshold choice depends on cost of missed fraud vs analyst load

## 5. Creating a Confusion Matrix in scikit-learn

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Binary output order in scikit-learn:

```text
[[TN FP]
 [FN TP]]
```

Rows are actual labels, columns are predicted labels.

You can unpack directly:

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
```

## 6. Visualizing the Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Legitimate", "Fraud"],
    cmap="Blues",
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
```

Normalized by actual class (row-wise):

```python
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Legitimate", "Fraud"],
    normalize="true",
    cmap="Blues",
)
plt.title("Normalized Confusion Matrix")
plt.show()
```

Use normalization when classes are imbalanced.

## 7. Imbalanced Dataset Interpretation

If positives are rare, a model can have high accuracy but still fail on positives.

Example matrix:

```text
[[190   0]
 [ 10   0]]
```

The model predicts all negatives:

- high accuracy
- zero TP
- recall = 0
- F1 = 0

Always inspect the minority-class row.

## 8. Multi-Class Confusion Matrices

For $K$ classes, the matrix is $K \times K$.

- diagonal cells: correct per class
- off-diagonal cells: specific confusions

Use off-diagonal concentration to target feature engineering and data collection.

## 9. Threshold Effects

Changing threshold directly reshapes matrix cells.

Lower threshold:

- TP tends to increase
- FP tends to increase
- FN tends to decrease

Higher threshold:

- FP tends to decrease
- TP tends to decrease
- FN tends to increase

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_prob = model.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    y_custom = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_custom).ravel()
    print(f"Threshold {threshold:.1f} | TP:{tp:4d} FP:{fp:4d} FN:{fn:4d} TN:{tn:4d}")
```

This converts threshold tuning into a business decision about tolerable false alarms vs missed positives.

## 10. Common Mistakes

- not generating a confusion matrix at all
- misreading row/column order
- focusing only on diagonal values
- skipping normalized matrix on imbalanced data
- comparing raw counts across different test sizes
- skipping baseline confusion matrix comparison

## 11. Practical Checklist

Before reporting classifier quality:

- confusion matrix computed on test data
- TP/TN/FP/FN labeling verified
- minority-class row inspected
- normalized matrix reviewed (if imbalanced)
- matrix connected to Precision/Recall/F1 interpretation
- threshold effect inspected at multiple operating points
- baseline confusion matrix compared

## 12. Repository Alignment

This project now exports confusion-matrix artifacts directly:

- overall confusion counts and normalized true-class rates by model variant
- threshold-wise confusion matrix counts and rates
- baseline, default-threshold model, and optimized-threshold model comparison

These are saved in the reports folder and aligned with metrics.json.

## 13. Final Takeaway

The confusion matrix is the evidence behind every classification metric.

Accuracy, Precision, Recall, and F1 are summaries derived from it. The matrix itself tells the full story: what the model catches, what it misses, and what it falsely flags.

Inspect it before threshold tuning. Inspect it before deployment. Inspect it before claiming success.
