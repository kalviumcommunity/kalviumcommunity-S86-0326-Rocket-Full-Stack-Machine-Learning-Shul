# Interpreting Feature Importance from Tree-Based Models

After training a Decision Tree, Random Forest, or Gradient Boosting model, a natural question follows:

- which features actually matter?

Feature importance turns a tree-based model from a prediction machine into an insight tool. It helps you identify useful predictors, simplify the feature set, and explain model behavior to others.

But importance scores are easy to misuse. They can be biased, unstable, and misleading if you treat them as causal truth.

## 1. What Feature Importance Means

Tree-based models assign importance based on how much a feature helped reduce impurity during training.

For a split on feature $f$ at a node, the impurity reduction is credited to that feature. Across the whole tree, those reductions are summed and normalized so the final scores add up to 1.

In practical terms:

- a feature with importance near 0 was rarely useful during splitting
- a feature with importance near 1 drove most of the model's decisions

This is often called Mean Decrease in Impurity (MDI), Gini importance, or simply `feature_importances_` in scikit-learn.

## 2. Why High Importance Happens

A feature tends to receive high importance when it:

- appears near the root of the tree
- splits a large number of samples
- produces a large reduction in impurity

The root effect matters. A strong split near the top can influence many samples and therefore collect more importance than several weak splits near the bottom.

## 3. Extracting Importance from a Single Tree

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": tree.feature_importances_,
    }
).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))
```

Reading the result:

- scores sum to 1.0 across all features
- a score of 0.0 means the feature was never used in a split
- a score of 0.35 means the feature accounts for 35 percent of the total impurity reduction

## 4. Visualizing Importance

```python
plt.figure(figsize=(9, 5))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue",
    edgecolor="white",
)
plt.xlabel("Importance Score")
plt.title("Feature Importance - Decision Tree")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

What to look for:

- one feature dominates the plot: the model may be highly dependent on a single predictor
- many features are near zero: some may be removable, but validate before dropping them
- an unexpected feature ranks highly: investigate for proxy effects or leakage

## 5. Feature Importance in Random Forests

Single-tree importance is unstable. A slightly different sample can produce a different ranking.

Random Forests improve stability by averaging importance across many trees.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf.feature_importances_,
    }
).sort_values("Importance", ascending=False)

print(rf_importance.to_string(index=False))
```

For model interpretation and feature selection, Random Forest importance is usually more reliable than a single tree.

## 6. What Importance Does Not Mean

Before acting on a ranking, be clear about three limits:

- importance is not causation
- importance is conditional on the other features present
- importance is specific to the model and sample used to train it

A feature can be important because it is a useful predictor, not because it causes the target.

## 7. Bias Toward High-Cardinality Features

Impurity-based importance often favors features with many unique values.

Why this happens:

- more unique values create more possible split points
- more split points increase the chance of finding a large impurity reduction by chance

This means continuous features and high-cardinality categoricals can look artificially important.

If an ID-like feature ranks highly, treat that as a warning sign.

## 8. Correlated Features Can Split Importance

When two features are highly correlated, the model often uses one and gives the other little or no importance.

That does not mean the low-ranked feature is useless. It may simply have lost the competition for the first good split.

Check feature correlations before removing anything.

```python
import seaborn as sns

corr = X_train.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

If two features are strongly correlated, low importance on one of them may not be a sign that it is irrelevant.

## 9. Permutation Importance

Permutation importance measures how much model performance drops when a single feature is shuffled.

The idea is simple:

1. train the model
2. measure baseline performance on held-out data
3. shuffle one feature at a time
4. measure performance again
5. importance equals baseline performance minus shuffled performance

This works because shuffling destroys the relationship between that feature and the target while leaving everything else unchanged.

```python
import pandas as pd
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy",
)

perm_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }
).sort_values("Importance", ascending=False)

print(perm_df.to_string(index=False))
```

Important rule:

- use the test set or another held-out set, not the training set

Training data would reflect memorization instead of generalization.

## 10. Negative Permutation Importance

If shuffling a feature improves performance, the feature may have been hurting the model.

That often means:

- the feature added noise
- the feature encouraged overfitting
- the feature is unstable or redundant

Negative permutation importance is a strong signal to inspect the feature carefully.

## 11. Comparing MDI and Permutation Importance

| Dimension | Impurity-Based Importance | Permutation Importance |
|---|---|---|
| Based on | training splits | held-out performance |
| Speed | very fast | slower |
| Bias | favors high-cardinality features | much less biased |
| Correlated features | can hide one feature behind another | often fairer |
| Model agnostic | no | yes |

Best practice:

- use MDI for a quick first look
- use permutation importance for decisions that matter
- investigate large disagreements between the two

## 12. Worked Interpretation Example

Suppose a churn model returns this ranking:

| Feature | MDI | Permutation |
|---|---|---|
| Tenure | 0.35 | 0.31 |
| MonthlyCharges | 0.28 | 0.24 |
| ContractType | 0.19 | 0.22 |
| PaymentMethod | 0.12 | 0.14 |
| TotalCharges | 0.05 | 0.02 |
| Gender | 0.01 | -0.01 |

Interpretation:

- Tenure and MonthlyCharges are strong, consistent predictors
- ContractType is also likely useful, even if the tree underweighted it a little
- TotalCharges is probably redundant if it overlaps with other features
- Gender may be adding noise and can be tested for removal

Always retrain before permanently removing any feature.

## 13. Practical Uses of Feature Importance

Feature importance is most useful for:

- debugging the model
- simplifying the feature set
- communicating model reasoning
- discovering potentially leaky fields
- guiding feature engineering

If a feature you expected to matter ranks near zero, check whether it was encoded correctly or overshadowed by a correlated variable.

## 14. Common Mistakes

- treating importance as causation
- removing features based only on MDI
- ignoring correlation among predictors
- using training data for permutation importance
- reporting importance from a poorly performing model
- interpreting a single unconstrained tree as if it were stable

## 15. Practical Checklist

Before reporting feature importance, confirm that:

- the model performs well on held-out data
- importance was extracted from a tuned model
- correlated features were checked
- MDI and permutation importance were compared
- a ranked table and bar chart were produced
- domain knowledge was used to sanity-check the results

## 16. Repository Alignment

This project already contains the pieces needed for feature-importance analysis:

- `src/train.py` trains the core tree-based classifier used in the lesson examples
- `src/evaluate.py` provides the held-out metrics needed before interpreting importance
- `notebooks/14_training_decision_tree_model.md` introduces the tree behavior that generates impurity-based scores
- `main.py` already uses cross-validation and test evaluation, which are useful guardrails before interpretation

That makes feature importance a natural extension of the tree lesson.

## 17. Final Takeaway

Feature importance helps explain what tree-based models rely on, but it must be interpreted carefully.

- MDI is fast and convenient, but biased
- permutation importance is slower, but more trustworthy
- correlated and high-cardinality features require extra caution

Use importance to generate hypotheses, not final conclusions.
Validate those hypotheses with held-out data, retraining, and domain knowledge.