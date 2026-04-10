# Training a K-Nearest Neighbors (KNN) Model

After linear and logistic regression, K-Nearest Neighbors (KNN) is a useful next model because it works in a completely different way.

It does not learn coefficients.
It does not fit a global decision boundary.
It does not optimize a traditional loss function during training.

Instead, KNN memorizes the training set and predicts by looking at the closest examples.

## 1. What KNN Is

KNN is an instance-based, or lazy, learning algorithm.

For a new sample, it:

1. Chooses a number of neighbors, `K`
2. Measures the distance to every training sample
3. Selects the `K` closest samples
4. Aggregates their targets

For classification:

- use majority vote

For regression:

- use the average target value

The training step is fast because the model mostly stores the data.

## 2. Why KNN Is Conceptually Useful

KNN makes the idea of similarity explicit.

It is easy to explain because you can point to the actual neighbors that influenced a prediction. That makes it a strong teaching algorithm even when it is not the best production choice.

## 3. Distance Metrics

Distance defines what “nearest” means.

Common options:

- Euclidean distance: straight-line distance
- Manhattan distance: sum of absolute differences
- Minkowski distance: general family that includes Euclidean and Manhattan
- Cosine distance/similarity: useful when direction matters more than magnitude

For most tabular problems, Euclidean distance is the default starting point.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
```

## 4. Feature Scaling Is Mandatory

KNN is distance-based, so features on large numeric scales dominate the distance calculation.

If one feature ranges from 0 to 1 and another ranges from 0 to 1,000,000, the second feature will overwhelm the first.

Use a pipeline so scaling happens safely on the training data only.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
```

Rule:

- never use KNN without scaling

## 5. Choosing K

`K` controls the bias-variance trade-off.

Small `K`:

- low bias
- high variance
- tends to overfit

Large `K`:

- high bias
- low variance
- tends to underfit

The right value depends on the dataset, so choose it with cross-validation.

## 6. KNN for Classification

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

param_grid = {"knn__n_neighbors": range(1, 31)}
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best K: {grid.best_params_['knn__n_neighbors']}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))
```

## 7. Compare Against a Baseline

Always compare KNN to a simple baseline.

```python
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print(f"Baseline Accuracy: {accuracy_score(y_test, baseline_pred):.3f}")
print(f"KNN Accuracy:      {accuracy_score(y_test, y_pred):.3f}")
```

If KNN cannot beat the baseline, the features may be noisy, poorly scaled, or too high-dimensional.

## 8. Cross-Validation and K Search

```python
import matplotlib.pyplot as plt

k_values = range(1, 31)
train_scores = grid.cv_results_["mean_train_score"]
cv_scores = grid.cv_results_["mean_test_score"]

plt.plot(k_values, train_scores, label="Train Accuracy", linestyle="--")
plt.plot(k_values, cv_scores, label="CV Accuracy", marker="o")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN: K vs Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

This curve helps you see whether the model is overfitting at small `K` or underfitting at large `K`.

## 9. KNN for Regression

KNN also works for regression.

```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5)),
])

pipeline_reg.fit(X_train, y_train)
y_pred = pipeline_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

print(f"KNN RMSE: {rmse:.3f}")
print(f"KNN R2:   {r2:.3f}")
```

For regression, KNN predicts the average of neighboring target values.

## 10. Curse of Dimensionality

KNN weakens as the number of features grows.

In high dimensions:

- distances become less informative
- the nearest neighbors are not much closer than the rest
- much more data is needed to preserve local structure

This is the curse of dimensionality.

If KNN struggles on a wide feature set, try feature selection or dimensionality reduction first.

```python
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
```

## 11. Strengths of KNN

- simple to explain
- no distributional assumptions
- naturally captures local non-linearity
- useful on small, well-scaled datasets
- prediction can be interpreted through neighbors

## 12. Weaknesses of KNN

- slow at prediction time
- memory-intensive
- sensitive to irrelevant features
- sensitive to scale
- degrades in high dimensions

## 13. Common Mistakes

- forgetting to scale features
- choosing `K` without cross-validation
- using KNN on very large datasets
- keeping irrelevant features in the input set
- skipping the baseline comparison
- using KNN for low-latency production systems

## 14. Practical Checklist

Before reporting KNN results, confirm that:

- features were scaled in a pipeline
- `K` was selected using cross-validation
- K vs accuracy behavior was inspected
- baseline performance was compared
- classification reports or regression metrics were reviewed
- dimensionality is reasonable for distance-based learning

## 15. Repository Alignment

For this project, KNN is best understood as a benchmark and teaching model for tabular classification or regression workflows.

It reinforces the same core lessons as the rest of the repository:

- split before fitting preprocessing
- scale numerical features correctly
- compare against a baseline
- validate with cross-validation rather than guesswork

## 16. Final Takeaway

KNN is simple, but it teaches important machine learning discipline.

It forces you to think about similarity, scaling, local structure, and the cost of high-dimensional data.

Use it carefully.
Scale properly.
Choose `K` deliberately.
Always compare against a baseline.