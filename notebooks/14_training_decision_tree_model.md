# Training a Decision Tree Model

Decision Trees work differently from the linear and distance-based models covered earlier in the project.

They do not fit a straight line.
They do not measure geometric distance.
They do not assume the relationship between features and target is smooth.

Instead, they learn by repeatedly asking simple binary questions like:

- Is `tenure <= 12`?
- Is `MonthlyCharges > 80`?
- Is `Contract = Month-to-month`?

That makes the model easy to read, but also easy to overfit if it is left unconstrained.

## 1. What a Decision Tree Is

A Decision Tree recursively partitions the feature space into smaller regions and assigns a prediction to each region.

For classification, each leaf predicts the dominant class in that region.
For regression, each leaf predicts the mean target value of the training samples in that region.

The model behaves like a flowchart:

1. start at the root node
2. choose the best split
3. send samples left or right
4. repeat until a stopping rule is reached

The result is a set of human-readable rules that can be traced from root to leaf.

## 2. Why Decision Trees Are Useful

Decision Trees have several properties that make them valuable in tabular machine learning:

- they are naturally non-linear
- they handle feature interactions automatically
- they do not require feature scaling
- they can be visualized and explained to non-technical stakeholders
- they work with mixed feature types once categorical values are encoded

That makes them a strong teaching model and a useful baseline for more advanced tree ensembles.

## 3. How Splits Are Chosen

At each node, the tree scans possible splits and selects the one that produces the biggest improvement in purity.

For classification, purity means that one class dominates the node.
For regression, purity means that target values in the node are similar.

The process is greedy:

- evaluate candidate splits
- measure impurity reduction
- choose the best local split
- recurse on the child nodes

The tree does not look ahead to future splits. That is one reason it can make locally good but globally suboptimal decisions.

## 4. Impurity Measures for Classification

Two common impurity criteria are used in scikit-learn:

### Gini Impurity

Gini impurity is the default for `DecisionTreeClassifier`.

- Gini = 0 means the node is perfectly pure
- Gini is highest when classes are evenly mixed

### Entropy

Entropy measures disorder.

- entropy = 0 means the node is perfectly pure
- entropy is highest when classes are evenly mixed

In practice, Gini and entropy usually produce similar trees. Gini is often a little faster.

```python
from sklearn.tree import DecisionTreeClassifier

tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
```

## 5. Impurity for Regression

For regression trees, split quality is based on target variance within the child nodes.

The tree chooses the split that minimizes the weighted average variance of the children.

Each leaf predicts the mean target value of the samples it contains.

That means regression trees cannot extrapolate beyond the observed target range. They produce piecewise constant predictions rather than smooth curves.

## 6. Tree Growth and Stopping Criteria

Trees grow recursively until a stopping condition is met.

Important stopping controls include:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- no impurity improvement
- pure node

Without constraints, a tree can keep splitting until it memorizes the training set.

That gives extremely high training accuracy and poor test performance.

## 7. Overfitting in Decision Trees

Decision Trees are high-variance learners by default.

An unconstrained tree can easily produce this pattern:

- training accuracy near 100 percent
- test accuracy much lower
- large train-test gap

That is the classic overfitting signature.

The tree is not learning a stable rule. It is memorizing sample-specific quirks.

## 8. Controlling Tree Complexity

The main levers for controlling bias and variance are:

| Parameter | Increasing It Does | Bias | Variance |
|---|---|---|---|
| `max_depth` | Allows deeper partitions | decreases | increases |
| `min_samples_split` | Makes splitting harder | increases | decreases |
| `min_samples_leaf` | Forces larger leaves | increases | decreases |
| `max_features` | Limits features considered at each split | slightly increases | decreases |

These controls all push the model toward a better generalization balance.

## 9. Classification Workflow in scikit-learn

```python
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42,
)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(f"Baseline Accuracy: {accuracy_score(y_test, baseline_pred):.3f}")
print(f"Tree Accuracy:     {accuracy_score(y_test, y_pred):.3f}")
print(f"Train Accuracy:    {tree.score(X_train, y_train):.3f}")
print(f"Train/Test Gap:    {tree.score(X_train, y_train) - accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, digits=3))

cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

Always compare the tree against a baseline and always print the train-test gap.

## 10. Visualizing the Tree

Decision Trees are one of the most interpretable models in machine learning because you can visualize the full rule structure.

```python
plt.figure(figsize=(16, 8))
plot_tree(
    tree,
    feature_names=X.columns.tolist(),
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree - Churn Prediction")
plt.tight_layout()
plt.show()
```

The visualization shows:

- the feature and threshold used at each split
- the number of samples in each node
- class distribution in each region
- how pure or mixed each split became

That makes Decision Trees especially useful when interpretability matters.

## 11. Cross-Validation for Depth Selection

Never choose tree depth by guessing.

Use cross-validation to find the point where validation performance peaks before overfitting starts.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": range(1, 21)}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True,
)
grid.fit(X_train, y_train)

print(f"Best depth: {grid.best_params_['max_depth']}")
print(f"Best CV score: {grid.best_score_:.3f}")

depths = range(1, 21)
train_scores = grid.cv_results_["mean_train_score"]
cv_scores = grid.cv_results_["mean_test_score"]

plt.figure(figsize=(9, 5))
plt.plot(depths, train_scores, label="Train Accuracy", linestyle="--", marker="o")
plt.plot(depths, cv_scores, label="CV Accuracy", marker="o")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Depth vs Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

The expected pattern is familiar:

- shallow trees underfit
- deeper trees fit the training set better
- validation accuracy peaks and then falls if the tree becomes too complex

## 12. Regression Workflow

Decision Trees also work for regression.

```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42,
)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

print(f"Tree RMSE:   {rmse:.3f}")
print(f"Tree R2:     {r2:.3f}")
print(f"Baseline R2: {r2_score(y_test, baseline.predict(X_test)):.3f}")
print(f"Train R2:    {tree_reg.score(X_train, y_train):.3f}")
```

Regression trees are useful when the relationship is highly non-linear, but their step-function predictions can be limiting when smooth extrapolation matters.

## 13. Strengths of Decision Trees

- no feature scaling required
- non-linear behavior is captured naturally
- feature interactions are discovered automatically
- the model can be visualized as a readable flowchart
- mixed feature types can be handled with appropriate preprocessing

## 14. Weaknesses of Decision Trees

- high variance if unconstrained
- unstable to small data changes
- poor extrapolation for regression
- axis-aligned splits only
- often outperformed by ensembles such as Random Forests and Gradient Boosting

## 15. Common Mistakes

- leaving `max_depth` unconstrained
- skipping baseline comparison
- ignoring the train-test gap
- treating feature importance as causality
- using a single tree as the final production model when an ensemble would be better

## 16. Practical Checklist

Before reporting results, confirm that:

- the train-test split was done correctly
- classification used stratification
- the tree was compared to a baseline
- train accuracy and test accuracy were both reported
- depth or leaf settings were validated with cross-validation
- the tree visualization was inspected
- feature importances were sanity-checked against domain knowledge

## 17. Repository Alignment

This project already includes the preprocessing and evaluation structure needed for Decision Trees:

- `src/data_preprocessing.py` handles clean splits before modeling
- `src/feature_engineering.py` prepares categorical and numerical columns
- `src/evaluate.py` supports classification metrics and confusion-matrix inspection
- `notebooks/13_understanding_bias_and_variance_through_model_behavior.md` explains the overfitting behavior that trees are especially prone to

That makes Decision Trees a natural next lesson in the curriculum.

## 18. Final Takeaway

Decision Trees are powerful because they are non-linear, interpretable, and easy to explain.

They are also dangerous if left unconstrained, because they can memorize the training data very quickly.

Use them with stopping criteria.
Tune depth with cross-validation.
Always compare against a baseline.

When used carefully, a Decision Tree is both a strong model and a useful diagnostic tool.