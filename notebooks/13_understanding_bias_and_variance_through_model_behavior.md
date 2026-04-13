# Understanding Bias and Variance Through Model Behavior

As you train more supervised learning models, you start noticing recurring patterns in their behavior:

- some models perform poorly on both training and test data
- some perform extremely well on training data but collapse on test data
- some strike a balance between the two

These are not random outcomes. They are the visible signs of the bias-variance trade-off.

This lesson focuses on recognizing those signs, diagnosing the failure mode correctly, and choosing the right fix.

## 1. What Bias and Variance Mean

### Bias

Bias is the error introduced by making overly simple assumptions about the relationship between features and target.

A high-bias model is too rigid. It assumes a shape for the data that is simpler than the real relationship, so it underfits.

Typical signs of high bias:

- high training error
- high test error
- small train-test gap
- structured residuals rather than random noise
- learning curves that flatten at a high error level

Example: if house prices rise nonlinearly with size, a linear model will systematically miss the curvature, even with lots of data.

### Variance

Variance is the error introduced by a model being too sensitive to the particular training sample it saw.

A high-variance model is too flexible. It learns the signal and the noise, so it fits the training set very closely but generalizes poorly.

Typical signs of high variance:

- very low training error
- much worse test error
- large train-test gap
- unstable cross-validation scores
- learning curves with a persistent gap between train and validation

Example: KNN with a very small `K` can memorize local quirks and outliers, creating a jagged decision boundary.

## 2. Underfitting and Overfitting

Bias and variance are easiest to understand through model behavior:

- underfitting usually means high bias
- overfitting usually means high variance

Underfitting means the model has not learned enough structure from the data. Overfitting means the model has learned too much of the training noise.

The practical question is not which label to use. The practical question is which fix is appropriate.

## 3. The Bias-Variance Trade-Off

As model complexity increases:

- bias tends to decrease
- variance tends to increase

As model complexity decreases:

- bias tends to increase
- variance tends to decrease

That is why the test error often follows a U-shaped curve as complexity changes.

The goal is not to eliminate both bias and variance. The goal is to find the point where generalization is best.

## 4. Decomposition of Error

Total prediction error can be thought of as having three parts:

- bias squared
- variance
- irreducible noise

Irreducible noise is the floor set by randomness in the data-generating process. No model can remove it.

That means when you reduce bias, variance often rises. When you reduce variance, bias often rises. You are balancing a fixed budget of reducible error.

## 5. How Algorithm Choice Changes the Trade-Off

Different algorithms shift the balance in different ways.

### Linear Regression

- linear fit on nonlinear data: high bias
- polynomial features: lower bias, more flexibility
- very high-degree polynomials: variance can explode

### Logistic Regression

- strong regularization can create high bias
- weaker regularization increases flexibility and can raise variance

### KNN

- small `K`: low bias, high variance
- large `K`: higher bias, lower variance

### Decision Trees

- shallow tree: high bias
- deep tree: high variance

The lever changes, but the principle stays the same.

## 6. Diagnosing Bias and Variance with Train and Test Metrics

Comparing training and test performance is the quickest diagnostic step.

| Train Performance | Test Performance | Train-Test Gap | Diagnosis |
|---|---|---|---|
| Poor | Poor | Small | High bias |
| Excellent | Poor | Large | High variance |
| Good | Good | Small | Good fit |
| Poor | Very poor | Large | Likely data or pipeline problem |

Examples:

- train accuracy 62 percent, test accuracy 60 percent: likely high bias
- train accuracy 99 percent, test accuracy 74 percent: likely high variance

The gap is the key signal for variance. A small gap with low metrics is the key signal for bias.

## 7. Learning Curves

Learning curves are the clearest way to see bias and variance together. They plot training and validation performance as training set size increases.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, val_mean, marker="o", label="Validation Accuracy")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Reading a High-Bias Learning Curve

- training score starts low and rises only modestly
- validation score starts low and rises only modestly
- both curves converge at a low performance level

Implication: more data will not solve the problem. The model needs more capacity or better features.

### Reading a High-Variance Learning Curve

- training score stays very high
- validation score is much lower
- the gap remains large even as data increases

Implication: more data may help, but regularization or lower complexity is usually also needed.

## 8. Why Dataset Size Matters Differently

More data does not help every problem equally.

### If the model has high bias

More data usually does not fix the core issue. If the functional form is wrong, the model will still be wrong in the same way.

### If the model has high variance

More data often helps because it reduces the chance that the model memorizes the quirks of one sample.

This is why data collection is most valuable when you have already confirmed that variance is the problem.

## 9. How to Reduce High Bias

If both train and test performance are poor and the gap is small, try:

- increasing model complexity
- adding informative features
- adding interaction or polynomial features
- reducing regularization strength
- decreasing `K` for KNN
- increasing tree depth

The principle is simple: give the model more freedom to fit the signal.

## 10. How to Reduce High Variance

If training performance is excellent but test performance is much worse, try:

- collecting more training data
- increasing regularization
- reducing model complexity
- increasing `K` for KNN
- removing irrelevant features
- using ensembles such as random forests or gradient boosting
- stopping iterative training earlier when appropriate

The principle is equally simple: constrain the model or average out noise.

## 11. Cross-Validation as a Variance Check

Cross-validation helps you see whether a model is stable across different folds.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.round(3)}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")
```

Interpretation:

- low mean, low std: likely high bias
- high mean, low std: stable fit
- high mean, high std: likely high variance
- low mean, high std: likely both the model and the data are problematic

Always report both the mean and the standard deviation. The standard deviation is often the first clue that the model is unstable.

## 12. Common Misconceptions

- high training accuracy does not mean the model is good
- more data does not automatically help
- more complex models are not always better
- regularization is not always beneficial if it becomes too strong
- a small train-test gap does not guarantee a good model if both metrics are low

## 13. Practical Checklist

Before deciding how to improve a model, confirm:

- training and test performance are both measured
- the train-test gap has been inspected
- cross-validation mean and standard deviation are reported
- a learning curve has been plotted if the diagnosis is unclear
- the fix matches the diagnosis

If the problem is bias, increase capacity or improve features. If the problem is variance, reduce complexity, regularize, or collect more data.

## 14. Repository Alignment

This repository already exposes several bias-variance levers in code:

- `src/train.py` trains a regularized Logistic Regression model
- `src/feature_engineering.py` scales numeric features and encodes categoricals safely
- `main.py` performs cross-validation and threshold tuning
- `notebooks/12_training_knn_model.md` shows how changing `K` changes the bias-variance balance

That makes this lesson a conceptual bridge between the earlier model-building modules and the evaluation modules that follow.

## 15. Final Takeaway

Bias and variance explain why models fail in different ways.

- high bias means the model is too simple
- high variance means the model is too sensitive
- the best model sits between those extremes

You do not remove the trade-off. You manage it.

The skill is not memorizing the definitions. The skill is reading the behavior, diagnosing the problem, and applying the right fix.