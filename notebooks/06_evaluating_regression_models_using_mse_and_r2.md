# Evaluating Regression Models Using MSE and R²

After training a regression model, two questions matter most:

- How large are the prediction errors?
- How much of the target variation does the model explain?

Mean Squared Error (MSE) and R² answer those questions from different angles. Used together, they show both absolute error magnitude and relative explanatory power.

## 1. What MSE Measures

MSE is the average of squared differences between actual and predicted values:

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Where:

- `y_i` is the true value
- `y_hat_i` is the predicted value
- `n` is the number of samples

Because errors are squared, large mistakes count much more than small ones.

## 2. Why Squaring Matters

Squaring does two useful things:

- It makes all errors positive, so over-predictions and under-predictions do not cancel out
- It penalizes large errors disproportionately, which is useful when big misses are especially costly

For example, an error of 10 contributes 100 to MSE, while an error of 2 contributes only 4.

## 3. Units of MSE

MSE is expressed in squared target units.

- If the target is in lakhs, MSE is in lakhs²
- If the target is in hours, MSE is in hours²
- If the target is in °C, MSE is in °C²

That makes MSE less intuitive to report directly. RMSE is often easier to communicate because it restores the original units.

## 4. What R² Means

R², the coefficient of determination, measures how much variance in the target your model explains relative to the mean baseline.

```text
R² = 1 - (SS_res / SS_tot)
```

Where:

- `SS_res` is the residual sum of squares from your model
- `SS_tot` is the total sum of squares from always predicting the mean

Interpretation:

- `R² = 1.0`: perfect predictions
- `R² = 0.0`: no better than the mean baseline
- `R² < 0`: worse than the mean baseline

## 5. MSE vs R²

These metrics measure different things:

- MSE is an absolute error measure in squared units
- R² is a relative measure of explained variance

They can tell different stories:

- Low MSE and low R²: the target itself has low variance, so the baseline is already strong
- High MSE and high R²: the target has wide spread, so even large absolute errors may still beat the baseline by a lot

That is why both metrics matter.

## 6. Compute MSE and R² in scikit-learn

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")
```

## 7. Compare Against a Baseline

Always compare your model against a mean baseline.

```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
model_preds = model.predict(X_test)

def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:25s} | MSE: {mse:8.2f} | RMSE: {rmse:6.2f} | R²: {r2:.3f}")

evaluate("Baseline (mean)", y_test, baseline_preds)
evaluate("Linear Regression", y_test, model_preds)
```

The baseline should have R² near 0. If your model does not beat it, the features may not contain enough signal or the model may be misspecified.

## 8. Interpreting MSE and R² Together

Use both metrics together:

- MSE tells you the size of the squared error
- R² tells you how much better you are than predicting the mean

Examples:

- Strong result: MSE drops sharply and R² rises well above 0
- Marginal result: MSE improves only slightly and R² stays near 0
- Bad result: R² is negative, which means the model is worse than the baseline

## 9. Cross-Validation

Cross-validation gives a more stable view of model quality than a single test split.

```python
from sklearn.model_selection import cross_val_score

cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
cv_mse = -cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
)
cv_rmse = np.sqrt(cv_mse)

print(f"CV R² scores: {cv_r2.round(3)}")
print(f"Mean CV R²:   {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print(f"CV RMSE scores: {cv_rmse.round(3)}")
print(f"Mean CV RMSE:   {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
```

Look for a low standard deviation across folds. Large variation suggests instability or a data split problem.

## 10. When to Use MSE

Use MSE when:

- large errors are especially harmful
- you want to punish outliers more strongly
- your training objective is based on squared loss
- you need a mathematically convenient metric for optimization

## 11. When to Prefer R²

Use R² when:

- you want to know how much variance your model explains
- you want a scale-free measure for comparing fit quality
- you need a quick check against the mean baseline

## 12. Common Mistakes

- Reporting MSE without the baseline
- Comparing MSE values across different target scales
- Treating R² as an accuracy percentage
- Ignoring negative R² values
- Evaluating on the training set instead of held-out data

## 13. Practical Checklist

Before reporting results, confirm that:

- MSE and R² were computed on test data
- both were compared to the mean baseline
- RMSE was reported for interpretability
- cross-validation supports the conclusion
- residuals do not show obvious patterns

## 14. Final Takeaway

MSE tells you how large your squared errors are.

R² tells you how much better your model is than predicting the mean.

Together they give a more complete picture than either metric alone.