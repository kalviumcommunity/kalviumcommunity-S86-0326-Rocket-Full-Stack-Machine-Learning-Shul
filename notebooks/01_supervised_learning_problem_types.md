# Understanding Supervised Learning Problem Types

Before training your first model, identify the exact supervised learning problem type. This decision controls:

- Which algorithms are appropriate
- Which evaluation metrics are meaningful
- How to define business success

If this step is wrong, everything after it is misleading.

## 1. What Is Supervised Learning?

Supervised learning uses labeled data:

- Inputs: feature matrix `X`
- Known outputs: target `y`
- Goal: learn a mapping from `X` to `y` that generalizes to unseen data

Conceptual flow:

`X + y -> training -> model -> predictions`

Differences from other paradigms:

- Supervised learning: labels available
- Unsupervised learning: no labels
- Reinforcement learning: reward-based trial and error

## 2. The Two Core Types

### Classification

Predict a discrete category.

Examples:

- Spam vs not spam
- Customer churn: yes or no
- Iris species: setosa/versicolor/virginica

Typical outputs:

- Predicted class label
- Class probabilities

### Regression

Predict a continuous numeric value.

Examples:

- House price
- Monthly revenue
- Delivery time

Typical outputs:

- Numeric prediction
- Optional confidence/prediction interval

## 3. Classification Subtypes

### Binary Classification

- Exactly two classes (for example, churn vs no churn)
- Usually define positive class explicitly
- Common challenge: class imbalance

### Multi-Class Classification

- Three or more mutually exclusive classes
- One sample belongs to exactly one class

### Multi-Label Classification

- Multiple labels can be true at once
- One sample can belong to many classes

## 4. Regression Subtypes

### Simple Linear Regression

- One feature predicts one continuous output
- Model form: `y = mx + b`

### Multiple Linear Regression

- Many features predict one continuous output
- Linear combination of features

### Non-Linear Regression

- Captures curved/complex relationships
- Examples: polynomial models, tree ensembles, neural networks

### Count Regression

- Predict non-negative integer counts (`0, 1, 2, ...`)
- Often modeled with Poisson or Negative Binomial methods

## 5. How to Identify Problem Type from Business Requirements

Use this checklist before coding:

1. What am I predicting?
   - Category -> classification
   - Number -> regression
2. How many outcomes exist?
   - 2 classes -> binary
   - many exclusive classes -> multi-class
   - many non-exclusive labels -> multi-label
3. What does the target column look like?
   - category labels -> classification
   - continuous numeric values -> regression
4. What does success look like to the business?
   - catch positives -> prioritize recall
   - reduce false alarms -> prioritize precision
   - minimize numeric error -> use MAE/RMSE

## 6. Common Pitfalls

1. Binning continuous targets into classes without business justification
2. Treating class labels as numeric regression targets
3. Reporting only accuracy on imbalanced data
4. Confusing multi-class and multi-label problems

## 7. Metrics by Problem Type

### Classification

- Accuracy (use carefully)
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

### Regression

- MAE
- MSE
- RMSE
- R2
- MAPE (with caution for small true values)

## 8. Algorithm Families by Type

### Classification

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost/LightGBM/CatBoost)
- SVM
- Naive Bayes

### Regression

- Linear Regression
- Ridge/Lasso
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Neural Networks

## 9. Quick Mapping Examples

- Predict if a customer will churn -> Binary classification
- Predict which product category a user buys -> Multi-class classification
- Predict which tags apply to an article -> Multi-label classification
- Predict next month revenue -> Regression
- Predict number of purchases next month -> Count regression

## 10. Practical Exercise (Solved)

### A) Predict credit risk score (300-850)

- Type: Regression
- Target: numeric score
- Success: low MAE/RMSE

### B) Predict 30-day readmission

- Type: Binary classification
- Target: readmitted yes/no
- Success: high recall with acceptable precision

### C) Predict number of items purchased

- Type: Count regression
- Target: non-negative purchase count
- Success: low count error (for example MAE)

### D) Predict article topics where multiple topics apply

- Type: Multi-label classification
- Target: set of topic labels
- Success: strong micro/macro F1, low Hamming loss

## 11. Project Context: What This Means for This Repository

This project uses Telco churn data, so the primary supervised task is:

- Binary classification
- Target: `Churn` (`Yes`/`No`)
- Positive class (recommended): `Yes` (customer churn)

For this use case:

- Accuracy alone is not enough
- Monitor precision/recall/F1/ROC-AUC
- Tune decision threshold based on retention team capacity and business cost

## 12. Final Reminder

Before training any model, always answer:

1. Category or number?
2. If category: binary, multi-class, or multi-label?
3. If number: continuous or count?
4. What metric matches business success?

Define the problem first. Then build the model.
