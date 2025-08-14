
# XGBoost Quick Reference (Regression & Classification)

## Overview

XGBoost is a fast, regularized gradient boosting library for tabular data. It supports regression, classification, and ranking tasks, with strong performance and flexible tuning.

---

## XGBRegressor vs XGBClassifier

| Feature         | XGBRegressor                        | XGBClassifier                       |
|-----------------|-------------------------------------|-------------------------------------|
| Task            | Regression (predict continuous)     | Classification (predict classes)    |
| Objective       | `"reg:squarederror"` (default)      | `"binary:logistic"`, `"multi:softmax"`, etc. |
| Metrics         | `"rmse"`, `"mae"`, `"r2"`           | `"logloss"`, `"error"`, `"auc"`, etc. |
| Target values   | Real numbers                        | Class labels (int or str)           |
| Example use     | Predict GPP, house prices, etc.     | Predict species, churn, etc.        |

---

## Core Parameters

- `n_estimators`: Max number of boosting rounds. Set high, use early stopping.
- `learning_rate`: Step size shrinkage. Lower = more trees, less overfit. (0.01–0.1 typical)
- `booster`: `"gbtree"` (default), `"dart"` (dropout), `"gblinear"` (linear).
- `max_depth`: Tree depth (4–10 typical).
- `min_child_weight`: Min sum Hessian in a leaf (1–10).
- `subsample`: Row sampling per tree (0.6–1.0).
- `colsample_bytree`: Feature sampling per tree (0.6–1.0).
- `reg_alpha`/`reg_lambda`: L1/L2 regularization.
- `gamma`: Min loss reduction to split.
- `tree_method`: `"hist"` (fast CPU), `"gpu_hist"` (GPU).
- `random_state`: For reproducibility.
- `n_jobs`: Parallel threads.

### DART-specific
- `rate_drop`, `skip_drop`, `one_drop`: Dropout regularization for trees.

---

## Best Practices for Regression (fit for GPP/data.csv)

1. **Data Prep**: Drop nulls, split into train/val/test.
2. **Start Simple**: Use `gbtree` with `"hist"` and early stopping.
3. **Tune**: Focus on `max_depth`, `min_child_weight`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
4. **Early Stopping**: Set `n_estimators` high (e.g., 10000), use `early_stopping_rounds` with a validation set.
5. **RandomizedSearchCV**: Use for hyperparameter search. Use `scipy.stats.uniform` for continuous params, lists for discrete.
6. **DART**: Try if overfitting persists. Only tune DART-specific params after core tree params.
7. **Final Model**: Retrain with best params and early stopping on full train+val, test on untouched test set.

---

## Hyperparameter Tuning Tips

- **Continuous params**: Use `scipy.stats.uniform` (e.g., `"learning_rate": scipy.stats.uniform(0.01, 0.2)`).
- **Discrete/integer params**: Use `np.arange`, `np.linspace`, or lists (e.g., `"max_depth": [4, 6, 8]`).
- **Reduce search time**: Limit parameter ranges, use fewer `n_iter`, and lower `n_estimators` during search.
- **Early stopping**: Only use in final `.fit()`, not during CV search.

---

## Model Saving

- `.bin`: Fastest, smallest, best for production (XGBoost only).
- `.json`: Human-readable, cross-language, good for debugging.
- `.txt`: Most readable, not lossless, for inspection/education.

---

## Regularization

- **L1 (`reg_alpha`)**: Drives weights to zero (feature selection).
- **L2 (`reg_lambda`)**: Smooths weights (reduces variance).
- **Tree structure**: Lower `max_depth`, higher `min_child_weight`, higher `gamma` = more regularization.
- **Stochasticity**: Lower `subsample`/`colsample_bytree` = more regularization.
- **DART**: Adds dropout to trees for extra regularization.

---

## Example: Regression Workflow (matches `main.ipynb`)

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Data prep
df = pd.read_csv("data.csv").dropna()
X = df.drop(columns=["GPP"])
y = df["GPP"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

# Simple model with early stopping
model = XGBRegressor(n_estimators=10000, early_stopping_rounds=50, tree_method="hist", random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Hyperparameter search (example)
param_dist = {
    "learning_rate": [0.03, 0.05, 0.07],
    "max_depth": [6, 8, 10],
    "min_child_weight": [2, 3, 4],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [5, 10, 15],
    "reg_alpha": [0, 0.05],
    "gamma": [0, 0.05],
}
search = RandomizedSearchCV(
    XGBRegressor(n_estimators=1000, tree_method="hist", random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

# Final model with early stopping
best = XGBRegressor(**search.best_params_, n_estimators=10000, tree_method="hist", random_state=42)
best.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
```

---

## Quick Reference Table

| Parameter Type         | Grid Type                | Example                                      |
|------------------------|--------------------------|----------------------------------------------|
| Continuous             | `scipy.stats.uniform`    | `"learning_rate": scipy.stats.uniform(0.01, 0.2)` |
| Integer/Discrete       | `np.arange`/list         | `"max_depth": [4, 6, 8]`                     |

---

## Notes

- For GPP regression, use `reg:squarederror` and `rmse`/`r2` metrics.
- Never use the test set for early stopping or hyperparameter search.
- Use DART only if gbtree overfits.
- Save models in `.bin` for deployment, `.json` for sharing/debugging.

---
```# XGBoost Quick Reference (Regression & Classification)

## Overview

XGBoost is a fast, regularized gradient boosting library for tabular data. It supports regression, classification, and ranking tasks, with strong performance and flexible tuning.

---

## XGBRegressor vs XGBClassifier

| Feature         | XGBRegressor                        | XGBClassifier                       |
|-----------------|-------------------------------------|-------------------------------------|
| Task            | Regression (predict continuous)     | Classification (predict classes)    |
| Objective       | `"reg:squarederror"` (default)      | `"binary:logistic"`, `"multi:softmax"`, etc. |
| Metrics         | `"rmse"`, `"mae"`, `"r2"`           | `"logloss"`, `"error"`, `"auc"`, etc. |
| Target values   | Real numbers                        | Class labels (int or str)           |
| Example use     | Predict GPP, house prices, etc.     | Predict species, churn, etc.        |

---

## Core Parameters

- `n_estimators`: Max number of boosting rounds. Set high, use early stopping.
- `learning_rate`: Step size shrinkage. Lower = more trees, less overfit. (0.01–0.1 typical)
- `booster`: `"gbtree"` (default), `"dart"` (dropout), `"gblinear"` (linear).
- `max_depth`: Tree depth (4–10 typical).
- `min_child_weight`: Min sum Hessian in a leaf (1–10).
- `subsample`: Row sampling per tree (0.6–1.0).
- `colsample_bytree`: Feature sampling per tree (0.6–1.0).
- `reg_alpha`/`reg_lambda`: L1/L2 regularization.
- `gamma`: Min loss reduction to split.
- `tree_method`: `"hist"` (fast CPU), `"gpu_hist"` (GPU).
- `random_state`: For reproducibility.
- `n_jobs`: Parallel threads.

### DART-specific
- `rate_drop`, `skip_drop`, `one_drop`: Dropout regularization for trees.

---

## Best Practices for Regression (fit for GPP/data.csv)

1. **Data Prep**: Drop nulls, split into train/val/test.
2. **Start Simple**: Use `gbtree` with `"hist"` and early stopping.
3. **Tune**: Focus on `max_depth`, `min_child_weight`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
4. **Early Stopping**: Set `n_estimators` high (e.g., 10000), use `early_stopping_rounds` with a validation set.
5. **RandomizedSearchCV**: Use for hyperparameter search. Use `scipy.stats.uniform` for continuous params, lists for discrete.
6. **DART**: Try if overfitting persists. Only tune DART-specific params after core tree params.
7. **Final Model**: Retrain with best params and early stopping on full train+val, test on untouched test set.

---

## Hyperparameter Tuning Tips

- **Continuous params**: Use `scipy.stats.uniform` (e.g., `"learning_rate": scipy.stats.uniform(0.01, 0.2)`).
- **Discrete/integer params**: Use `np.arange`, `np.linspace`, or lists (e.g., `"max_depth": [4, 6, 8]`).
- **Reduce search time**: Limit parameter ranges, use fewer `n_iter`, and lower `n_estimators` during search.
- **Early stopping**: Only use in final `.fit()`, not during CV search.

---

## Model Saving

- `.bin`: Fastest, smallest, best for production (XGBoost only).
- `.json`: Human-readable, cross-language, good for debugging.
- `.txt`: Most readable, not lossless, for inspection/education.

---

## Regularization

- **L1 (`reg_alpha`)**: Drives weights to zero (feature selection).
- **L2 (`reg_lambda`)**: Smooths weights (reduces variance).
- **Tree structure**: Lower `max_depth`, higher `min_child_weight`, higher `gamma` = more regularization.
- **Stochasticity**: Lower `subsample`/`colsample_bytree` = more regularization.
- **DART**: Adds dropout to trees for extra regularization.

---

## Example: Regression Workflow (matches `main.ipynb`)

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Data prep
df = pd.read_csv("data.csv").dropna()
X = df.drop(columns=["GPP"])
y = df["GPP"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

# Simple model with early stopping
model = XGBRegressor(n_estimators=10000, early_stopping_rounds=50, tree_method="hist", random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Hyperparameter search (example)
param_dist = {
    "learning_rate": [0.03, 0.05, 0.07],
    "max_depth": [6, 8, 10],
    "min_child_weight": [2, 3, 4],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [5, 10, 15],
    "reg_alpha": [0, 0.05],
    "gamma": [0, 0.05],
}
search = RandomizedSearchCV(
    XGBRegressor(n_estimators=1000, tree_method="hist", random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

# Final model with early stopping
best = XGBRegressor(**search.best_params_, n_estimators=10000, tree_method="hist", random_state=42)
best.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
```

---

## Quick Reference Table

| Parameter Type         | Grid Type                | Example                                      |
|------------------------|--------------------------|----------------------------------------------|
| Continuous             | `scipy.stats.uniform`    | `"learning_rate": scipy.stats.uniform(0.01, 0.2)` |
| Integer/Discrete       | `np.arange`/`np.nplist`         | `"max_depth": [4, 6, 8]`                     |

---

## Notes

- For GPP regression, use `reg:squarederror` and `rmse`/`r2` metrics.
- Never use the test set for early stopping or hyperparameter search.
- Use DART only if gbtree overfits.
- Save models in `.bin` for deployment,