import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ✅ 1) Create synthetic regression dataset (no internet needed)
X, y = make_regression( # type: ignore
    n_samples=500, n_features=5,
    noise=10.0, random_state=42
)

# ✅ 2) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 3) Model + hyperparameters
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

rf = RandomForestRegressor(random_state=0)

# ✅ 4) Grid Search CV
gs = GridSearchCV(
    rf, param_grid, cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

gs.fit(X_train, y_train)

# ✅ 5) Evaluation
print("Best Parameters:", gs.best_params_)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)
print("Test Score (R²):", best_model.score(X_test, y_test))
