# train_and_save_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

# Create a small dummy dataset for testing
data = pd.DataFrame({
    "n_bed": [2, 3, 4, 3, 5],
    "n_bath": [1, 2, 3, 2, 4],
    "lat": [47.6, 47.7, 47.8, 47.65, 47.62],
    "long": [-122.3, -122.2, -122.25, -122.31, -122.28],
    "sqft": [1200, 1800, 2400, 2000, 3000],
    "price": [300000, 450000, 520000, 480000, 700000]
})

X = data.drop("price", axis=1)
y = data["price"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=50, random_state=0))
])

pipe.fit(X, y)

# Save model + pipeline as preprocessor
joblib.dump(pipe, "models/preprocessor.joblib")
joblib.dump(pipe["model"], "models/house_model.joblib") # type: ignore

print("✅ Model and preprocessor saved in /models/")
