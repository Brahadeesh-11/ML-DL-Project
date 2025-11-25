# train_and_save.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import sys

CSV_PATH = "C:/Users/braha/OneDrive/Desktop/programs/ML and DL/day 3/kc_house_data.csv"  # <-- uploaded CSV path
MODEL_PATH = "model1.pkl"

def main():
    df = pd.read_csv(CSV_PATH)
    # required columns
    features = ["bedrooms", "bathrooms", "sqft_living", "lat", "long"]
    for c in features + ["price"]:
        if c not in df.columns:
            print(f"CSV missing column: {c}")
            sys.exit(1)

    X = df[features].astype(float)
    y = df["price"].astype(float)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # optional: print simple metric
    r2 = model.score(X_val, y_val)
    print(f"Validation R^2: {r2:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
