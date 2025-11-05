# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load saved artifacts
preproc = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/house_model.joblib")

class InputData(BaseModel):
    n_bed: int
    n_bath: int
    lat: float
    long: float
    sqft: float

@app.post("/predict")
def predict(item: InputData):
    data = [[item.n_bed, item.n_bath, item.lat, item.long, item.sqft]]

    # Scale using preprocessor
    X = preproc.named_steps["scaler"].transform(data)

    # Predict
    pred = model.predict(data)[0]

    # Static confidence band (replace with real residual stdev from training)
    resid_std = 30000

    return {
        "price": float(pred),
        "lower": float(pred - 1.96 * resid_std),
        "upper": float(pred + 1.96 * resid_std)
    }
