from django.shortcuts import render
import joblib
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

model = joblib.load("model1.pkl")

def home(request):
    return render(request, "index.html")

@csrf_exempt
def predict_house_price(request):
    try:
        data = json.loads(request.body)
        features = np.array([[
            float(data["bedrooms"]),
            float(data["bathrooms"]),
            float(data["sqft"]),
            float(data["lat"]),
            float(data["long"])
        ]])
        price = float(model.predict(features)[0])
        return JsonResponse({"price": price})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
