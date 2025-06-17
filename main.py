from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import numpy as np
import warnings
from datetime import datetime, timedelta
import random 

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# Other
from typing import List

# Warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize FastAPI
app = FastAPI()

# Input model
class DateRange(BaseModel):
    start_date: datetime
    end_date: datetime

# Dummy data
df = pd.DataFrame({
    "temperature": [20, 22, 25, 18, 17, 21],
    "humidity": [50, 60, 55, 40, 45, 65],
    "classification": [0, 1, 2, 3, 4, 5],
    "amount": [10, 15, 12, 8, 6, 14]
})

X = df[["temperature", "humidity", "classification"]]
y = df["amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Mapping classificaties
class_map = {
    0: "paper",
    1: "plastic",
    2: "biodegradable",
    3: "cardboard",
    4: "glass",
    5: "metal"
}

@app.get("/")
def root():
    return {"message": "API Aan"}

@app.post("/predict/")
def predict_litter(date_range: DateRange):
    start = date_range.start_date
    end = date_range.end_date

    if start > end:
        return {"error": "Startdatum moet vóór of gelijk zijn aan einddatum"}

    delta = end - start
    results = []

    for i in range(delta.days + 1):
        date = start + timedelta(days=i)

        # Simuleer features
        temp = random.uniform(15, 30)
        humidity = random.uniform(30, 70)
        classification = random.randint(0, 5)

        input_data = np.array([[temp, humidity, classification]])
        prediction = model.predict(input_data)[0]

        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "classification": class_map[classification],
            "predicted_amount": round(prediction, 2)
        })

    return {"predictions": results}
