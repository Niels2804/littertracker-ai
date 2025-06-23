from datetime import datetime, timedelta
import random
from statistics import mean, stdev
from fastapi import FastAPI, Query
import pandas as pd
import warnings
import requests

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = FastAPI()

# Holiday Data
def fetch_holidays():
    url = "https://avansict2231011.azurewebsites.net/api/External/holidays"
    try:
        response = requests.get(url)
        response.raise_for_status()
        holidays = response.json()
        holiday_dates = set(h["date"][:10] for h in holidays)
        return holiday_dates
    except Exception as e:
        print(f"Error retrieving holidays: {e}")
        return set()

# Sensoring Data
def fetch_sensoring_data_with_holidays():
    holidays = fetch_holidays()
    url = "https://avansict2231011.azurewebsites.net/api/LitterFromSensoring/GetAllLitterFromSensoring"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        records = []
        for entry in data:
            temp = entry.get("weatherInfo", {}).get("temperatureCelsius")
            humidity = entry.get("weatherInfo", {}).get("humidity")
            classification = entry.get("classification")
            confidence = entry.get("confidence", 1.0)
            detection_time = entry.get("detectionTime", "")[:10] 

            amount = confidence * 10

            is_holiday = detection_time in holidays

            if temp is not None and humidity is not None and classification is not None:
                records.append({
                    "temperature": temp,
                    "humidity": humidity,
                    "classification": classification,
                    "amount": amount,
                    "is_holiday": int(is_holiday)  # 1 or 0 for holiday
                })

        df = pd.DataFrame(records)
        return df

    except Exception as e:
        print(f"Error retrieving sensoring data: {e}")
        return pd.DataFrame()

# Train AI-model
df = fetch_sensoring_data_with_holidays()

if df.empty:
    raise ValueError("No training data received!")

X = df[["temperature", "humidity", "classification", "is_holiday"]]
y = df["amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# API Endpoints

# Default endpoint to check if the API is working
@app.get("/")
def root():
    return {"message": "API is up and running!! ✅"}

# Endpoint to predict litter based on start and end date (../predict?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD)
@app.get("/predict/")
def predictLitter(
    startDate: datetime = Query(..., description="Begin date in format YYYY-MM-DD"),
    endDate: datetime = Query(..., description="End date in format YYYY-MM-DD")
):
    holidays = fetch_holidays()

    monthTemp = {
        1: 3.5, 2: 4.0, 3: 7.5, 4: 11.0, 5: 15.0, 6: 18.0,
        7: 20.5, 8: 20.0, 9: 16.5, 10: 12.0, 11: 7.0, 12: 4.5
    }
    
    monthHumidity = {
        1: 85, 2: 80, 3: 75, 4: 70, 5: 65, 6: 60,
        7: 60, 8: 65, 9: 70, 10: 75, 11: 80, 12: 85
    }

    if startDate > endDate:
        return {"error": "Start date must be before or equal to endDate."}

    days = (endDate - startDate).days + 1
    results = []

    for i in range(days):
        currentDate = startDate + timedelta(days=i)
        date_str = currentDate.strftime("%Y-%m-%d")
        month = currentDate.month

        baseTemp = monthTemp.get(month, 15)
        baseHumidity = monthHumidity.get(month, 70)
        is_holiday = 1 if date_str in holidays else 0

        confidence = 0
        while confidence <= 0.75:
            totalDetections = random.randint(3, 10)
            dailyPredictions = []

            for _ in range(totalDetections):
                temp = random.uniform(baseTemp - 3, baseTemp + 3)
                humidity = random.uniform(baseHumidity - 10, baseHumidity + 10)
                humidity = max(20, min(100, humidity))
                classification = random.randint(0, 7)

                inputData = pd.DataFrame([[temp, humidity, classification, is_holiday]],
                                         columns=["temperature", "humidity", "classification", "is_holiday"])
                prediction = model.predict(inputData)[0]
                dailyPredictions.append(prediction)

            predictedTotal = round(sum(dailyPredictions))
            meanPrediction = mean(dailyPredictions)
            stdDev = stdev(dailyPredictions) if len(dailyPredictions) > 1 else 0.0
            confidence = max(0.0, 1.0 - (stdDev / meanPrediction)) if meanPrediction > 0 else 0.0

        results.append({
            "date": date_str,
            "predictedTotal": predictedTotal,
            "confidence": round(confidence, 2)
        })

    return {"predictions": results}


