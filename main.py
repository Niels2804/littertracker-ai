from datetime import datetime, timedelta
import random
from statistics import mean, stdev
from fastapi import FastAPI, Query
import pandas as pd
import warnings

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = FastAPI()

# Generate realistic data for training the model
def generate_data(n=1500):
    monthTemp = {
        1: 3.5, 2: 4.0, 3: 7.5, 4: 11.0, 5: 15.0, 6: 18.0,
        7: 20.5, 8: 20.0, 9: 16.5, 10: 12.0, 11: 7.0, 12: 4.5
    }

    monthHumidity = {
        1: 85, 2: 80, 3: 75, 4: 70, 5: 65, 6: 60,
        7: 60, 8: 65, 9: 70, 10: 75, 11: 80, 12: 85
    }

    currentYear = datetime.now().year

    features = {
        "temperature": [],
        "humidity": [],
        "classification": [],
        "amount": []
    }

    for _ in range(n):
        randomDay = random.randint(1, 366)
        date = datetime(currentYear, 1, 1) + timedelta(days=randomDay - 1)
        month = date.month

        baseTemp = monthTemp.get(month, 10)
        temp = random.uniform(baseTemp - 3, baseTemp + 3)

        baseHumidity = monthHumidity.get(month, 70)
        humidity = random.uniform(baseHumidity - 10, baseHumidity + 10)
        humidity = max(20, min(100, humidity))

        classification = random.randint(0, 7)

        baseAmounts = {
            0: 1,  # battery
            1: 3,  # cardboard
            2: 2,  # glass
            3: 2,  # metal
            4: 5,  # organic
            5: 4,  # paper
            6: 8,  # plastic (much more common)
            7: 3   # tissue
        }

        tempFactors = {
            0: 0.01,  # battery (low effect)
            1: 0.02,  # cardboard
            2: 0.00,  # glass
            3: 0.00,  # metal
            4: 0.06,  # organic (high effect)
            5: 0.03,  # paper
            6: 0.05,  # plastic
            7: 0.04   # tissue
        }

        tempFactor = 1.0 + tempFactors[classification] * (temp - 20)

        # Humidity factor limiting between 0.5 and 1.2
        humidityFactor = 1.0 - 0.01 * (humidity - 50)
        humidityFactor = max(0.5, min(1.2, humidityFactor))

        amount = baseAmounts[classification] * tempFactor * humidityFactor
        noise = random.uniform(0.8, 1.2)

        amount = max(0, round(amount * noise, 2))

        features["temperature"].append(round(temp, 2))
        features["humidity"].append(round(humidity, 2))
        features["classification"].append(classification)
        features["amount"].append(amount)

    return pd.DataFrame(features)

# Machine Learning by using Random Forest Regressor

# Generates 10.000 Dummy training data for model
df = generate_data(10000) 

X = df[["temperature", "humidity", "classification"]]
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
    if startDate > endDate:
        return {"error": "Start date must be before or equal to end date"}

    monthTemp = {
        1: 3.5, 2: 4.0, 3: 7.5, 4: 11.0, 5: 15.0, 6: 18.0,
        7: 20.5, 8: 20.0, 9: 16.5, 10: 12.0, 11: 7.0, 12: 4.5
    }
    
    monthHumidity = {
        1: 85, 2: 80, 3: 75, 4: 70, 5: 65, 6: 60,
        7: 60, 8: 65, 9: 70, 10: 75, 11: 80, 12: 85
    }

    results = []

    days = (endDate - startDate).days + 1

    for i in range(days):
        currentDate = startDate + timedelta(days=i)
        month = currentDate.month

        baseTemp = monthTemp.get(month, 15)
        baseHumidity = monthHumidity.get(month, 70)

        confidence = 0
        while confidence <= 0.75:
            totalDetections = random.randint(3, 10)
            dailyPredictions = []

            for _ in range(totalDetections):
                temp = random.uniform(baseTemp - 3, baseTemp + 3)
                humidity = random.uniform(baseHumidity - 10, baseHumidity + 10)
                humidity = max(20, min(100, humidity))

                classification = random.randint(0, 7)
                inputData = pd.DataFrame([[temp, humidity, classification]], columns=["temperature", "humidity", "classification"])
                prediction = model.predict(inputData)[0]
                dailyPredictions.append(prediction)

            predictedTotal = round(sum(dailyPredictions))
            meanPrediction = mean(dailyPredictions)
            stdDev = stdev(dailyPredictions) if len(dailyPredictions) > 1 else 0.0
            confidence = max(0.0, 1.0 - (stdDev / meanPrediction)) if meanPrediction > 0 else 0.0

        # Pas als confidence > 0.5, voeg toe aan resultaten
        results.append({
            "date": currentDate.strftime("%Y-%m-%d"),
            "predictedTotal": predictedTotal,
            "confidence": round(confidence, 2)
        })


    return {"predictions": results}

