from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import numpy as np
import warnings

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

# Other
from typing import List

# Warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize FastAPI
app = FastAPI()

# Functions

def calculate_rmse(predictions, actuals):
    if(len(predictions) != len(actuals)):
        raise Exception("The amount of predictions did not equal the amount of actuals")
    
    return (((predictions - actuals) ** 2).sum() / len(actuals)) ** (1/2)

def calculate_accuracy(predictions, actuals):
    if(len(predictions) != len(actuals)):
        raise Exception("The amount of predictions did not equal the amount of actuals")
    
    return (predictions == actuals).sum() / len(actuals)

# AI-model

# Dummy data
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5, 6],
    "feature2": [10, 9, 8, 1, 2, 3],
    "label":    [0, 0, 0, 1, 1, 1]  # Boven 5: afval
})

# Features en labels splitsen
X = df[["feature1", "feature2"]]
y = df["label"]

target_names = {
    0: "Geen afval",
    1: "Wel afval"
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# API endpoints

@app.get("/")
def root():
    return {"API Aan"}

class Features(BaseModel):
    features: List[float]   

@app.post("/predict/")
def predict_litter(item: Features):
    data = np.array(item.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    name = target_names[prediction]
    return {"prediction": name}