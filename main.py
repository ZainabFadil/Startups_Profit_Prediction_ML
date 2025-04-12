
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

app = FastAPI()

class InputData(BaseModel):
    RnD_Spend: float
    Administration: float
    Marketing_Spend: float

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.RnD_Spend, data.Administration, data.Marketing_Spend]])
    prediction = model.predict(X)
    return {"predicted_profit": prediction[0]}
