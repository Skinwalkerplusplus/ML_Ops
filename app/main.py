from fastapi import FastAPI
from model.preprocessing import preprocess
import joblib

app = FastAPI()
model = joblib.load("model/model.pkl")  # Load your trained model

@app.post("/predict")
def predict(input_data: dict):
    processed_data = preprocess(input_data)  # Use your preprocessing
    prediction = model.predict([processed_data])
    return {"prediction": prediction.tolist()}