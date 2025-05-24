from fastapi import FastAPI
from model.preprocessing import preprocess
from ultralytics import YOLO
import torch

app = FastAPI()
model = YOLO("model/Food_seg_model.pt")

@app.post("/predict")
def predict(input_data: dict):
    processed_data = preprocess(input_data)
    prediction = model.predict([processed_data])
    with torch.no_grad():
        output = model(processed_data)

    return output.item()