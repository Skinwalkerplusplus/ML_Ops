from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import cv2
from model.preprocessing import preprocess
from model.model_import import load_seg_model
#from model.model_import import load_reg_model
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

app = FastAPI()

model = load_seg_model()
#model_reg = load_reg_model()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>YOLO Model Server</h1>
    <p>Try <a href="/predict">/predict</a></p>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #processed_data = preprocess(input_data)

    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    except:
        print('File type not supported')

    #prediction = model.predict([processed_data])
    results = model(img)
    

    for result in results:

        annotated_frame = result.plot()

        pred_text, annotated_frame = run_regressor(result, img, annotated_frame)

    _, jpg_bytes = cv2.imencode(".jpg", annotated_frame)

    return Response(content=jpg_bytes.tobytes(), media_type="image/jpeg")

def run_regressor(result, frame, annotated_frame):
    try:
        # Comprobamos si hay mscaras
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
            predictions = []

            # Device del modelo para evitar error
            #device = next(model_reg.parameters()).device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            for i, mask in enumerate(result.masks.data):
                try:
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].cpu().numpy())

                    # Crop para modelo consecuente
                    cropped_region = frame[y1:y2, x1:x2]

                    cropped_resized = cv2.resize(cropped_region, (224, 224))

                    # BGR
                    cropped_rgb = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
                    cropped_tensor = torch.from_numpy(cropped_rgb).permute(2, 0, 1).float() / 255.0

                    # Normalizar
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
                    cropped_tensor = (cropped_tensor.to(device) - mean) / std

                    # Pasamos datos al modelo
                    #with torch.no_grad():
                        #prediction = model_reg(cropped_tensor.unsqueeze(0))

                    # Texto que aparecer
                    #pred_text = f"Estimate: {prediction.item():.2f}"
                    #predictions.append(pred_text)
                    pred_text = "hola"

                    # Aadimos al frame
                    if hasattr(result, 'boxes') and result.boxes is not None and i < len(result.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].cpu().numpy())
                        cv2.putText(annotated_frame, pred_text,
                                    (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

                except Exception as mask_error:
                    print(f"Error processing mask {i}: {str(mask_error)}")
                    continue

            return predictions, annotated_frame

        return None, annotated_frame

    except Exception as e:
        print(f"Error in run_regressor: {str(e)}")
        return None, annotated_frame

# hello