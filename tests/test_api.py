import requests
import cv2

# Load image
image = cv2.imread("test.jpg")
_, img_encoded = cv2.imencode(".jpg", image)

# Send to API
response = requests.post(
    "http://localhost/predict",
    files={"file": ("test.jpg", img_encoded.tobytes())}
)
print(response.json())