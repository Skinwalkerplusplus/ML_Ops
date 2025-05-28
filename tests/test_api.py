import requests
import cv2
from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import pytest

client = TestClient(app)

TEST_IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)
_, TEST_IMAGE_BYTES = cv2.imencode(".jpg", TEST_IMAGE)

def test_predict_returns_valid_image():
    """Test that /predict returns a valid JPEG image"""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", TEST_IMAGE_BYTES.tobytes())}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[0] > 0 and img.shape[1] > 0