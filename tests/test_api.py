from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"feature": 5.1})
    assert response.status_code == 200