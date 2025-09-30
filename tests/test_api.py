from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Sentiment API is running 🚀"}

def test_predict_positive():
    response = client.post("/predict", params={"text": "I love this"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]

def test_predict_negative():
    response = client.post("/predict", params={"text": "I hate this"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
