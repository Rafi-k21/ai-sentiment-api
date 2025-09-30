from fastapi import FastAPI
import joblib

# Load trained model
model = joblib.load("model/model.joblib")

# Initialize FastAPI app
app = FastAPI(title="AI Sentiment API", version="1.0")

@app.get("/")
def root():
    return {"message": "AI Sentiment API is running 🚀"}

@app.post("/predict")
def predict(text: str):
    """
    Predict sentiment (positive/negative) from input text.
    """
    prediction = model.predict([text])[0]
    return {"text": text, "sentiment": prediction}
