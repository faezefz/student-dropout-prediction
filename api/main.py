import joblib
from fastapi import FastAPI

MODEL_PATH = "models/model.joblib"

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predicts the risk of a student dropping out.",
    version="0.1.0",
)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]


@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}