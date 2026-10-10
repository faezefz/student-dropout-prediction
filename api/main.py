import joblib
from fastapi import FastAPI

from api.schemas import StudentData
import pandas as pd

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

@app.post("/predict")
def predict(student: StudentData):
    df = pd.DataFrame([student.model_dump()])
    proba = model.predict_proba(df)[0,1]

    return {
        "dropout_probability": round(float(proba), 3),
        "at_risk": bool(proba >= threshold),
    }