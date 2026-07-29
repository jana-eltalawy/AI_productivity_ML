"""
main.py
-------
FastAPI service for the AI-adoption productivity-gain predictor.

Expects an `artifacts/` folder (created by `train_model.py`) next to this
file, containing:
    - model.joblib
    - scaler.joblib
    - feature_columns.joblib

Run locally with:
    uvicorn main:app --reload --port 8000

Docs available at:  http://127.0.0.1:8000/docs
"""

import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

CAT_FEATURES = [
    "Industry",
    "Job_Role",
    "Location",
    "Primary_AI_Tool",
    "Experience_Level",
]
EXPERIENCE_BINS = [0, 5, 10, 20, 25]
EXPERIENCE_LABELS = ["Junior", "Mid", "Senior", "Expert"]

app = FastAPI(
    title="AI Adoption Productivity Predictor",
    description="Predicts Productivity_Gain_Percent from user AI-adoption data.",
    version="1.0.0",
)

# --- load artifacts once at startup -------------------------------------
model = None
scaler = None
feature_columns: List[str] = []


@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_columns
    try:
        model = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
        scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
        feature_columns = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))
    except FileNotFoundError as e:
        # Service will still start, but /predict will fail until artifacts exist.
        print(f"[WARNING] Could not load artifacts: {e}")


# --- request / response schemas ------------------------------------------
class PredictionRequest(BaseModel):
    Industry: str = Field(..., example="Technology")
    Job_Role: str = Field(..., example="Software Engineer")
    Location: str = Field(..., example="Remote")
    Primary_AI_Tool: str = Field(..., example="ChatGPT")
    Experience_Years: float = Field(..., ge=0, example=4)
    Daily_Token_Usage: float = Field(..., ge=0, example=15000)
    Tasks_Automated_Per_Week: float = Field(..., gt=0, example=12)


class PredictionResponse(BaseModel):
    predicted_productivity_gain_percent: float


# --- helpers ---------------------------------------------------------------
def build_feature_row(payload: PredictionRequest) -> pd.DataFrame:
    row = {
        "Industry": payload.Industry,
        "Job_Role": payload.Job_Role,
        "Location": payload.Location,
        "Primary_AI_Tool": payload.Primary_AI_Tool,
        "Daily_Token_Usage": payload.Daily_Token_Usage,
        "Tasks_Automated_Per_Week": payload.Tasks_Automated_Per_Week,
    }
    df = pd.DataFrame([row])

    # Same feature engineering as training
    df["Experience_Level"] = pd.cut(
        [payload.Experience_Years],
        bins=EXPERIENCE_BINS,
        labels=EXPERIENCE_LABELS,
    )
    df["Tokens_per_Task"] = df["Daily_Token_Usage"] / df["Tasks_Automated_Per_Week"]

    df = pd.get_dummies(df, columns=CAT_FEATURES, drop_first=True)

    # Align to the exact columns/order seen during training; anything
    # missing (a category that wasn't the reference/dropped level) becomes 0.
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


# --- routes ------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "FastAPI is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artifacts not found in '{ARTIFACT_DIR}/'. "
                "Run train_model.py first and restart the API."
            ),
        )
    if payload.Tasks_Automated_Per_Week <= 0:
        raise HTTPException(status_code=400, detail="Tasks_Automated_Per_Week must be > 0")

    x = build_feature_row(payload)
    x_scaled = scaler.transform(x)
    prediction = float(model.predict(x_scaled)[0])

    return PredictionResponse(predicted_productivity_gain_percent=round(prediction, 2))
