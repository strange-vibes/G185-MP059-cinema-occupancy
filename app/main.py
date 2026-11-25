from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

app = FastAPI(title="Cinema Occupancy API")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_price_rs: float
    days_since_release: int = Field(ge=0)
    star_rating: float
    show_hour: int
    screen_seats: int = Field(le=700)
    distance_to_metro_km: float
    language: str
    location: str


@app.get("/health")
def health():
    return {"status": "ok"}


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(request: PredictionRequest):
    features = pd.DataFrame([request.model_dump()])
    prediction = get_model().predict(features)[0]
    return {"prediction": float(prediction)}