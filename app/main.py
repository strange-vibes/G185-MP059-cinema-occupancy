import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

import joblib
import pandas as pd


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout dynamically on each write."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


logger = logging.getLogger("cinema_occupancy.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = _StdoutHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

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
    start_time = time.perf_counter()
    features = pd.DataFrame([request.model_dump()])
    prediction = float(get_model().predict(features)[0])
    duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
    logger.info(
        json.dumps({
            "event": "prediction",
            "prediction": prediction,
            "duration_ms": duration_ms,
        })
    )
    return {"prediction": prediction}
