import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


class StubModel:
    """Deterministic stub model for unit-testing the API without disk or MLflow state."""

    def __init__(self, value: float = 62.5):
        self.value = value

    def predict(self, features):
        return [self.value for _ in range(len(features))]


VALID_PAYLOAD = {
    "ticket_price_rs": 321.0,
    "days_since_release": 33,
    "star_rating": 3.1,
    "show_hour": 15,
    "screen_seats": 517,
    "distance_to_metro_km": 4.74,
    "language": "hindi",
    "location": "Mall",
}


@pytest.fixture(autouse=True)
def stub_model_boundary(monkeypatch, tmp_path):
    """Replace the model-loading boundary so tests never depend on raw CSVs or MLflow."""
    if hasattr(app_main.get_model, "cache_clear"):
        app_main.get_model.cache_clear()
    monkeypatch.setattr(app_main, "MODEL_PATH", tmp_path / "nonexistent.joblib")
    monkeypatch.setattr(app_main, "get_model", lambda: StubModel(62.5))


@pytest.fixture
def client():
    with TestClient(app_main.app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload_returns_numeric_prediction(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] == 62.5


def test_predict_rejects_screen_seats_over_limit(client):
    payload = {**VALID_PAYLOAD, "screen_seats": 701}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("screen_seats" in err.get("loc", []) for err in errors)


def test_predict_rejects_negative_days_since_release(client):
    payload = {**VALID_PAYLOAD, "days_since_release": -1}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("days_since_release" in err.get("loc", []) for err in errors)


def test_predict_logs_json_line_to_stdout(client, capsys, caplog):
    capsys.readouterr()

    with caplog.at_level(logging.INFO, logger=app_main.logger.name):
        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "prediction"
    assert isinstance(record["prediction"], (int, float))
    assert record["prediction"] == 62.5
    assert isinstance(record["duration_ms"], (int, float))
    assert record["duration_ms"] >= 0

    for field in VALID_PAYLOAD:
        assert field not in record
    assert "Mall" not in lines[0]
    assert "hindi" not in lines[0]

    assert any(
        r.name == app_main.logger.name and json.loads(r.getMessage()) == record
        for r in caplog.records
    )


def test_validation_error_does_not_log_prediction(client, capsys):
    capsys.readouterr()

    response = client.post("/predict", json={**VALID_PAYLOAD, "screen_seats": 701})
    assert response.status_code == 422

    captured = capsys.readouterr()
    assert captured.out.strip() == ""
