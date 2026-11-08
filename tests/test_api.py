from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, model_loader
from api.model_loader import FEATURE_COLUMNS

VALID_SOIL = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.88,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.94,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_metadata(client):
    health = client.get("/health")
    metadata = client.get("/metadata")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "model_loaded": True}
    assert metadata.json()["service"] == "Crop Recommendation API"
    assert metadata.json()["model_loaded"] is True
    assert len(metadata.json()["classes"]) == 22


def test_predict_returns_a_crop_and_probabilities(client):
    response = client.post("/predict", json=VALID_SOIL)
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_crop"] in client.get("/metadata").json()["classes"]
    assert 0 <= body["confidence"] <= 1
    assert len(body["probabilities"]) == 22
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


def test_portable_model_matches_training_model(client):
    root = Path(__file__).resolve().parents[1]
    training_model = joblib.load(root / "models" / "crop_random_forest.pkl")
    frame = pd.DataFrame([VALID_SOIL], columns=FEATURE_COLUMNS)
    expected = training_model.predict_proba(frame)[0]
    assert model_loader.model is not None
    actual = model_loader.model.predict_proba([frame.iloc[0].tolist()])[0]
    for actual_value, expected_value in zip(actual, expected):
        assert actual_value == pytest.approx(expected_value, abs=1e-12)


def test_batch_prediction(client):
    response = client.post("/predict/batch", json={"items": [VALID_SOIL, VALID_SOIL]})
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "update",
    [
        {"ph": 15},
        {"humidity": 101},
        {"N": -1},
        {"unknown": 2},
    ],
)
def test_invalid_input_is_rejected(client, update):
    response = client.post("/predict", json={**VALID_SOIL, **update})
    assert response.status_code == 422


def test_missing_feature_is_rejected(client):
    payload = VALID_SOIL.copy()
    del payload["rainfall"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422
