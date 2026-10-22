
import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.main import app


class FakeProphetModel:
    """Deterministic fitted-model substitute for API tests."""

    def __init__(self):
        self.history = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    [
                        "2024-05-24 22:30:00",
                        "2024-05-24 23:00:00",
                        "2024-05-24 23:30:00",
                    ]
                ),
                "y": [0.35, 0.40, 0.45],
            }
        )

    def predict(self, future):
        count = len(future)

        return pd.DataFrame(
            {
                "ds": future["ds"].to_numpy(),
                "yhat": np.full(count, 0.65),
                "yhat_lower": np.full(count, 0.50),
                "yhat_upper": np.full(count, 0.80),
            }
        )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "CycleCast Forecasting API"
    assert response.json()["model_family"] == "Prophet"


def test_health_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.discover_station_ids",
        lambda: ["BikePoints_1", "BikePoints_2"],
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["available_station_models"] == 2


def test_station_listing(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.discover_station_ids",
        lambda: ["BikePoints_1", "BikePoints_2", "BikePoints_3"],
    )

    response = client.get("/stations?limit=2&offset=1")

    assert response.status_code == 200
    assert response.json()["total"] == 3
    assert response.json()["station_ids"] == [
        "BikePoints_2",
        "BikePoints_3",
    ]


def test_forecast_returns_96_points(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.load_model",
        lambda station_id: FakeProphetModel(),
    )

    response = client.get(
        "/forecast/BikePoints_1?horizon_hours=24"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["station_id"] == "BikePoints_1"
    assert body["model_family"] == "Prophet"
    assert body["target"] == "occupancy_ratio"
    assert body["points_returned"] == 96
    assert len(body["forecast"]) == 96


def test_forecast_timestamps_are_15_minutes_apart(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.load_model",
        lambda station_id: FakeProphetModel(),
    )

    response = client.get(
        "/forecast/BikePoints_1?horizon_hours=1"
    )

    assert response.status_code == 200

    points = response.json()["forecast"]

    assert len(points) == 4

    timestamps = pd.to_datetime(
        [point["timestamp_utc"] for point in points],
        utc=True,
    )

    differences = timestamps.to_series().diff().dropna()

    assert all(
        difference == pd.Timedelta(minutes=15)
        for difference in differences
    )


def test_forecast_bounds_are_valid(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.load_model",
        lambda station_id: FakeProphetModel(),
    )

    response = client.get(
        "/forecast/BikePoints_1?horizon_hours=1"
    )

    assert response.status_code == 200

    for point in response.json()["forecast"]:
        assert 0.0 <= point["occupancy_ratio"] <= 1.0
        assert 0.0 <= point["lower_bound"] <= 1.0
        assert 0.0 <= point["upper_bound"] <= 1.0
        assert point["lower_bound"] <= point["occupancy_ratio"]
        assert point["occupancy_ratio"] <= point["upper_bound"]


def test_invalid_horizon_returns_422(client):
    response = client.get(
        "/forecast/BikePoints_1?horizon_hours=200"
    )

    assert response.status_code == 422


def test_missing_model_returns_404(client, monkeypatch):
    def missing_model(station_id):
        raise HTTPException(
            status_code=404,
            detail="No trained model exists.",
        )

    monkeypatch.setattr("app.main.load_model", missing_model)

    response = client.get(
        "/forecast/BikePoints_999999?horizon_hours=24"
    )

    assert response.status_code == 404


def test_invalid_station_id_returns_422(client):
    response = client.get(
        "/forecast/invalid%20station?horizon_hours=24"
    )

    assert response.status_code == 422
