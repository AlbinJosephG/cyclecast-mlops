from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE_PREFIX = "prophet_station_"
MODEL_FILE_SUFFIX = ".joblib"

FORECAST_FREQUENCY_MINUTES = 15
MAX_FORECAST_HOURS = 168
OCCUPANCY_MIN = 0.0
OCCUPANCY_MAX = 1.0

STATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("cyclecast.api")

# Prophet prediction is protected against concurrent calls
# on a shared model instance.
prediction_lock = threading.Lock()


# ---------------------------------------------------------
# Response schemas
# ---------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_family: str
    available_station_models: int


class StationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    station_ids: list[str]


class ForecastPoint(BaseModel):
    timestamp_utc: str
    occupancy_ratio: float = Field(ge=0.0, le=1.0)
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)


class ForecastResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "station_id": "BikePoints_1",
                "model_family": "Prophet",
                "target": "occupancy_ratio",
                "frequency": "15min",
                "model_training_cutoff_utc": "2024-05-24T23:30:00Z",
                "forecast_start_utc": "2024-05-24T23:45:00Z",
                "forecast_end_utc": "2024-05-25T23:30:00Z",
                "horizon_hours": 24,
                "points_returned": 96,
                "forecast": [],
            }
        }
    )

    station_id: str
    model_family: Literal["Prophet"]
    target: Literal["occupancy_ratio"]
    frequency: Literal["15min"]
    model_training_cutoff_utc: str
    forecast_start_utc: str
    forecast_end_utc: str
    horizon_hours: int
    points_returned: int
    forecast: list[ForecastPoint]


# ---------------------------------------------------------
# Model discovery and loading
# ---------------------------------------------------------


def discover_station_ids() -> list[str]:
    """Return station IDs for model files that actually exist."""

    if not MODEL_DIR.is_dir():
        return []

    station_ids = []

    for model_path in MODEL_DIR.glob(f"{MODEL_FILE_PREFIX}*{MODEL_FILE_SUFFIX}"):
        station_id = model_path.name[len(MODEL_FILE_PREFIX) : -len(MODEL_FILE_SUFFIX)]

        if STATION_ID_PATTERN.fullmatch(station_id):
            station_ids.append(station_id)

    return sorted(station_ids)


def get_model_path(station_id: str) -> Path:
    """Resolve a validated station ID to its model artifact."""

    if not STATION_ID_PATTERN.fullmatch(station_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid station ID format.",
        )

    model_path = MODEL_DIR / f"{MODEL_FILE_PREFIX}{station_id}{MODEL_FILE_SUFFIX}"

    if not model_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No trained Prophet model exists for station '{station_id}'.",
        )

    return model_path


@lru_cache(maxsize=1)
def load_model(station_id: str):
    """
    Load and cache a previously trained Prophet model.

    No training or fitting is performed by this API.
    """

    model_path = get_model_path(station_id)

    logger.info(
        "Loading model for station=%s from %s",
        station_id,
        model_path.name,
    )

    try:
        model = joblib.load(model_path)

        if not hasattr(model, "predict") or not hasattr(model, "history"):
            raise TypeError("Artifact is not a fitted Prophet-compatible model.")

        if model.history is None or model.history.empty:
            raise ValueError("The saved model has no training history.")

        if "ds" not in model.history.columns:
            raise ValueError("Model history does not contain the 'ds' column.")

        return model

    except Exception as exc:
        logger.exception(
            "Failed to load model for station=%s",
            station_id,
        )
        raise RuntimeError(
            f"Could not load the saved model for station '{station_id}'."
        ) from exc


def format_utc(timestamp) -> str:
    """
    Format timezone-naive timestamps as UTC.

    Training data was converted to timezone-naive UTC before
    Prophet was fitted.
    """

    return pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    station_count = len(discover_station_ids())

    logger.info(
        "CycleCast API starting | station_models=%d | model_dir=%s",
        station_count,
        MODEL_DIR,
    )

    if station_count == 0:
        logger.warning("No station model files were found.")

    yield

    load_model.cache_clear()
    logger.info("CycleCast API stopped.")


app = FastAPI(
    title="CycleCast Forecasting API",
    description=(
        "REST API for 15-minute bicycle station occupancy forecasts "
        "using previously trained Prophet models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://cyclecast-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request logging and error handling
# ---------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled API error | method=%s | path=%s",
            request.method,
            request.url.path,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "HTTP request | method=%s | path=%s | status=%d | duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(HTTPException)
async def handle_http_exception(
    request: Request,
    exc: HTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------


@app.get("/", tags=["System"])
def root():
    return {
        "service": "CycleCast Forecasting API",
        "version": app.version,
        "model_family": "Prophet",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():
    return HealthResponse(
        status="ok",
        service="CycleCast Forecasting API",
        model_family="Prophet",
        available_station_models=len(discover_station_ids()),
    )


@app.get(
    "/stations",
    response_model=StationListResponse,
    tags=["Stations"],
)
def list_stations(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List stations for which a trained model artifact exists."""

    station_ids = discover_station_ids()

    return StationListResponse(
        total=len(station_ids),
        limit=limit,
        offset=offset,
        station_ids=station_ids[offset : offset + limit],
    )


@app.get(
    "/forecast/{station_id}",
    response_model=ForecastResponse,
    tags=["Forecasting"],
)
def forecast_station(
    station_id: str = PathParam(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Station identifier, such as BikePoints_1.",
    ),
    horizon_hours: int = Query(
        default=24,
        ge=1,
        le=MAX_FORECAST_HOURS,
        description="Forecast horizon in hours, from 1 to 168.",
    ),
):
    """
    Forecast occupancy for a station using its existing Prophet model.

    Forecasts begin 15 minutes after that model's last training
    timestamp. They do not represent live, current-day predictions.
    """

    try:
        model = load_model(station_id)

        last_training_timestamp = pd.Timestamp(model.history["ds"].max())

        if pd.isna(last_training_timestamp):
            raise ValueError("Model training cutoff could not be determined.")

        steps = horizon_hours * 60 // FORECAST_FREQUENCY_MINUTES

        forecast_dates = pd.date_range(
            start=last_training_timestamp
            + pd.Timedelta(minutes=FORECAST_FREQUENCY_MINUTES),
            periods=steps,
            freq=f"{FORECAST_FREQUENCY_MINUTES}min",
        )

        future = pd.DataFrame({"ds": forecast_dates})

        # Serialize prediction calls on shared model instances.
        with prediction_lock:
            predictions = model.predict(future)

        required_columns = {"ds", "yhat", "yhat_lower", "yhat_upper"}

        if not required_columns.issubset(predictions.columns):
            raise ValueError("Prophet returned an unexpected forecast schema.")

        predictions = predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        predictions = predictions.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

        if predictions.isna().any().any():
            raise ValueError("Forecast contains missing or non-finite values.")

        # Occupancy ratio must be within [0, 1].
        for column in ["yhat", "yhat_lower", "yhat_upper"]:
            predictions[column] = predictions[column].clip(
                OCCUPANCY_MIN,
                OCCUPANCY_MAX,
            )

        forecast_points = [
            ForecastPoint(
                timestamp_utc=format_utc(row.ds),
                occupancy_ratio=round(float(row.yhat), 6),
                lower_bound=round(float(row.yhat_lower), 6),
                upper_bound=round(float(row.yhat_upper), 6),
            )
            for row in predictions.itertuples(index=False)
        ]

        response = ForecastResponse(
            station_id=station_id,
            model_family="Prophet",
            target="occupancy_ratio",
            frequency="15min",
            model_training_cutoff_utc=format_utc(last_training_timestamp),
            forecast_start_utc=forecast_points[0].timestamp_utc,
            forecast_end_utc=forecast_points[-1].timestamp_utc,
            horizon_hours=horizon_hours,
            points_returned=len(forecast_points),
            forecast=forecast_points,
        )

        logger.info(
            "Forecast generated | station=%s | horizon_hours=%d | points=%d",
            station_id,
            horizon_hours,
            len(forecast_points),
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Forecast failed | station=%s | horizon_hours=%d",
            station_id,
            horizon_hours,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast generation failed. Check the API logs for diagnostic details."
            ),
        ) from exc
