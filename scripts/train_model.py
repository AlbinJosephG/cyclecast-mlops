"""Train and evaluate Prophet models for cycle-station occupancy."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FREQUENCY = "15min"
HOLDOUT_DAYS = 7
MIN_TRAIN_DAYS = 7


def prepare_station_data(group: pd.DataFrame) -> pd.DataFrame:
    """Create a regular 15-minute occupancy time series."""
    group = group.sort_values("query_time").copy()
    group = group.drop_duplicates("query_time", keep="last")

    # Occupancy ratio is the fraction of total docks occupied by bikes.
    group["occupancy_ratio"] = np.where(
        group["docks"] > 0,
        group["bikes"] / group["docks"],
        np.nan,
    )

    group = group[group["occupancy_ratio"].between(0, 1, inclusive="both")]

    if group.empty:
        return pd.DataFrame(columns=["ds", "y"])

    series = (
        group.set_index("query_time")["occupancy_ratio"]
        .resample(FREQUENCY)
        .mean()
        .rename("y")
        .reset_index()
        .rename(columns={"query_time": "ds"})
    )

    # Prophet requires timezone-naive timestamps.
    # Remove timezone metadata after converting to UTC.
    series["ds"] = pd.to_datetime(series["ds"], utc=True).dt.tz_localize(None)

    # Keep missing intervals missing. Do not interpolate across the
    # training/holdout boundary or manufacture evaluation targets.
    return series.dropna(subset=["ds", "y"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/observations_2024-05-01_2024-05-03.parquet",
        help="Validated observations Parquet file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw = pd.read_parquet(input_path)
    required = {"query_time", "place_id", "bikes", "docks"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    raw["query_time"] = pd.to_datetime(raw["query_time"], utc=True, errors="coerce")
    for column in ["bikes", "docks"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna(subset=["query_time", "place_id", "bikes", "docks"])
    raw = raw[(raw["bikes"] >= 0) & (raw["docks"] > 0)]

    if raw.empty:
        raise ValueError("No usable station observations remain.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    station_reports = []
    start_time = time.perf_counter()

    for station_id, group in raw.groupby("place_id", sort=True):
        series = prepare_station_data(group)
        if series.empty:
            continue

        cutoff = series["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        training = series[series["ds"] < cutoff].copy()
        holdout = series[series["ds"] >= cutoff].copy()

        if training.empty or holdout.empty:
            station_reports.append(
                {
                    "place_id": str(station_id),
                    "status": "skipped",
                    "reason": "Insufficient history for a seven-day holdout",
                    "observations": int(len(series)),
                }
            )
            continue

        training_days = (
            training["ds"].max() - training["ds"].min()
        ).total_seconds() / 86400

        if training_days < MIN_TRAIN_DAYS or len(holdout) < 2:
            station_reports.append(
                {
                    "place_id": str(station_id),
                    "status": "skipped",
                    "reason": "Insufficient training history or holdout observations",
                    "observations": int(len(series)),
                }
            )
            continue

        train_frame = training[["ds", "y"]].copy()
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            uncertainty_samples=100,
        )
        model.fit(train_frame)

        forecast = model.predict(holdout[["ds"]])
        actual = holdout["y"].to_numpy()
        predicted = forecast["yhat"].to_numpy()

        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

        model_path = MODEL_DIR / f"prophet_station_{station_id}.joblib"
        # Save Prophet's fitted model with joblib for this local prototype.
        import joblib

        joblib.dump(model, model_path)

        prediction_frame = holdout[["ds", "y"]].copy()
        prediction_frame["yhat"] = predicted
        prediction_frame["yhat_lower"] = forecast["yhat_lower"].to_numpy()
        prediction_frame["yhat_upper"] = forecast["yhat_upper"].to_numpy()

        prediction_path = REPORT_DIR / f"predictions_station_{station_id}.csv"
        prediction_frame.to_csv(prediction_path, index=False)

        fig = model.plot(forecast)
        fig.suptitle(f"Prophet occupancy forecast: {station_id}")
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"forecast_station_{station_id}.png",
            dpi=140,
        )
        plt.close(fig)

        station_reports.append(
            {
                "place_id": str(station_id),
                "status": "trained",
                "training_observations": int(len(training)),
                "holdout_observations": int(len(holdout)),
                "cutoff_utc": cutoff.isoformat(),
                "mae": mae,
                "rmse": rmse,
                "model_file": str(model_path.relative_to(ROOT)),
                "predictions_file": str(prediction_path.relative_to(ROOT)),
            }
        )

        print(
            f"[OK] {station_id}: MAE={mae:.4f}, RMSE={rmse:.4f}, holdout={len(holdout)}"
        )

    trained = [result for result in station_reports if result["status"] == "trained"]

    report = {
        "model": "Prophet",
        "forecast_variable": "occupancy_ratio",
        "occupancy_ratio_definition": "bikes / docks",
        "resampling_frequency": FREQUENCY,
        "holdout_days": HOLDOUT_DAYS,
        "split_strategy": "last seven days per station",
        "input_file": str(input_path.relative_to(ROOT)),
        "stations_seen": len(station_reports),
        "stations_trained": len(trained),
        "stations_skipped": len(station_reports) - len(trained),
        "mean_station_mae": (
            float(np.mean([item["mae"] for item in trained])) if trained else None
        ),
        "mean_station_rmse": (
            float(np.mean([item["rmse"] for item in trained])) if trained else None
        ),
        "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        "stations": station_reports,
    }

    report_path = REPORT_DIR / "prophet_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nEvaluation saved to: {report_path}")

    if not trained:
        raise SystemExit(
            "No station had enough history to train. Use a longer input date range."
        )


if __name__ == "__main__":
    main()
