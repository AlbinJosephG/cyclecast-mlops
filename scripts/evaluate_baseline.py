"""Compare Prophet against a persistence baseline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from train_model import prepare_station_data

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def calculate_metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def main():
    evaluation_path = REPORT_DIR / "prophet_evaluation.json"

    with evaluation_path.open(encoding="utf-8") as file:
        evaluation = json.load(file)

    input_path = ROOT / evaluation["input_file"]
    raw = pd.read_parquet(input_path)

    raw["query_time"] = pd.to_datetime(raw["query_time"], utc=True, errors="coerce")
    for column in ["bikes", "docks"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna(subset=["query_time", "place_id", "bikes", "docks"])
    raw = raw[(raw["bikes"] >= 0) & (raw["docks"] > 0)]

    results = []

    for station in evaluation["stations"]:
        if station["status"] != "trained":
            continue

        station_id = station["place_id"]
        group = raw[raw["place_id"].astype(str) == str(station_id)]

        if group.empty:
            continue

        series = prepare_station_data(group)
        if series.empty:
            continue

        cutoff = series["ds"].max() - pd.Timedelta(days=evaluation["holdout_days"])
        training = series[series["ds"] < cutoff]

        if training.empty:
            continue

        prediction_path = ROOT / station["predictions_file"]
        predictions = pd.read_csv(prediction_path)

        # Predict the last training occupancy for every test timestamp.
        baseline_value = float(training["y"].iloc[-1])
        baseline_predictions = np.full(len(predictions), baseline_value)

        actual = predictions["y"].to_numpy()
        prophet_predictions = predictions["yhat"].to_numpy()

        baseline_metrics = calculate_metrics(actual, baseline_predictions)
        prophet_metrics = calculate_metrics(actual, prophet_predictions)
        # Occupancy ratio must remain between 0 and 1.
        clipped_predictions = np.clip(prophet_predictions, 0.0, 1.0)

        clipped_metrics = calculate_metrics(actual, clipped_predictions)

        results.append(
            {
                "place_id": station_id,
                "baseline_value": baseline_value,
                "observations": len(predictions),
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "prophet_mae": prophet_metrics["mae"],
                "prophet_rmse": prophet_metrics["rmse"],
                "mae_improvement_percent": (
                    100
                    * (baseline_metrics["mae"] - prophet_metrics["mae"])
                    / baseline_metrics["mae"]
                    if baseline_metrics["mae"] > 0
                    else None
                ),
                "clipped_prophet_mae": clipped_metrics["mae"],
                "clipped_prophet_rmse": clipped_metrics["rmse"],
            }
        )

    if not results:
        raise SystemExit("No stations could be evaluated.")

    baseline_mae = float(np.mean([row["baseline_mae"] for row in results]))
    baseline_rmse = float(np.mean([row["baseline_rmse"] for row in results]))
    prophet_mae = float(np.mean([row["prophet_mae"] for row in results]))
    prophet_rmse = float(np.mean([row["prophet_rmse"] for row in results]))
    clipped_prophet_mae = float(
        np.mean([row["clipped_prophet_mae"] for row in results])
    )

    clipped_prophet_rmse = float(
        np.mean([row["clipped_prophet_rmse"] for row in results])
    )

    summary = {
        "baseline": "last training occupancy held constant",
        "stations_compared": len(results),
        "baseline_mean_station_mae": baseline_mae,
        "baseline_mean_station_rmse": baseline_rmse,
        "prophet_mean_station_mae": prophet_mae,
        "prophet_mean_station_rmse": prophet_rmse,
        "mae_improvement_percent": (
            100 * (baseline_mae - prophet_mae) / baseline_mae
            if baseline_mae > 0
            else None
        ),
        "station_results": results,
        "clipped_prophet_mean_station_mae": clipped_prophet_mae,
        "clipped_prophet_mean_station_rmse": clipped_prophet_rmse,
    }

    output_path = REPORT_DIR / "baseline_comparison.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Stations compared: {len(results)}")
    print(f"Baseline mean MAE: {baseline_mae:.4f}")
    print(f"Prophet mean MAE:  {prophet_mae:.4f}")
    print(f"Baseline mean RMSE: {baseline_rmse:.4f}")
    print(f"Prophet mean RMSE:  {prophet_rmse:.4f}")
    print(f"Clipped Prophet mean MAE:  {clipped_prophet_mae:.4f}")
    print(f"Clipped Prophet mean RMSE: {clipped_prophet_rmse:.4f}")
    if summary["mae_improvement_percent"] is not None:
        print(f"Prophet MAE improvement: {summary['mae_improvement_percent']:.2f}%")

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
