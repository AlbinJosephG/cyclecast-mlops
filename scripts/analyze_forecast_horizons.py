from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

PREDICTION_PATTERN = "predictions_fullgrid_station_*.csv"
HORIZON_FILE = REPORT_DIR / "phase4_final_horizon_diagnostics.csv"

HORIZON_BINS = [0, 24, 48, 72, 96, 120, 144, 168, np.inf]
HORIZON_LABELS = [
    "0-24h",
    "24-48h",
    "48-72h",
    "72-96h",
    "96-120h",
    "120-144h",
    "144-168h",
    ">168h",
]

MODEL_COLUMNS = {
    "persistence": "persistence",
    "seasonal_naive": "seasonal_naive",
    "prophet_unclipped": "yhat",
    "prophet_clipped": "yhat_clipped",
}


def calculate_metrics(actual, predicted):
    valid = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")],
        axis=1,
    ).dropna()

    if valid.empty:
        return None, None, 0

    mae = mean_absolute_error(valid["actual"], valid["predicted"])
    rmse = np.sqrt(mean_squared_error(valid["actual"], valid["predicted"]))

    return float(mae), float(rmse), len(valid)


def main():
    if not HORIZON_FILE.exists():
        raise FileNotFoundError(f"Missing horizon diagnostics: {HORIZON_FILE}")

    horizon_info = pd.read_csv(HORIZON_FILE)
    horizon_info["station_id"] = horizon_info["station_id"].astype(str)
    horizon_info["last_training_timestamp"] = pd.to_datetime(
        horizon_info["last_training_timestamp"]
    )

    training_times = horizon_info.set_index("station_id")[
        "last_training_timestamp"
    ].to_dict()

    prediction_files = sorted(REPORT_DIR.glob(PREDICTION_PATTERN))

    if not prediction_files:
        raise FileNotFoundError(
            "No full-grid prediction files found. Run scripts\\assess_phase4.py first."
        )

    frames = []

    for file_path in prediction_files:
        station_id = file_path.stem.replace("predictions_fullgrid_station_", "")

        if station_id not in training_times:
            print(f"Skipping {station_id}: no horizon metadata")
            continue

        df = pd.read_csv(file_path, parse_dates=["ds"])

        required = {
            "ds",
            "y",
            "persistence",
            "seasonal_naive",
            "yhat",
            "yhat_clipped",
        }

        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{file_path.name} is missing columns: {sorted(missing)}")

        df["station_id"] = station_id

        # Forecast lead time is measured from the last training observation.
        df["horizon_hours"] = (
            df["ds"] - training_times[station_id]
        ).dt.total_seconds() / 3600

        df["horizon_bucket"] = pd.cut(
            df["horizon_hours"],
            bins=HORIZON_BINS,
            labels=HORIZON_LABELS,
            right=True,
            include_lowest=True,
        )

        # Use the same timestamps for all four model comparisons.
        df = df.dropna(subset=["y", *MODEL_COLUMNS.values(), "horizon_bucket"])

        frames.append(
            df[
                [
                    "station_id",
                    "y",
                    "horizon_hours",
                    "horizon_bucket",
                    *MODEL_COLUMNS.values(),
                ]
            ]
        )

    if not frames:
        raise RuntimeError("No valid prediction data was available.")

    data = pd.concat(frames, ignore_index=True)

    data = data[(data["horizon_hours"] > 0) & (data["horizon_hours"] <= 168)].copy()
    station_rows = []

    # Calculate each station's metrics within each horizon bucket.
    for (bucket, station_id), group in data.groupby(
        ["horizon_bucket", "station_id"],
        observed=True,
    ):
        row = {
            "horizon_bucket": str(bucket),
            "station_id": station_id,
            "observations": len(group),
        }

        for model_name, column in MODEL_COLUMNS.items():
            mae, rmse, count = calculate_metrics(group["y"], group[column])
            row[f"{model_name}_mae"] = mae
            row[f"{model_name}_rmse"] = rmse
            row[f"{model_name}_observations"] = count

        station_rows.append(row)

    station_metrics = pd.DataFrame(station_rows)

    station_metrics.to_csv(
        REPORT_DIR / "phase4_horizon_station_metrics.csv",
        index=False,
    )

    # Summarize both macro (equal weight per station) and pooled metrics.
    summary_rows = []

    for bucket in HORIZON_LABELS:
        group = data.loc[data["horizon_bucket"].astype(str) == bucket]
        station_group = station_metrics.loc[station_metrics["horizon_bucket"] == bucket]

        if group.empty:
            continue

        row = {
            "horizon_bucket": bucket,
            "stations_evaluated": int(group["station_id"].nunique()),
            "pooled_observations": int(len(group)),
        }

        for model_name, column in MODEL_COLUMNS.items():
            mae, rmse, count = calculate_metrics(group["y"], group[column])

            row[f"{model_name}_pooled_mae"] = mae
            row[f"{model_name}_pooled_rmse"] = rmse
            row[f"{model_name}_pooled_observations"] = count

            row[f"{model_name}_mean_station_mae"] = station_group[
                f"{model_name}_mae"
            ].mean()
            row[f"{model_name}_mean_station_rmse"] = station_group[
                f"{model_name}_rmse"
            ].mean()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    output_path = REPORT_DIR / "phase4_horizon_summary.csv"
    summary.to_csv(output_path, index=False)

    display_columns = [
        "horizon_bucket",
        "stations_evaluated",
        "prophet_clipped_mean_station_mae",
        "seasonal_naive_mean_station_mae",
        "prophet_clipped_mean_station_rmse",
        "seasonal_naive_mean_station_rmse",
    ]

    print("\n=== FORECAST-HORIZON ASSESSMENT ===")
    print(summary[display_columns].round(4).to_string(index=False))
    print(f"\nSummary saved to: {output_path}")
    print(
        "Station-level metrics saved to: "
        f"{REPORT_DIR / 'phase4_horizon_station_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
