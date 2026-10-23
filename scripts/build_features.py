"""Build a leakage-aware one-hour-ahead forecasting dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

FREQUENCY = "15min"
FORECAST_STEPS = 4  # Four 15-minute intervals = one hour
MAX_FORWARD_FILL = 2  # At most 30 minutes of carried-forward features


def build_station_features(station: pd.DataFrame) -> pd.DataFrame:
    """Create features for one station on a regular time grid."""
    station = station.sort_values("query_time").copy()
    station = station.drop_duplicates("query_time", keep="last")

    station_id = str(station["place_id"].iloc[0])
    station = station.set_index("query_time")

    # Keep the original observations as-is. Resampling does not fill gaps.
    regular = station[["bikes", "empty_docks", "docks"]].resample(FREQUENCY).last()

    # Only historical predictors may be forward-filled.
    regular["bikes_feature"] = regular["bikes"].ffill(limit=MAX_FORWARD_FILL)
    regular["bikes_current"] = regular["bikes_feature"]

    # Lag features refer only to current or previous information.
    regular["bike_lag_1"] = regular["bikes_feature"].shift(1)
    regular["bike_lag_2"] = regular["bikes_feature"].shift(2)
    regular["bike_lag_4"] = regular["bikes_feature"].shift(4)
    regular["bike_lag_8"] = regular["bikes_feature"].shift(8)

    # Shift before rolling so the current observation is excluded.
    history = regular["bikes_feature"].shift(1)
    regular["bike_rolling_mean_1h"] = history.rolling(4, min_periods=4).mean()
    regular["bike_rolling_std_1h"] = history.rolling(4, min_periods=4).std()
    regular["bike_rolling_mean_3h"] = history.rolling(12, min_periods=8).mean()

    # Calendar features are known at prediction time.
    regular["hour"] = regular.index.hour
    regular["day_of_week"] = regular.index.dayofweek
    regular["is_weekend"] = (regular.index.dayofweek >= 5).astype(int)
    regular["hour_sin"] = __import__("numpy").sin(
        2 * __import__("numpy").pi * regular["hour"] / 24
    )
    regular["hour_cos"] = __import__("numpy").cos(
        2 * __import__("numpy").pi * regular["hour"] / 24
    )

    # Target is the actual observation exactly one hour in the future.
    regular["target_bikes_1h"] = regular["bikes"].shift(-FORECAST_STEPS)
    regular["target_time"] = regular.index + pd.Timedelta(hours=1)
    regular["place_id"] = station_id
    regular["feature_time"] = regular.index

    # Preserve only rows with a real target and complete required features.
    feature_columns = [
        "bikes_current",
        "bike_lag_1",
        "bike_lag_2",
        "bike_lag_4",
        "bike_lag_8",
        "bike_rolling_mean_1h",
        "bike_rolling_std_1h",
        "bike_rolling_mean_3h",
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
    ]

    output_columns = [
        "place_id",
        "feature_time",
        "target_time",
        *feature_columns,
        "target_bikes_1h",
    ]

    result = regular[output_columns].dropna().reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--output-name",
        default="forecast_features.parquet",
        help="Output Parquet filename",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw = pd.read_parquet(input_path)
    required = {"query_time", "place_id", "bikes", "empty_docks", "docks"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw["query_time"] = pd.to_datetime(raw["query_time"], utc=True)
    raw["bikes"] = pd.to_numeric(raw["bikes"], errors="coerce")
    raw = raw.dropna(subset=["query_time", "place_id", "bikes"])
    raw = raw[raw["bikes"] >= 0]

    if raw.empty:
        raise ValueError("No usable observations found.")

    parts = []
    for _, station in raw.groupby("place_id", sort=False):
        result = build_station_features(station)
        if not result.empty:
            parts.append(result)

    if not parts:
        raise ValueError(
            "No feature rows were generated. The selected time range may "
            "be too short for the one-hour forecast and rolling features."
        )

    features = pd.concat(parts, ignore_index=True)
    features = features.sort_values(["feature_time", "place_id"]).reset_index(drop=True)

    # Essential forecasting safeguards.
    if (features["target_time"] <= features["feature_time"]).any():
        raise AssertionError("Target timestamp must be in the future.")
    if features["target_bikes_1h"].isna().any():
        raise AssertionError("Missing target values found.")
    if features.duplicated(["place_id", "feature_time"]).any():
        raise AssertionError("Duplicate station feature timestamps found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / args.output_name
    features.to_parquet(output_path, index=False)

    report = {
        "input_file": str(input_path.relative_to(ROOT)),
        "output_file": str(output_path.relative_to(ROOT)),
        "frequency": FREQUENCY,
        "forecast_horizon_minutes": 60,
        "max_forward_fill_intervals": MAX_FORWARD_FILL,
        "rows": int(len(features)),
        "stations": int(features["place_id"].nunique()),
        "feature_time_min": features["feature_time"].min().isoformat(),
        "feature_time_max": features["feature_time"].max().isoformat(),
        "target_mean": float(features["target_bikes_1h"].mean()),
        "target_mae_baseline_reference": (
            "Evaluate against persistence and other baselines in Phase 3."
        ),
        "feature_columns": [
            col
            for col in features.columns
            if col not in {"place_id", "feature_time", "target_time", "target_bikes_1h"}
        ],
        "target_leakage_check": "target_time is one hour after feature_time",
    }

    report_path = REPORT_DIR / "feature_build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
