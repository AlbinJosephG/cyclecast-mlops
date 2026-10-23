from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
HORIZON_FILE = REPORT_DIR / "phase4_final_horizon_diagnostics.csv"
PREDICTION_PATTERN = "predictions_fullgrid_station_*.csv"

horizon = pd.read_csv(HORIZON_FILE)
horizon["station_id"] = horizon["station_id"].astype(str)
horizon["last_training_timestamp"] = pd.to_datetime(horizon["last_training_timestamp"])

common_cutoff = pd.Timestamp("2024-05-24 23:30:00")
common_end = common_cutoff + pd.Timedelta(days=7)

rows = []

for _, item in horizon.iterrows():
    station_id = item["station_id"]
    path = REPORT_DIR / f"predictions_fullgrid_station_{station_id}.csv"

    if not path.exists():
        continue

    df = pd.read_csv(path, parse_dates=["ds"])
    ds = df["ds"]

    rows.append(
        {
            "station_id": station_id,
            "last_training_timestamp": item["last_training_timestamp"],
            "first_prediction_timestamp": ds.min(),
            "last_prediction_timestamp": ds.max(),
            "rows_after_common_cutoff": int((ds > common_cutoff).sum()),
            "rows_within_common_7day_window": int(
                ((ds > common_cutoff) & (ds <= common_end)).sum()
            ),
            "actuals_within_common_7day_window": int(
                ((ds > common_cutoff) & (ds <= common_end) & df["y"].notna()).sum()
            ),
        }
    )

result = pd.DataFrame(rows)
result.to_csv(
    REPORT_DIR / "phase4_common_window_diagnostic.csv",
    index=False,
)

print("Stations inspected:", len(result))
print("Common cutoff:", common_cutoff)
print("Common window end:", common_end)

print("\nPrediction range:")
print("Earliest first prediction:", result["first_prediction_timestamp"].min())
print("Latest first prediction:", result["first_prediction_timestamp"].max())
print("Earliest last prediction:", result["last_prediction_timestamp"].min())
print("Latest last prediction:", result["last_prediction_timestamp"].max())


print("\nStations with earlier training timestamps:")
print(
    result[result["last_training_timestamp"] < common_cutoff][
        [
            "station_id",
            "last_training_timestamp",
            "first_prediction_timestamp",
            "last_prediction_timestamp",
            "rows_within_common_7day_window",
            "actuals_within_common_7day_window",
        ]
    ].to_string(index=False)
)

print("\nSaved diagnostic:")
print(REPORT_DIR / "phase4_common_window_diagnostic.csv")
