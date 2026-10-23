from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

CUTOFF = pd.Timestamp("2024-05-24 23:30:00")
END = CUTOFF + pd.Timedelta(days=7)

MODEL_COLUMNS = {
    "persistence": "persistence",
    "seasonal_naive": "seasonal_naive",
    "prophet_unclipped": "yhat",
    "prophet_clipped": "yhat_clipped",
}

horizon = pd.read_csv(REPORT_DIR / "phase4_final_horizon_diagnostics.csv")
horizon["station_id"] = horizon["station_id"].astype(str)
horizon["last_training_timestamp"] = pd.to_datetime(horizon["last_training_timestamp"])
training_times = horizon.set_index("station_id")["last_training_timestamp"].to_dict()

station_rows = []
all_rows = []

for path in sorted(REPORT_DIR.glob("predictions_fullgrid_station_*.csv")):
    station_id = path.stem.replace("predictions_fullgrid_station_", "")

    if station_id not in training_times:
        continue

    # Strict comparison: only models trained at the common cutoff.
    if training_times[station_id] != CUTOFF:
        continue

    df = pd.read_csv(path, parse_dates=["ds"])

    # Restrict the evaluation to the common seven-day window.
    df = df[(df["ds"] > CUTOFF) & (df["ds"] <= END)].copy()

    df = df.dropna(subset=["y", *MODEL_COLUMNS.values()])

    if df.empty:
        continue

    row = {
        "station_id": station_id,
        "observations": len(df),
    }

    for name, column in MODEL_COLUMNS.items():
        row[f"{name}_mae"] = mean_absolute_error(df["y"], df[column])
        row[f"{name}_rmse"] = np.sqrt(mean_squared_error(df["y"], df[column]))

    station_rows.append(row)
    df["station_id"] = station_id
    all_rows.append(df)

if not station_rows:
    raise RuntimeError("No stations qualify for common-window evaluation.")

station_metrics = pd.DataFrame(station_rows)
station_metrics.to_csv(
    REPORT_DIR / "phase4_common_window_station_metrics.csv",
    index=False,
)

pooled = pd.concat(all_rows, ignore_index=True)

summary = {
    "common_cutoff": str(CUTOFF),
    "common_window_end": str(END),
    "stations_evaluated": int(station_metrics["station_id"].nunique()),
    "stations_with_actuals": int(pooled["station_id"].nunique()),
    "valid_station_timestamp_rows": int(len(pooled)),
}

for name, column in MODEL_COLUMNS.items():
    summary[f"{name}_pooled_mae"] = float(
        mean_absolute_error(pooled["y"], pooled[column])
    )
    summary[f"{name}_pooled_rmse"] = float(
        np.sqrt(mean_squared_error(pooled["y"], pooled[column]))
    )
    summary[f"{name}_mean_station_mae"] = float(station_metrics[f"{name}_mae"].mean())
    summary[f"{name}_mean_station_rmse"] = float(station_metrics[f"{name}_rmse"].mean())

summary_path = REPORT_DIR / "phase4_common_window_summary.json"
summary_path.write_text(
    pd.Series(summary).to_json(indent=2),
    encoding="utf-8",
)

print("\n=== COMMON-WINDOW EVALUATION ===")
for key, value in summary.items():
    print(f"{key}: {value}")

print(f"\nSummary: {summary_path}")
print(f"Station metrics: {REPORT_DIR / 'phase4_common_window_station_metrics.csv'}")
