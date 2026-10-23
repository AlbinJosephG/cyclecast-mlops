from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_model import prepare_station_data

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "observations_2024-05-01_2024-05-31.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FREQUENCY = "15min"
HOLDOUT_DAYS = 7
OCCUPANCY_MIN = 0.0
OCCUPANCY_MAX = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Calculate metrics only for timestamps with valid actual and predicted values."""
    matched = pd.concat(
        [
            actual.rename("actual"),
            predicted.rename("predicted"),
        ],
        axis=1,
    ).dropna()

    if matched.empty:
        return {
            "mae": None,
            "rmse": None,
            "observations": 0,
        }

    return {
        "mae": float(mean_absolute_error(matched["actual"], matched["predicted"])),
        "rmse": float(
            np.sqrt(mean_squared_error(matched["actual"], matched["predicted"]))
        ),
        "observations": int(len(matched)),
    }


def seasonal_naive_prediction(
    timestamp: pd.Timestamp,
    training: pd.Series,
    cutoff: pd.Timestamp,
) -> float:
    """
    Use the most recent available training observation at the same
    time of day, looking back in 24-hour increments without leakage.
    """
    candidate = timestamp - pd.Timedelta(days=1)

    while candidate >= cutoff:
        candidate -= pd.Timedelta(days=1)

    earliest_training_time = training.index.min()

    while candidate >= earliest_training_time:
        value = training.get(candidate, np.nan)

        if pd.notna(value):
            return float(value)

        candidate -= pd.Timedelta(days=1)

    return np.nan


def safe_mean(values: list[float]) -> float | None:
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Observation dataset not found: {DATA_PATH}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    logger.info("Loading observation dataset: %s", DATA_PATH)
    observations = pd.read_parquet(DATA_PATH)

    required_columns = {"place_id", "query_time", "bikes", "docks"}
    missing_columns = required_columns.difference(observations.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    observations["place_id"] = observations["place_id"].astype(str)

    station_results = []
    horizon_results = []
    failures = []

    station_groups = observations.groupby("place_id", sort=False)
    logger.info("Stations in input dataset: %d", observations["place_id"].nunique())

    for position, (station_id, group) in enumerate(station_groups, start=1):
        model_path = MODEL_DIR / f"prophet_station_{station_id}.joblib"

        if not model_path.exists():
            failures.append(
                {"station_id": station_id, "reason": "Saved model not found"}
            )
            continue

        try:
            # Reuse the exact preprocessing function from model training.
            series = prepare_station_data(group)

            if series.empty:
                failures.append(
                    {"station_id": station_id, "reason": "No valid observations"}
                )
                continue

            series = series.copy()
            series["ds"] = pd.to_datetime(series["ds"])
            series = (
                series.dropna(subset=["ds", "y"])
                .sort_values("ds")
                .drop_duplicates(subset=["ds"], keep="last")
            )

            if len(series) < 2:
                failures.append(
                    {"station_id": station_id, "reason": "Insufficient observations"}
                )
                continue

            last_timestamp = series["ds"].max()
            cutoff = last_timestamp - pd.Timedelta(days=HOLDOUT_DAYS)

            training = series.loc[series["ds"] < cutoff].copy()

            if training.empty:
                failures.append(
                    {"station_id": station_id, "reason": "Empty training period"}
                )
                continue

            # The complete expected grid includes slots with no actual observation.
            holdout_grid = pd.date_range(
                start=cutoff,
                end=last_timestamp,
                freq=FREQUENCY,
            )

            model = joblib.load(model_path)

            future = pd.DataFrame({"ds": holdout_grid})
            forecast = model.predict(future)

            predictions = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

            # Align actual observations onto the complete grid.
            actual_lookup = series.set_index("ds")["y"]
            predictions["y"] = predictions["ds"].map(actual_lookup)

            predictions["yhat_clipped"] = predictions["yhat"].clip(
                OCCUPANCY_MIN, OCCUPANCY_MAX
            )
            predictions["has_actual"] = predictions["y"].notna()

            # Baselines use training data only; no holdout values are used.
            training_lookup = training.set_index("ds")["y"]
            persistence_value = float(training["y"].iloc[-1])

            predictions["persistence"] = persistence_value
            predictions["seasonal_naive"] = [
                seasonal_naive_prediction(
                    timestamp=timestamp,
                    training=training_lookup,
                    cutoff=cutoff,
                )
                for timestamp in predictions["ds"]
            ]

            # Save full-grid forecasts, including rows with missing actual values.
            output_path = REPORT_DIR / f"predictions_fullgrid_station_{station_id}.csv"
            predictions.to_csv(output_path, index=False)

            # Evaluate only on timestamps where actual occupancy exists.
            evaluated = predictions.loc[predictions["y"].notna()].copy()


            # Compare all models on the same timestamps.
            common = evaluated[
                [
                    "y",
                    "persistence",
                    "seasonal_naive",
                    "yhat",
                    "yhat_clipped",
                ]
            ].dropna()

            common_actual = common["y"]
            common_metrics = {
                "persistence": calculate_metrics(common_actual, common["persistence"]),
                "seasonal_naive": calculate_metrics(
                    common_actual, common["seasonal_naive"]
                ),
                "prophet_unclipped": calculate_metrics(common_actual, common["yhat"]),
                "prophet_clipped": calculate_metrics(
                    common_actual, common["yhat_clipped"]
                ),
            }

            prophet_mae = common_metrics["prophet_clipped"]["mae"]
            persistence_mae = common_metrics["persistence"]["mae"]
            seasonal_mae = common_metrics["seasonal_naive"]["mae"]

            station_results.append(
                {
                    "station_id": station_id,
                    "expected_grid_slots": int(len(holdout_grid)),
                    "forecast_rows": int(len(predictions)),
                    "actual_observations": int(predictions["has_actual"].sum()),
                    "missing_actual_slots": int(predictions["y"].isna().sum()),
                    "evaluation_observations": int(len(common)),
                    "actual_coverage_pct": float(
                        100 * predictions["has_actual"].mean()
                    ),
                    "persistence_mae": common_metrics["persistence"]["mae"],
                    "persistence_rmse": common_metrics["persistence"]["rmse"],
                    "seasonal_naive_mae": common_metrics["seasonal_naive"]["mae"],
                    "seasonal_naive_rmse": common_metrics["seasonal_naive"]["rmse"],
                    "prophet_unclipped_mae": common_metrics["prophet_unclipped"]["mae"],
                    "prophet_unclipped_rmse": common_metrics["prophet_unclipped"][
                        "rmse"
                    ],
                    "prophet_clipped_mae": prophet_mae,
                    "prophet_clipped_rmse": common_metrics["prophet_clipped"]["rmse"],
                    "clipped_prophet_vs_persistence": (
                        "win"
                        if prophet_mae is not None
                        and persistence_mae is not None
                        and prophet_mae < persistence_mae
                        else "loss_or_tie"
                    ),
                    "clipped_prophet_vs_seasonal_naive": (
                        "win"
                        if prophet_mae is not None
                        and seasonal_mae is not None
                        and prophet_mae < seasonal_mae
                        else "loss_or_tie"
                    ),
                }
            )

            # Forecast lead time is measured from the last training observation.
            last_training_timestamp = training["ds"].max()
            horizons = (
                predictions["ds"] - last_training_timestamp
            ).dt.total_seconds() / 3600

            horizon_results.append(
                {
                    "station_id": station_id,
                    "last_training_timestamp": str(last_training_timestamp),
                    "cutoff": str(cutoff),
                    "holdout_end": str(last_timestamp),
                    "expected_grid_slots": int(len(holdout_grid)),
                    "forecast_rows": int(len(predictions)),
                    "actual_observations": int(predictions["has_actual"].sum()),
                    "missing_actual_slots": int(predictions["y"].isna().sum()),
                    "minimum_horizon_hours": float(horizons.min()),
                    "median_horizon_hours": float(horizons.median()),
                    "maximum_horizon_hours": float(horizons.max()),
                }
            )

            if position % 50 == 0:
                logger.info(
                    "Processed %d/%d station groups",
                    position,
                    len(station_groups),
                )

        except Exception as exc:
            logger.exception("Assessment failed for station %s", station_id)
            failures.append(
                {
                    "station_id": station_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not station_results:
        raise RuntimeError(
            "No stations were evaluated successfully. Check model paths, "
            "dataset schema, and prepare_station_data()."
        )

    station_df = pd.DataFrame(station_results)
    horizon_df = pd.DataFrame(horizon_results)

    station_df.to_csv(
        REPORT_DIR / "phase4_final_station_comparison.csv",
        index=False,
    )
    horizon_df.to_csv(
        REPORT_DIR / "phase4_final_horizon_diagnostics.csv",
        index=False,
    )

    model_names = [
        "persistence",
        "seasonal_naive",
        "prophet_unclipped",
        "prophet_clipped",
    ]

    metric_summary = {}
    for model_name in model_names:
        mae_column = f"{model_name}_mae"
        rmse_column = f"{model_name}_rmse"

        metric_summary[model_name] = {
            "mean_station_mae": safe_mean(station_df[mae_column].dropna().tolist()),
            "mean_station_rmse": safe_mean(station_df[rmse_column].dropna().tolist()),
        }

    win_loss = {}
    for comparison in [
        "clipped_prophet_vs_persistence",
        "clipped_prophet_vs_seasonal_naive",
    ]:
        counts = station_df[comparison].value_counts()
        win_loss[comparison] = {
            "wins": int(counts.get("win", 0)),
            "losses_or_ties": int(counts.get("loss_or_tie", 0)),
        }

    total_grid_slots = int(station_df["expected_grid_slots"].sum())
    total_forecast_rows = int(station_df["forecast_rows"].sum())
    total_actuals = int(station_df["actual_observations"].sum())

    report = {
        "assessment": "Phase 4 full-grid forecast assessment",
        "model": "Prophet",
        "target": "bike occupancy ratio",
        "frequency": FREQUENCY,
        "holdout_days": HOLDOUT_DAYS,
        "stations_evaluated": int(len(station_df)),
        "station_failures": failures,
        "metrics": metric_summary,
        "station_level_mae_comparison": win_loss,
        "coverage": {
            "expected_grid_slots": total_grid_slots,
            "forecast_rows": total_forecast_rows,
            "forecast_grid_coverage_pct": (
                100 * total_forecast_rows / total_grid_slots
                if total_grid_slots
                else None
            ),
            "actual_observations": total_actuals,
            "actual_observation_coverage_pct": (
                100 * total_actuals / total_grid_slots if total_grid_slots else None
            ),
            "missing_actual_slots": total_grid_slots - total_actuals,
        },
        "horizon_hours": {
            "median_of_station_medians": safe_mean(
                horizon_df["median_horizon_hours"].tolist()
            ),
            "minimum_observed": float(horizon_df["minimum_horizon_hours"].min()),
            "maximum_observed": float(horizon_df["maximum_horizon_hours"].max()),
        },
        "notes": [
        "MAE and RMSE are calculated only where actual occupancy is available.",
        (
            "Station-level comparisons use common timestamps available "
            "for all compared models."
        ),
        (
            "Clipped Prophet predictions are constrained to the valid "
            "occupancy-ratio range [0, 1]."
        ),
        (
            "Missing actual observations are reported separately from "
            "forecast-grid coverage."
        ),
        (
            "This is a retrospective holdout assessment, not a guarantee "
            "of future performance."
        ),
    ],
    }

    report_path = REPORT_DIR / "phase4_final_diagnostics.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n=== PHASE 4: FULL-GRID ASSESSMENT ===")
    print(f"Stations evaluated: {len(station_df)}")
    print(f"Station failures: {len(failures)}")

    print("\nMean station metrics (lower is better):")
    for model_name, metrics in metric_summary.items():
        print(
            f"{model_name:22} "
            f"MAE={metrics['mean_station_mae']}  "
            f"RMSE={metrics['mean_station_rmse']}"
        )

    print("\nClipped Prophet station-level MAE comparisons:")
    for name, values in win_loss.items():
        print(
            f"{name}: wins={values['wins']}, losses_or_ties={values['losses_or_ties']}"
        )

    print("\nCoverage:")
    print(f"Expected grid slots: {total_grid_slots:,}")
    print(f"Forecast rows: {total_forecast_rows:,}")
    print(
        "Forecast-grid coverage: "
        f"{report['coverage']['forecast_grid_coverage_pct']:.2f}%"
    )
    print(
        "Actual-observation coverage: "
        f"{report['coverage']['actual_observation_coverage_pct']:.2f}%"
    )
    print(f"Missing actual slots: {total_grid_slots - total_actuals:,}")

    print("\nReports saved:")
    print(REPORT_DIR / "phase4_final_diagnostics.json")
    print(REPORT_DIR / "phase4_final_station_comparison.csv")
    print(REPORT_DIR / "phase4_final_horizon_diagnostics.csv")
    print("Full-grid forecasts: reports/predictions_fullgrid_station_*.csv")

    if failures:
        print("\nFirst station failures:")
        for failure in failures[:10]:
            print(f"  {failure['station_id']}: {failure['reason']}")


if __name__ == "__main__":
    main()
