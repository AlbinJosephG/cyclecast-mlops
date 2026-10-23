"""Phase 4 diagnostics for station-level Prophet forecasts.

Compares Prophet against persistence and seasonal-naive baselines.
Evaluates station-level performance, timestamp alignment, missing
15-minute intervals, and the actual forecast horizon.

This script reuses existing predictions; it does not retrain models.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FREQUENCY = "15min"
HOLDOUT_DAYS = 7
ONE_DAY = pd.Timedelta(days=1)


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE for aligned arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def prepare_station_series(group):
    """Resample station occupancy into 15-minute UTC bins."""
    group = group.sort_values("query_time").copy()
    group = group.drop_duplicates("query_time", keep="last")

    group["occupancy_ratio"] = np.where(
        group["docks"] > 0,
        group["bikes"] / group["docks"],
        np.nan,
    )

    group = group[group["occupancy_ratio"].between(0, 1, inclusive="both")]

    if group.empty:
        return pd.Series(dtype=float)

    # Keep missing 15-minute bins as NaN for coverage diagnostics.
    series = group.set_index("query_time")["occupancy_ratio"].resample(FREQUENCY).mean()

    # Prophet prediction CSV timestamps are timezone-naive UTC.
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)

    return series


def seasonal_naive_prediction(timestamp, training, cutoff):
    """Repeat the latest available training-day value at the same slot.

    Never uses actual observations from the holdout period.
    Searches backward by whole days if the matching slot is missing.
    """
    candidate = timestamp - ONE_DAY

    while candidate >= cutoff:
        candidate -= ONE_DAY

    # Try the same 15-minute slot on earlier training days.
    while candidate >= training.index.min():
        value = training.get(candidate, np.nan)

        if pd.notna(value):
            return float(value)

        candidate -= ONE_DAY

    return np.nan


def safe_mean(values):
    """Return a finite mean, or None if no values are available."""
    values = [v for v in values if pd.notna(v) and np.isfinite(v)]

    if not values:
        return None

    return float(np.mean(values))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    evaluation_path = REPORTS / "prophet_evaluation.json"

    with evaluation_path.open(encoding="utf-8") as file:
        evaluation = json.load(file)

    input_path = ROOT / evaluation["input_file"]

    if not input_path.exists():
        raise FileNotFoundError(f"Processed input not found: {input_path}")

    raw = pd.read_parquet(input_path)

    required = {"query_time", "place_id", "bikes", "docks"}
    missing = required - set(raw.columns)

    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    raw["query_time"] = pd.to_datetime(raw["query_time"], utc=True, errors="coerce")

    for column in ["bikes", "docks"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna(subset=["query_time", "place_id", "bikes", "docks"])

    raw = raw[(raw["bikes"] >= 0) & (raw["docks"] > 0)].copy()

    grouped = {
        str(station_id): group
        for station_id, group in raw.groupby("place_id", sort=False)
    }

    station_results = []
    horizon_results = []
    skipped = []

    for station in evaluation["stations"]:
        if station["status"] != "trained":
            continue

        station_id = str(station["place_id"])
        prediction_path = ROOT / station["predictions_file"]

        if station_id not in grouped or not prediction_path.exists():
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "Source data or prediction file missing",
                }
            )
            continue

        raw_station = grouped[station_id]
        series = prepare_station_series(raw_station)

        # Match the original training script's split strategy.
        observed_series = series.dropna()
        if observed_series.empty:
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "No valid resampled observations",
                }
            )
            continue

        cutoff = observed_series.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)

        training = observed_series[observed_series.index < cutoff]

        if training.empty:
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "No training observations before cutoff",
                }
            )
            continue

        predictions = pd.read_csv(prediction_path)

        required_prediction_columns = {"ds", "y", "yhat"}
        if not required_prediction_columns.issubset(predictions.columns):
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "Prediction CSV missing required columns",
                }
            )
            continue

        predictions["ds"] = pd.to_datetime(
            predictions["ds"], utc=True, errors="coerce"
        ).dt.tz_localize(None)

        for column in ["y", "yhat"]:
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

        predictions = (
            predictions.dropna(subset=["ds", "y", "yhat"])
            .sort_values("ds")
            .drop_duplicates("ds", keep="last")
        )

        # Confirm the predictions correspond to the expected holdout.
        predictions = predictions[predictions["ds"] >= cutoff].copy()

        if predictions.empty:
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "No valid holdout predictions",
                }
            )
            continue

        # Reconstruct the original test observations independently.
        actual_lookup = observed_series
        predictions["source_actual"] = predictions["ds"].map(actual_lookup)

        matched = predictions.dropna(subset=["source_actual"]).copy()

        if matched.empty:
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "No prediction timestamps match source data",
                }
            )
            continue

        # Use source observations as the authoritative actual values.
        # This also reveals any discrepancy in the stored prediction CSV.
        predictions = matched
        predictions["actual"] = predictions["source_actual"]

        predictions["prophet_clipped"] = predictions["yhat"].clip(0, 1)

        last_training_timestamp = training.index.max()
        last_training_value = float(training.iloc[-1])

        predictions["persistence"] = last_training_value

        predictions["seasonal_naive"] = predictions["ds"].map(
            lambda ts: seasonal_naive_prediction(ts, training, cutoff)
        )

        # Raw Prophet metrics.
        raw_metrics = calculate_metrics(
            predictions["actual"],
            predictions["yhat"],
        )

        # All main comparisons use the same timestamps.
        common = predictions.dropna(subset=["seasonal_naive"]).copy()

        if common.empty:
            skipped.append(
                {
                    "place_id": station_id,
                    "reason": "No seasonal-naive predictions available",
                }
            )
            continue

        baseline_metrics = calculate_metrics(common["actual"], common["persistence"])
        seasonal_metrics = calculate_metrics(common["actual"], common["seasonal_naive"])
        clipped_metrics = calculate_metrics(common["actual"], common["prophet_clipped"])

        # Calculate whether each approach wins on this station.

        clipped_mae = clipped_metrics["mae"]
        persistence_mae = baseline_metrics["mae"]
        seasonal_mae = seasonal_metrics["mae"]

        station_results.append(
            {
                "place_id": station_id,
                "holdout_observations": len(predictions),
                "common_comparison_observations": len(common),
                "prophet_mae": raw_metrics["mae"],
                "prophet_rmse": raw_metrics["rmse"],
                "clipped_prophet_mae": clipped_metrics["mae"],
                "clipped_prophet_rmse": clipped_metrics["rmse"],
                "persistence_mae": persistence_mae,
                "persistence_rmse": baseline_metrics["rmse"],
                "seasonal_naive_mae": seasonal_mae,
                "seasonal_naive_rmse": seasonal_metrics["rmse"],
                "clipped_prophet_vs_persistence": (
                    "win"
                    if clipped_mae < persistence_mae
                    else "loss"
                    if clipped_mae > persistence_mae
                    else "tie"
                ),
                "clipped_prophet_vs_seasonal": (
                    "win"
                    if clipped_mae < seasonal_mae
                    else "loss"
                    if clipped_mae > seasonal_mae
                    else "tie"
                ),
                "clipped_prophet_mae_improvement_vs_persistence_pct": (
                    100 * (persistence_mae - clipped_mae) / persistence_mae
                    if persistence_mae > 0
                    else None
                ),
            }
        )

        # Forecast-horizon and resampling diagnostics.
        horizons = (
            predictions["ds"] - last_training_timestamp
        ).dt.total_seconds() / 3600

        observed_grid = series.dropna()


        holdout_grid = pd.date_range(
            start=observed_grid.index.min(),
            end=observed_grid.index.max(),
            freq=FREQUENCY,
        )

        missing_bins_total = len(series) - int(series.notna().sum())

        # Source observation intervals for this station.
        source_times = raw_station["query_time"].drop_duplicates().sort_values()
        source_intervals = source_times.diff().dropna().dt.total_seconds() / 60

        source_interval_median = (
            float(source_intervals.median()) if not source_intervals.empty else None
        )
        source_interval_p95 = (
            float(source_intervals.quantile(0.95))
            if not source_intervals.empty
            else None
        )

        prediction_timestamp_set = set(predictions["ds"])
        missing_prediction_slots = sum(
            timestamp not in prediction_timestamp_set for timestamp in holdout_grid
        )

        horizon_results.append(
            {
                "place_id": station_id,
                "cutoff_utc": cutoff.isoformat(),
                "last_training_timestamp_utc": (last_training_timestamp.isoformat()),
                "first_test_timestamp_utc": (predictions["ds"].min().isoformat()),
                "last_test_timestamp_utc": (predictions["ds"].max().isoformat()),
                "test_predictions": len(predictions),
                "minimum_horizon_hours": float(horizons.min()),
                "median_horizon_hours": float(horizons.median()),
                "maximum_horizon_hours": float(horizons.max()),
                "resampled_bins": len(series),
                "observed_resampled_bins": int(series.notna().sum()),
                "missing_resampled_bins": int(missing_bins_total),
                "resampled_bin_coverage_pct": (
                    100 * series.notna().sum() / len(series) if len(series) else None
                ),
                "holdout_expected_15min_slots": len(holdout_grid),
                "holdout_missing_prediction_slots": (int(missing_prediction_slots)),
                "source_interval_median_minutes": source_interval_median,
                "source_interval_p95_minutes": source_interval_p95,
                "stored_vs_source_actual_max_abs_difference": float(
                    np.max(
                        np.abs(
                            predictions["y"].to_numpy()
                            - predictions["actual"].to_numpy()
                        )
                    )
                ),
            }
        )

    if not station_results:
        raise SystemExit(
            "No stations could be evaluated. Check input paths and reports."
        )

    station_df = pd.DataFrame(station_results)
    horizon_df = pd.DataFrame(horizon_results)

    station_csv = REPORTS / "station_level_comparison.csv"
    horizon_csv = REPORTS / "forecast_horizon_diagnostics.csv"

    station_df.to_csv(station_csv, index=False)
    horizon_df.to_csv(horizon_csv, index=False)

    def wins_losses_ties(column):
        counts = station_df[column].value_counts()
        return {
            "wins": int(counts.get("win", 0)),
            "losses": int(counts.get("loss", 0)),
            "ties": int(counts.get("tie", 0)),
        }

    common_summary = {
        "stations_evaluated": len(station_df),
        "mean_station_mae": {
            "persistence": safe_mean(station_df["persistence_mae"]),
            "seasonal_naive": safe_mean(station_df["seasonal_naive_mae"]),
            "prophet_unclipped": safe_mean(station_df["prophet_mae"]),
            "prophet_clipped": safe_mean(station_df["clipped_prophet_mae"]),
        },
        "mean_station_rmse": {
            "persistence": safe_mean(station_df["persistence_rmse"]),
            "seasonal_naive": safe_mean(station_df["seasonal_naive_rmse"]),
            "prophet_unclipped": safe_mean(station_df["prophet_rmse"]),
            "prophet_clipped": safe_mean(station_df["clipped_prophet_rmse"]),
        },
        "clipped_prophet_station_results": {
            "vs_persistence": wins_losses_ties("clipped_prophet_vs_persistence"),
            "vs_seasonal_naive": wins_losses_ties("clipped_prophet_vs_seasonal"),
        },
        "mean_station_mae_improvement_pct": {
            "clipped_prophet_vs_persistence": safe_mean(
                station_df["clipped_prophet_mae_improvement_vs_persistence_pct"]
            ),
        },
        "horizon_diagnostics": {
            "stations_with_horizon_data": len(horizon_df),
            "median_of_station_median_horizons_hours": safe_mean(
                horizon_df["median_horizon_hours"]
            ),
            "minimum_observed_horizon_hours": (
                float(horizon_df["minimum_horizon_hours"].min())
            ),
            "maximum_observed_horizon_hours": (
                float(horizon_df["maximum_horizon_hours"].max())
            ),
            "mean_resampled_bin_coverage_pct": safe_mean(
                horizon_df["resampled_bin_coverage_pct"]
            ),
            "total_missing_resampled_bins": int(
                horizon_df["missing_resampled_bins"].sum()
            ),
            "total_missing_holdout_prediction_slots": int(
                horizon_df["holdout_missing_prediction_slots"].sum()
            ),
            "max_stored_vs_source_actual_difference": float(
                horizon_df["stored_vs_source_actual_max_abs_difference"].max()
            ),
        },
        "comparison_method": (
            "All baseline and Prophet metrics are calculated on matching "
            "test timestamps where seasonal-naive predictions are available. "
            "Seasonal-naive forecasts only use training observations; "
            "test actuals are never used as predictors."
        ),
        "seasonal_naive_definition": (
            "For each test timestamp, use the latest available training "
            "observation at the same 15-minute clock slot on an earlier day."
        ),
        "skipped_stations": skipped,
        "output_files": {
            "station_level_csv": str(station_csv.relative_to(ROOT)),
            "horizon_diagnostics_csv": str(horizon_csv.relative_to(ROOT)),
        },
    }

    report_path = REPORTS / "phase4_diagnostics.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(common_summary, file, indent=2)

    print("\n=== PHASE 4: BASELINE COMPARISON ===")
    print(f"Stations evaluated: {len(station_df)}")
    print("\nMean station MAE (lower is better):")

    for name, value in common_summary["mean_station_mae"].items():
        print(f"  {name:22s}: {value:.4f}")

    print("\nMean station RMSE (lower is better):")

    for name, value in common_summary["mean_station_rmse"].items():
        print(f"  {name:22s}: {value:.4f}")

    print("\nClipped Prophet station wins/losses/ties:")
    print(
        "  vs persistence:",
        common_summary["clipped_prophet_station_results"]["vs_persistence"],
    )
    print(
        "  vs seasonal naive:",
        common_summary["clipped_prophet_station_results"]["vs_seasonal_naive"],
    )

    print("\n=== FORECAST HORIZON DIAGNOSTICS ===")
    print(
        "Median of station median horizons (hours):",
        common_summary["horizon_diagnostics"][
            "median_of_station_median_horizons_hours"
        ],
    )
    print(
        "Observed horizon range (hours):",
        common_summary["horizon_diagnostics"]["minimum_observed_horizon_hours"],
        "to",
        common_summary["horizon_diagnostics"]["maximum_observed_horizon_hours"],
    )
    print(
        "Mean resampled-bin coverage (%):",
        common_summary["horizon_diagnostics"]["mean_resampled_bin_coverage_pct"],
    )
    print(
        "Total missing 15-minute bins:",
        common_summary["horizon_diagnostics"]["total_missing_resampled_bins"],
    )
    print(
        "Total missing prediction slots:",
        common_summary["horizon_diagnostics"]["total_missing_holdout_prediction_slots"],
    )

    print("\nReports saved:")
    print(f"  {report_path}")
    print(f"  {station_csv}")
    print(f"  {horizon_csv}")


if __name__ == "__main__":
    main()
