"""Profile cycle-station observations and produce EDA artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def profile_data(input_path: Path) -> dict:
    """Calculate data quality and temporal coverage statistics."""
    df = pd.read_parquet(input_path)

    required = {"query_time", "place_id", "bikes", "empty_docks", "docks"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    if df.empty:
        raise ValueError("Input dataset is empty.")

    df["query_time"] = pd.to_datetime(df["query_time"], utc=True)
    df = df.sort_values(["place_id", "query_time"])

    gaps = (
        df.groupby("place_id", sort=False)["query_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )

    station_summary = (
        df.groupby("place_id")
        .agg(
            observations=("bikes", "size"),
            mean_bikes=("bikes", "mean"),
            max_bikes=("bikes", "max"),
            mean_docks=("docks", "mean"),
        )
        .sort_values("observations", ascending=False)
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: number of observed bikes over time across the network.
    network = (
        df.groupby("query_time", as_index=False)["bikes"]
        .sum()
        .sort_values("query_time")
    )
    plt.figure(figsize=(12, 5))
    plt.plot(network["query_time"], network["bikes"], linewidth=1)
    plt.title("Total observed bikes across stations")
    plt.xlabel("Observation time (UTC)")
    plt.ylabel("Sum of available bikes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "network_bikes_over_time.png", dpi=150)
    plt.close()

    # Plot 2: distribution of bike counts.
    plt.figure(figsize=(9, 5))
    plt.hist(df["bikes"].dropna(), bins=30)
    plt.title("Distribution of observed bikes per station")
    plt.xlabel("Available bikes")
    plt.ylabel("Observation count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bike_count_distribution.png", dpi=150)
    plt.close()

    # Plot 3: sampling intervals. Large gaps can affect forecasting.
    if not gaps.empty:
        plt.figure(figsize=(9, 5))
        plt.hist(gaps.clip(upper=gaps.quantile(0.99)), bins=40)
        plt.title("Time between consecutive observations per station")
        plt.xlabel("Minutes between observations (99th percentile clipped)")
        plt.ylabel("Interval count")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "sampling_intervals.png", dpi=150)
        plt.close()

    report = {
        "input_file": str(input_path.relative_to(ROOT)),
        "row_count": int(len(df)),
        "station_count": int(df["place_id"].nunique()),
        "start_time_utc": df["query_time"].min().isoformat(),
        "end_time_utc": df["query_time"].max().isoformat(),
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
        "duplicate_station_timestamps": int(
            df.duplicated(["place_id", "query_time"]).sum()
        ),
        "capacity_mismatches": int(
            (df["bikes"] + df["empty_docks"] != df["docks"]).sum()
        ),
        "bike_count_summary": {
            key: float(value) for key, value in df["bikes"].describe().items()
        },
        "sampling_interval_minutes": (
            {
                "median": float(gaps.median()),
                "p90": float(gaps.quantile(0.90)),
                "p99": float(gaps.quantile(0.99)),
                "max": float(gaps.max()),
            }
            if not gaps.empty
            else {}
        ),
        "stations_with_most_observations": [
            {
                "place_id": str(index),
                "observations": int(row["observations"]),
                "mean_bikes": float(row["mean_bikes"]),
            }
            for index, row in station_summary.head(10).iterrows()
        ],
        "figures": [
            "reports/figures/network_bikes_over_time.png",
            "reports/figures/bike_count_distribution.png",
            "reports/figures/sampling_intervals.png",
        ],
    }

    report_path = REPORT_DIR / "data_profile.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Processed Parquet path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    report = profile_data(input_path)
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {REPORT_DIR / 'data_profile.json'}")


if __name__ == "__main__":
    main()
