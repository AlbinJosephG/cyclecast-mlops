"""Validate raw observations and build a processed Parquet dataset."""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REQUIRED_COLUMNS = [
    "query_time",
    "place_id",
    "bikes",
    "empty_docks",
    "docks",
]
COUNT_COLUMNS = ["bikes", "empty_docks", "docks"]


def get_daily_files(start: date, end: date):
    files = []
    current = start

    while current <= end:
        path = (
            RAW_DIR
            / f"year={current.year}"
            / f"month={current.month:02d}"
            / f"day={current.day:02d}"
            / "part.csv"
        )
        if path.exists():
            files.append(path)
        current += timedelta(days=1)

    return files


def validate_file(path: Path):
    """Read one daily CSV and return records plus a quality summary."""
    df = pd.read_csv(
        path,
        dtype={"place_id": "string"},
        low_memory=False,
    )

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{path}: missing required columns {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    rows_read = len(df)

    df["query_time"] = pd.to_datetime(
        df["query_time"], errors="coerce", utc=True
    )

    for column in COUNT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_required = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())

    valid_counts = (
        df[COUNT_COLUMNS].notna().all(axis=1)
        & df[COUNT_COLUMNS].ge(0).all(axis=1)
    )

    valid = df.loc[
        df["query_time"].notna()
        & df["place_id"].notna()
        & df["place_id"].str.len().gt(0)
        & valid_counts
    ].copy()

    duplicate_rows = int(
        valid.duplicated(["place_id", "query_time"]).sum()
    )

    # Repeated station/timestamp records are removed deterministically.
    valid = valid.drop_duplicates(
        subset=["place_id", "query_time"],
        keep="last",
    )

    # Keep capacity mismatches in the processed dataset for investigation.
    # They are flagged in the report rather than silently discarded.
    valid["capacity_consistent"] = (
        valid["bikes"] + valid["empty_docks"] == valid["docks"]
    )

    summary = {
        "file": str(path.relative_to(ROOT)),
        "rows_read": rows_read,
        "rows_with_missing_required_values": missing_required,
        "invalid_rows_removed": int(rows_read - len(df.loc[
            df["query_time"].notna()
            & df["place_id"].notna()
            & df["place_id"].str.len().gt(0)
            & valid_counts
        ])),
        "duplicate_rows_removed": duplicate_rows,
        "capacity_mismatch_rows": int(
            (~valid["capacity_consistent"]).sum()
        ),
        "rows_retained": int(len(valid)),
    }

    return valid, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if end < start:
        parser.error("--end must be on or after --start")

    files = get_daily_files(start, end)
    if not files:
        raise SystemExit(
            f"No daily CSV files found for {start} through {end}. "
            "Run ingestion first."
        )

    frames = []
    daily_reports = []

    for path in files:
        print(f"[VALIDATE] {path}")
        frame, report = validate_file(path)
        frames.append(frame)
        daily_reports.append(report)

    data = pd.concat(frames, ignore_index=True)

    duplicate_count = int(
        data.duplicated(["place_id", "query_time"]).sum()
    )
    if duplicate_count:
        data = data.drop_duplicates(
            ["place_id", "query_time"], keep="last"
        )

    data = data.sort_values(
        ["place_id", "query_time"]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        PROCESSED_DIR
        / f"observations_{start.isoformat()}_{end.isoformat()}.parquet"
    )
    report_file = (
        REPORT_DIR
        / f"data_quality_{start.isoformat()}_{end.isoformat()}.json"
    )

    data.to_parquet(output_file, index=False, compression="snappy")

    report = {
        "dataset": "feregrino/london-cycles",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "source_files_found": len(files),
        "total_rows_retained": int(len(data)),
        "unique_stations": int(data["place_id"].nunique()),
        "first_timestamp_utc": (
            data["query_time"].min().isoformat() if len(data) else None
        ),
        "last_timestamp_utc": (
            data["query_time"].max().isoformat() if len(data) else None
        ),
        "duplicate_station_timestamp_pairs": int(
            data.duplicated(["place_id", "query_time"]).sum()
        ),
        "capacity_mismatch_rows": int(
            (~data["capacity_consistent"]).sum()
        ),
        "observed_bike_count_min": (
            int(data["bikes"].min()) if len(data) else None
        ),
        "observed_bike_count_max": (
            int(data["bikes"].max()) if len(data) else None
        ),
        "daily_reports": daily_reports,
        "processed_file": str(output_file.relative_to(ROOT)),
    }

    report_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nDATA QUALITY SUMMARY")
    print(json.dumps(report, indent=2))
    print(f"\nParquet: {output_file}")
    print(f"Report:  {report_file}")

    if data.empty:
        raise SystemExit("No valid records remain after validation.")


if __name__ == "__main__":
    main()
