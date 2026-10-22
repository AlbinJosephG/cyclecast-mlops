"""Automated tests for CycleCast data validation."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_data import REQUIRED_COLUMNS, validate_file

ROOT = Path(__file__).resolve().parents[1]


def test_required_columns_are_defined():
    assert REQUIRED_COLUMNS == [
        "query_time", "place_id", "bikes", "empty_docks", "docks"
    ]


def test_valid_records_are_retained_and_duplicates_removed():
    fixture = ROOT / "data" / "raw" / "_test_observations.csv"
    fixture.parent.mkdir(parents=True, exist_ok=True)

    records = [
        ["2024-05-01 10:00:00", "BikePoints_1", 5, 10, 15],
        ["2024-05-01 10:00:00", "BikePoints_1", 5, 10, 15],
        ["2024-05-01 10:15:00", "BikePoints_1", -1, 16, 15],
    ]

    try:
        pd.DataFrame(records, columns=REQUIRED_COLUMNS).to_csv(
            fixture, index=False
        )

        result, summary = validate_file(fixture)

        assert len(result) == 1
        assert result.iloc[0]["place_id"] == "BikePoints_1"
        assert summary["duplicate_rows_removed"] == 1
        assert summary["rows_retained"] == 1
    finally:
        fixture.unlink(missing_ok=True)


def test_missing_required_column_is_rejected():
    fixture = ROOT / "data" / "raw" / "_test_invalid.csv"
    fixture.parent.mkdir(parents=True, exist_ok=True)

    try:
        pd.DataFrame({
            "query_time": ["2024-05-01 10:00:00"],
            "place_id": ["BikePoints_1"],
        }).to_csv(fixture, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            validate_file(fixture)
    finally:
        fixture.unlink(missing_ok=True)
