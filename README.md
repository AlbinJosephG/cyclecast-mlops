
# CycleCast MLOps

A reproducible bike-station availability forecasting project built with
Python, time-series evaluation, experiment tracking, Docker, and AWS.

## Current milestone

Phase 1: data acquisition and data quality.

## Data source

Dataset: https://huggingface.co/datasets/feregrino/london-cycles

The dataset contains historical London cycle-hire station observations.
See its dataset card for schema, known gaps, and attribution requirements.

## Requirements

- Python 3.11
- Git
- Internet access for initial data acquisition
- Docker for later containerization phases

## Local setup

Activate the virtual environment and install dependencies:

    python -m pip install -e ".[dev]"

Download a small date range:

    python scripts/ingest_data.py --start 2024-05-01 --end 2024-05-03

Validate and create Parquet:

    python scripts/validate_data.py --start 2024-05-01 --end 2024-05-03

Run tests:

    python -m pytest -v

## Data layout

- data/raw/: immutable source CSV files and station metadata
- data/processed/: validated Parquet datasets
- reports/: machine-readable quality reports

Large datasets and generated model artifacts are not committed to Git.

## Roadmap

1. Data acquisition and quality
2. Exploratory analysis and temporal splitting
3. Baseline and advanced forecasting
4. MLflow experiment tracking
5. Prediction API and dashboard
6. Docker integration
7. AWS hosting
8. CI/CD and monitoring

## Data attribution

The upstream dataset is distributed under CC-BY-SA 3.0 according to its
dataset card. Review the upstream terms before redistributing derived data.
