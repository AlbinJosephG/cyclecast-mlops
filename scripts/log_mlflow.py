"""Log the existing CycleCast evaluation to a local MLflow experiment."""

import hashlib
import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
SCRIPT_DIR = ROOT / "scripts"

TRACKING_DB = ROOT / "mlflow.db"
EXPERIMENT_NAME = "CycleCast-Prophet-Evaluation"

SUMMARY_FILE = REPORT_DIR / "phase4_common_window_summary.json"
STATION_METRICS_FILE = REPORT_DIR / "phase4_common_window_station_metrics.csv"
DATASET_FILE = (
    ROOT / "data" / "processed" / "observations_2024-05-01_2024-05-31.parquet"
)

ARTIFACT_FILES = [
    SUMMARY_FILE,
    STATION_METRICS_FILE,
    REPORT_DIR / "phase4_horizon_summary.csv",
    REPORT_DIR / "phase4_horizon_station_metrics.csv",
    REPORT_DIR / "phase4_final_diagnostics.json",
    REPORT_DIR / "phase4_final_station_comparison.csv",
    REPORT_DIR / "phase4_final_horizon_diagnostics.csv",
    REPORT_DIR / "phase4_common_window_diagnostic.csv",
    SCRIPT_DIR / "train_model.py",
    SCRIPT_DIR / "assess_phase4.py",
    SCRIPT_DIR / "analyze_forecast_horizons.py",
    SCRIPT_DIR / "evaluate_common_window.py",
    SCRIPT_DIR / "log_mlflow.py",
]

MODEL_NAMES = [
    "persistence",
    "seasonal_naive",
    "prophet_unclipped",
    "prophet_clipped",
]


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_summary(summary: dict) -> None:
    """Fail early if the evaluation summary is incomplete."""
    required = {
        "common_cutoff",
        "common_window_end",
        "stations_evaluated",
        "valid_station_timestamp_rows",
    }

    missing = required.difference(summary)
    if missing:
        raise ValueError(f"Evaluation summary is missing fields: {sorted(missing)}")

    if summary["stations_evaluated"] <= 0:
        raise ValueError("No stations were evaluated.")

    if summary["valid_station_timestamp_rows"] <= 0:
        raise ValueError("No valid evaluation observations were found.")

    for model in MODEL_NAMES:
        for metric in [
            "pooled_mae",
            "pooled_rmse",
            "mean_station_mae",
            "mean_station_rmse",
        ]:
            key = f"{model}_{metric}"
            if key not in summary:
                raise ValueError(f"Evaluation summary is missing metric: {key}")


def main() -> None:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Run the common-window evaluation first: {SUMMARY_FILE}"
        )

    if not STATION_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Missing station-level metrics: {STATION_METRICS_FILE}"
        )

    with SUMMARY_FILE.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    validate_summary(summary)

    # A local SQLite database stores MLflow experiment metadata.
    tracking_uri = f"sqlite:///{TRACKING_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset_exists = DATASET_FILE.exists()

    with mlflow.start_run(run_name="prophet-common-window-evaluation") as run:
        # Experiment configuration and provenance.
        mlflow.log_params(
            {
                "model_family": "Prophet",
                "forecast_target": "occupancy_ratio",
                "target_definition": "bikes / docks",
                "frequency": "15min",
                "holdout_days": 7,
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False,
                "uncertainty_samples": 100,
                "clipping_range": "[0, 1]",
                "split_strategy": "last seven days per station",
                "evaluation_strategy": "common training cutoff",
                "common_cutoff": summary["common_cutoff"],
                "common_window_end": summary["common_window_end"],
                "dataset_filename": DATASET_FILE.name,
                "dataset_available_for_fingerprinting": dataset_exists,
            }
        )

        mlflow.set_tags(
            {
                "project": "CycleCast",
                "stage": "evaluation",
                "model_family": "Prophet",
                "tracking_mode": "local",
                "retraining_performed": "false",
            }
        )

        # Log aggregate evaluation counts.
        mlflow.log_metrics(
            {
                "evaluation.stations_evaluated": float(summary["stations_evaluated"]),
                "evaluation.stations_with_actuals": float(
                    summary.get(
                        "stations_with_actuals",
                        summary["stations_evaluated"],
                    )
                ),
                "evaluation.valid_timestamp_rows": float(
                    summary["valid_station_timestamp_rows"]
                ),
            }
        )

        # Log pooled and equal-weight-per-station metrics.
        for model in MODEL_NAMES:
            for metric in [
                "pooled_mae",
                "pooled_rmse",
                "mean_station_mae",
                "mean_station_rmse",
            ]:
                key = f"{model}_{metric}"
                mlflow.log_metric(
                    f"{model}.{metric}",
                    float(summary[key]),
                )

        # Fingerprint the source dataset, but do not upload the dataset.
        if dataset_exists:
            mlflow.log_param(
                "dataset_sha256",
                sha256_file(DATASET_FILE),
            )
            mlflow.log_param(
                "dataset_size_bytes",
                DATASET_FILE.stat().st_size,
            )

        # Attach existing reports and source files for reproducibility.
        logged_files = []

        for path in ARTIFACT_FILES:
            if path.is_file():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="project_artifacts",
                )
                logged_files.append(str(path.relative_to(ROOT)))

        print("\n=== MLFLOW RUN CREATED ===")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking database: {TRACKING_DB}")
        print(f"Stations evaluated: {summary['stations_evaluated']}")
        print(
            f"Valid station-timestamp rows: {summary['valid_station_timestamp_rows']}"
        )
        print(f"Artifacts logged: {len(logged_files)}")

        if dataset_exists:
            print("Dataset fingerprint: recorded")
        else:
            print("Dataset fingerprint: skipped (processed dataset file not found)")

        print("\nMean station metrics:")
        for model in MODEL_NAMES:
            mae = summary[f"{model}_mean_station_mae"]
            rmse = summary[f"{model}_mean_station_rmse"]
            print(f"{model:20s} MAE={mae:.4f} RMSE={rmse:.4f}")

        print("\nOpen the MLflow UI to inspect this run.")


if __name__ == "__main__":
    main()
