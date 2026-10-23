
import hashlib
import json

import pytest
from mlflow.tracking import MlflowClient

from scripts import log_mlflow


@pytest.fixture
def tracking_fixture(tmp_path, monkeypatch):
    """Prepare an isolated MLflow run with synthetic evaluation data."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    summary_file = report_dir / "phase4_common_window_summary.json"
    station_metrics_file = (
        report_dir / "phase4_common_window_station_metrics.csv"
    )
    dataset_file = tmp_path / "observations.parquet"
    source_file = tmp_path / "train_model.py"
    tracking_db = tmp_path / "mlflow.db"

    models = [
        "persistence",
        "seasonal_naive",
        "prophet_unclipped",
        "prophet_clipped",
    ]

    summary = {
        "common_cutoff": "2024-05-24 23:30:00",
        "common_window_end": "2024-05-31 23:30:00",
        "stations_evaluated": 2,
        "stations_with_actuals": 2,
        "valid_station_timestamp_rows": 192,
    }

    for model in models:
        summary[f"{model}_pooled_mae"] = 0.20
        summary[f"{model}_pooled_rmse"] = 0.30
        summary[f"{model}_mean_station_mae"] = 0.21
        summary[f"{model}_mean_station_rmse"] = 0.31

    summary_file.write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    station_metrics_file.write_text(
        "place_id,model,mae,rmse\n"
        "BikePoints_1,persistence,0.20,0.30\n",
        encoding="utf-8",
    )
    dataset_file.write_bytes(b"synthetic dataset fingerprint fixture")
    source_file.write_text(
        "# Synthetic source artifact for testing\n",
        encoding="utf-8",
    )

    experiment_name = "CycleCast-Test-Experiment"

    monkeypatch.setattr(log_mlflow, "SUMMARY_FILE", summary_file)
    monkeypatch.setattr(
        log_mlflow,
        "STATION_METRICS_FILE",
        station_metrics_file,
    )
    monkeypatch.setattr(log_mlflow, "DATASET_FILE", dataset_file)
    monkeypatch.setattr(log_mlflow, "TRACKING_DB", tracking_db)
    monkeypatch.setattr(log_mlflow, "EXPERIMENT_NAME", experiment_name)
    monkeypatch.setattr(
        log_mlflow,
        "ARTIFACT_FILES",
        [summary_file, station_metrics_file, source_file],
    )

    return {
        "summary": summary,
        "dataset": dataset_file,
        "tracking_db": tracking_db,
        "experiment_name": experiment_name,
    }


def test_mlflow_run_records_metrics_parameters_and_artifacts(
    tracking_fixture,
):
    """Verify the logger records a complete run without uploading data."""
    log_mlflow.main()

    tracking_uri = (
        f"sqlite:///{tracking_fixture['tracking_db'].as_posix()}"
    )
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(
        tracking_fixture["experiment_name"]
    )
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert run.info.status == "FINISHED"

    assert run.data.params["model_family"] == "Prophet"
    assert run.data.params["target_definition"] == "bikes / docks"
    assert run.data.tags["retraining_performed"] == "false"


    expected_metrics = [
        "persistence.pooled_mae",
        "seasonal_naive.mean_station_mae",
        "prophet_unclipped.pooled_rmse",
        "prophet_clipped.mean_station_rmse",
    ]
    for metric in expected_metrics:
        assert metric in run.data.metrics

    assert run.data.metrics["evaluation.stations_evaluated"] == 2.0
    assert run.data.params["dataset_sha256"] == hashlib.sha256(
        tracking_fixture["dataset"].read_bytes()
    ).hexdigest()

    artifacts = client.list_artifacts(run.info.run_id, "project_artifacts")
    artifact_paths = {artifact.path for artifact in artifacts}

    assert "project_artifacts/phase4_common_window_summary.json" in (
        artifact_paths
    )
    assert "project_artifacts/phase4_common_window_station_metrics.csv" in (
        artifact_paths
    )
    assert "project_artifacts/train_model.py" in artifact_paths

    assert not any(path.endswith(".parquet") for path in artifact_paths)


def test_mlflow_summary_validation_rejects_missing_metrics():
    with pytest.raises(ValueError, match="Evaluation summary is missing fields"):
        log_mlflow.validate_summary({})
