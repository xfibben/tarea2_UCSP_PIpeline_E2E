from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    reference_dir: Path = data_dir / "reference"
    reports_dir: Path = project_root / "reports"
    generated_reports_dir: Path = reports_dir / "generated"
    logs_dir: Path = project_root / "logs"
    models_dir: Path = project_root / "models"
    mlflow_dir: Path = project_root / "mlflow"
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{project_root / 'mlflow' / 'mlflow.db'}")
    mlflow_artifact_root: Path = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", str(project_root / "mlartifacts")))
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "tarea2-breast-cancer")
    registered_model_name: str = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "BreastCancerClassifier")
    random_state: int = 42
    test_size: float = 0.2


settings = Settings()


def ensure_directories() -> None:
    for directory in (
        settings.raw_dir,
        settings.processed_dir,
        settings.reference_dir,
        settings.reports_dir,
        settings.generated_reports_dir,
        settings.logs_dir,
        settings.models_dir,
        settings.mlflow_dir,
        settings.mlflow_artifact_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
