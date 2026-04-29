from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tarea2.config import ensure_directories, settings


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["target", "target_label"])
    y = df["target"]
    return x, y


def train_and_register(train_path: Path, test_path: Path) -> dict[str, object]:
    ensure_directories()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=settings.mlflow_experiment_name,
            artifact_location=str(settings.mlflow_artifact_root.resolve()),
        )
    else:
        experiment_id = experiment.experiment_id

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    x_train, y_train = _split_features_target(train_df)
    x_test, y_test = _split_features_target(test_df)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=settings.random_state)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    input_example = x_test.head(2)

    with mlflow.start_run(experiment_id=experiment_id, run_name="baseline-logistic-regression") as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "test_size": settings.test_size,
                "random_state": settings.random_state,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(train_path), artifact_path="datasets")
        mlflow.log_artifact(str(test_path), artifact_path="datasets")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=settings.registered_model_name,
        )

        run_id = run.info.run_id

    latest_versions = client.get_latest_versions(settings.registered_model_name)
    latest_version = max(latest_versions, key=lambda item: int(item.version))
    client.set_registered_model_alias(
        name=settings.registered_model_name,
        alias="champion",
        version=latest_version.version,
    )

    model_info = {
        "run_id": run_id,
        "model_name": settings.registered_model_name,
        "model_version": latest_version.version,
        "metrics": metrics,
    }

    metrics_path = settings.generated_reports_dir / "metrics.json"
    model_info_path = settings.generated_reports_dir / "model_info.json"
    local_model_path = settings.models_dir / "champion.joblib"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_info_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    joblib.dump(model, local_model_path)

    return model_info
