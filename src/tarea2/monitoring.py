from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tarea2.config import ensure_directories, settings


def monitor_batch(input_path: Path) -> dict[str, object]:
    ensure_directories()
    reference_path = settings.reference_dir / "reference_features.csv"

    if not reference_path.exists():
        raise FileNotFoundError("No existe dataset de referencia. Ejecuta primero el pipeline.")
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de monitoreo: {input_path}")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(input_path)

    feature_columns = [column for column in reference_df.columns if column in current_df.columns]
    drift_results: list[dict[str, object]] = []

    for column in feature_columns:
        statistic, p_value = ks_2samp(reference_df[column], current_df[column])
        drift_results.append(
            {
                "feature": column,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < 0.05),
            }
        )

    summary = {
        "drift_features": sum(1 for item in drift_results if item["drift_detected"]),
        "total_features": len(drift_results),
    }

    performance = None
    if {"target", "prediction"}.issubset(current_df.columns):
        performance = {
            "accuracy": accuracy_score(current_df["target"], current_df["prediction"]),
            "precision": precision_score(current_df["target"], current_df["prediction"]),
            "recall": recall_score(current_df["target"], current_df["prediction"]),
            "f1_score": f1_score(current_df["target"], current_df["prediction"]),
        }

    output = {
        "input_path": str(input_path),
        "summary": summary,
        "drift": drift_results,
        "performance": performance,
    }

    output_path = settings.generated_reports_dir / "monitoring_report.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

