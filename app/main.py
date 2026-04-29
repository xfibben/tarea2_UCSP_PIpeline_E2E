from __future__ import annotations

from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from tarea2.config import ensure_directories, settings

app = FastAPI(
    title="Tarea 2 ML Inference API",
    description="API de inferencia para el modelo de clasificacion de cancer de mama.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    mean_radius: float = Field(..., examples=[17.99])
    mean_texture: float = Field(..., examples=[10.38])
    mean_perimeter: float = Field(..., examples=[122.8])
    mean_area: float = Field(..., examples=[1001.0])
    mean_smoothness: float = Field(..., examples=[0.1184])
    mean_compactness: float = Field(..., examples=[0.2776])
    mean_concavity: float = Field(..., examples=[0.3001])
    mean_concave_points: float = Field(..., examples=[0.1471])
    mean_symmetry: float = Field(..., examples=[0.2419])
    mean_fractal_dimension: float = Field(..., examples=[0.07871])
    radius_error: float = Field(..., examples=[1.095])
    texture_error: float = Field(..., examples=[0.9053])
    perimeter_error: float = Field(..., examples=[8.589])
    area_error: float = Field(..., examples=[153.4])
    smoothness_error: float = Field(..., examples=[0.006399])
    compactness_error: float = Field(..., examples=[0.04904])
    concavity_error: float = Field(..., examples=[0.05373])
    concave_points_error: float = Field(..., examples=[0.01587])
    symmetry_error: float = Field(..., examples=[0.03003])
    fractal_dimension_error: float = Field(..., examples=[0.006193])
    worst_radius: float = Field(..., examples=[25.38])
    worst_texture: float = Field(..., examples=[17.33])
    worst_perimeter: float = Field(..., examples=[184.6])
    worst_area: float = Field(..., examples=[2019.0])
    worst_smoothness: float = Field(..., examples=[0.1622])
    worst_compactness: float = Field(..., examples=[0.6656])
    worst_concavity: float = Field(..., examples=[0.7119])
    worst_concave_points: float = Field(..., examples=[0.2654])
    worst_symmetry: float = Field(..., examples=[0.4601])
    worst_fractal_dimension: float = Field(..., examples=[0.1189])


def load_model():
    ensure_directories()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        return mlflow.sklearn.load_model(f"models:/{settings.registered_model_name}@champion")
    except Exception:
        fallback_path = settings.models_dir / "champion.joblib"
        if not fallback_path.exists():
            raise RuntimeError("No existe un modelo registrado. Ejecuta primero el pipeline.")
        return joblib.load(fallback_path)


MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


def _append_prediction_log(features: dict[str, float], prediction: int, probability: float) -> None:
    log_path = settings.logs_dir / "predictions.csv"
    row = {
        **features,
        "prediction": prediction,
        "prediction_probability": probability,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, str]:
    return {
        "tracking_uri": settings.mlflow_tracking_uri,
        "model_name": settings.registered_model_name,
        "alias": "champion",
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    payload = request.model_dump()
    frame = pd.DataFrame([payload])
    model = get_model()

    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    label = "benign" if prediction == 1 else "malignant"

    _append_prediction_log(payload, prediction, probability)

    return {
        "prediction": prediction,
        "label": label,
        "probability_benign": probability,
    }
