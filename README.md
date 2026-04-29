# Tarea 2 - Pipeline ML End-to-End

Este repositorio implementa un pipeline de Machine Learning end-to-end con enfoque MLOps para predecir diagnósticos de cáncer de mama a partir de variables clínicas del dataset `breast_cancer` de `scikit-learn`.

## Objetivo del caso

El objetivo de negocio es apoyar a una organización del sector salud con una herramienta analítica que ayude a priorizar casos con alta probabilidad de malignidad, reduciendo tiempos de atención y mejorando la trazabilidad de modelos en producción.

## Componentes implementados

- Ingesta reproducible del dataset
- Preprocesamiento y partición train/test
- Entrenamiento con `LogisticRegression`
- Evaluación automática con métricas de clasificación
- Registro del modelo y experimentos en MLflow
- API de inferencia con FastAPI
- Monitoreo básico de drift y performance
- Contenedores Docker para reproducibilidad y despliegue local

## Estructura

```text
.
├── app/                     # API de inferencia
├── reports/                 # Informe tecnico y artefactos de evaluacion
├── src/tarea2/              # Codigo del pipeline
├── Dockerfile               # Imagen reproducible del proyecto
├── docker-compose.yml       # API, MLflow y jobs del pipeline
├── Makefile                 # Comandos de ejecucion
└── pyproject.toml           # Dependencias
```

## Ejecucion con Docker

1. Construir la imagen:

```bash
make docker-build
```

2. Ejecutar el pipeline completo y registrar el modelo:

```bash
make docker-pipeline
```

3. Levantar API y MLflow:

```bash
make docker-up
```

4. Abrir los servicios:

- API: `http://localhost:8000/docs`
- MLflow: `http://localhost:5001`

5. Ejecutar monitoreo despues de generar predicciones:

```bash
make docker-monitor
```

## Ejecucion local sin Docker

```bash
make setup
make pipeline
make api
```

MLflow UI local:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlartifacts
```

## Flujo del pipeline

```text
ingesta -> preprocesamiento -> entrenamiento -> evaluacion -> registro -> despliegue API -> monitoreo
```

## Endpoints principales

- `GET /health`
- `GET /model/info`
- `POST /predict`

La documentacion interactiva queda disponible en `http://localhost:8000/docs`.

Ejemplo de request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

## Monitoreo

El pipeline guarda un dataset de referencia del entrenamiento y la API registra solicitudes en `logs/predictions.csv`. El comando de monitoreo calcula:

- Drift de datos por variable usando la prueba KS
- Accuracy, precision, recall y F1 si se cuenta con columna `target`

## Informe tecnico

El borrador editable del informe se encuentra en `reports/informe_tecnico.md`.
