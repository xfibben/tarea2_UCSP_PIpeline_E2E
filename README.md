# Tarea 2 - Pipeline ML End-to-End

Este repositorio implementa un pipeline de Machine Learning end-to-end con enfoque MLOps para clasificar tumores de mama como benignos o malignos a partir de variables clinicas del dataset `breast_cancer` de `scikit-learn`.

La idea de la entrega no es solo entrenar un modelo, sino mostrar como un modelo puede pasar de un experimento aislado a un flujo trazable, reproducible, versionado, desplegable y monitoreable.

## Objetivo del caso

El caso se plantea para una red de clinicas o laboratorios que recibe estudios de pacientes y necesita priorizar los casos con mayor probabilidad de malignidad para acelerar la revision medica.

El modelo no reemplaza el diagnostico clinico. Su valor esta en apoyar el triage, ordenar prioridades y dejar trazabilidad sobre como se genero cada version del modelo.

## Componentes implementados

- Ingesta reproducible del dataset
- Preprocesamiento y partición train/test
- Entrenamiento con `LogisticRegression`
- Evaluación automática con métricas de clasificación
- Registro del modelo y experimentos en MLflow
- API de inferencia con FastAPI
- Monitoreo básico de drift y performance
- Contenedores Docker para reproducibilidad y despliegue local

## Decisiones de diseno

- Se uso el dataset `breast_cancer` incluido en `scikit-learn` porque permite que cualquier evaluador ejecute el pipeline sin descargar datos externos.
- Se eligio `LogisticRegression` como modelo base porque es simple, interpretable y suficiente para demostrar el flujo MLOps completo.
- Se uso MLflow local para registrar experimentos, metricas, artefactos y versiones del modelo sin depender de servicios cloud.
- Se definio el alias `champion` para simular el modelo promovido a produccion.
- Se agrego Docker Compose para ejecutar pipeline, API, MLflow y monitoreo en un entorno reproducible.

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

## Validacion realizada

Durante la prueba del proyecto se valido:

- Construccion correcta de la imagen Docker.
- Ejecucion del pipeline completo dentro del contenedor.
- Registro del modelo `BreastCancerClassifier` en MLflow.
- Asignacion del alias `champion` al modelo registrado.
- Levantamiento de la API en `http://localhost:8000`.
- Levantamiento de MLflow en `http://localhost:5001`.
- Prueba real del endpoint `/predict`.
- Generacion del reporte de monitoreo en `reports/generated/monitoring_report.json`.

Metricas obtenidas en la ejecucion validada:

| Metrica | Valor |
| --- | ---: |
| Accuracy | 0.9825 |
| Precision | 0.9861 |
| Recall | 0.9861 |
| F1-score | 0.9861 |
| ROC AUC | 0.9954 |

## Limitaciones

- El dataset es academico, limpio y pequeno; por eso las metricas pueden ser mas altas que en un caso productivo real.
- El monitoreo de drift queda implementado, pero sus conclusiones requieren batches de prediccion mas grandes.
- El modelo debe entenderse como apoyo a priorizacion, no como sustituto de una decision medica.
- En un despliegue real faltarian controles adicionales como autenticacion, CI/CD, monitoreo continuo y validacion con datos recientes.

## Informe tecnico

El borrador editable del informe se encuentra en `reports/informe_tecnico.md`.
