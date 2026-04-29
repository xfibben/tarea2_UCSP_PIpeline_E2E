# Informe Tecnico - Pipeline ML End-to-End

## 1. Resumen ejecutivo

Se implemento un pipeline de Machine Learning end-to-end para clasificar tumores de mama como benignos o malignos. La solucion integra automatizacion, reproducibilidad, trazabilidad experimental y preparacion para produccion mediante MLflow, FastAPI, Docker y monitoreo basico.

## 2. Problema de negocio

En organizaciones de salud, una prediccion temprana de malignidad puede apoyar el triage clinico, priorizar atenciones y reducir tiempos de respuesta. El valor de negocio del modelo radica en:

- Priorizar casos con mayor riesgo
- Estandarizar la evaluacion analitica
- Reducir reprocesos al contar con trazabilidad del modelo
- Facilitar auditoria y mejora continua

## 3. Arquitectura de la solucion

El flujo implementado es:

1. Ingesta del dataset `breast_cancer` desde `scikit-learn`
2. Preprocesamiento y particion train/test
3. Entrenamiento de un modelo `LogisticRegression`
4. Evaluacion automatica con accuracy, precision, recall, F1 y ROC AUC
5. Registro del modelo en MLflow Model Registry
6. Exposicion de inferencia mediante FastAPI
7. Monitoreo de drift de datos y performance

La solucion se ejecuta en contenedores Docker. `docker-compose.yml` define servicios para API, MLflow, ejecucion del pipeline y monitoreo.

## 4. Reproducibilidad y MLOps

- Semilla fija para reproducibilidad
- Dependencias declaradas en `pyproject.toml`
- Pipeline ejecutable con `make pipeline`
- Pipeline reproducible en Docker con `make docker-pipeline`
- Registro de experimentos, metricas y artefactos en MLflow
- Modelo versionado con alias `champion`
- API desacoplada del entrenamiento

## 5. Resultados tecnicos

Las metricas se guardan automaticamente en `reports/generated/metrics.json`. El pipeline tambien genera informacion del modelo registrado y deja trazabilidad del experimento.

## 6. Despliegue e inferencia

La API publica los endpoints `/health`, `/model/info` y `/predict`. La documentacion interactiva se expone en `/docs`. Adicionalmente, cada inferencia se registra en `logs/predictions.csv` para habilitar monitoreo posterior.

## 7. Monitoreo

Se implemento un monitoreo basico orientado a produccion:

- Drift de datos usando la prueba estadistica KS
- Performance del modelo cuando el batch evaluado incluye valores reales

## 8. Conclusiones

La solucion demuestra un pipeline funcional, reproducible y alineado con principios MLOps. Aunque se trata de un caso academico, la arquitectura es extensible a un entorno real con datos corporativos, CI/CD, observabilidad avanzada y despliegue en nube.
