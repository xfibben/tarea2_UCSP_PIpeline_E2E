FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
    MLFLOW_ARTIFACT_ROOT=/app/mlartifacts

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY reports ./reports

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 8000 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

