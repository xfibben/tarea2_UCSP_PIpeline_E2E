PYTHON ?= python3

.PHONY: setup pipeline api monitor docker-build docker-pipeline docker-up docker-monitor clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e .

pipeline:
	.venv/bin/python -m tarea2.cli run-all

api:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

monitor:
	.venv/bin/python -m tarea2.cli monitor --input logs/predictions.csv

docker-build:
	docker compose build

docker-pipeline:
	docker compose --profile jobs run --rm pipeline

docker-up:
	docker compose up api mlflow

docker-monitor:
	docker compose --profile jobs run --rm monitor

clean:
	rm -rf .venv __pycache__ .pytest_cache
