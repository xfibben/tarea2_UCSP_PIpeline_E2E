from __future__ import annotations

import argparse
import json
from pathlib import Path

from tarea2.data import ingest_data, preprocess_data
from tarea2.modeling import train_and_register
from tarea2.monitoring import monitor_batch


def run_all() -> dict[str, object]:
    raw_path = ingest_data()
    processed_paths = preprocess_data(raw_path)
    return train_and_register(processed_paths["train"], processed_paths["test"])


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI del pipeline ML end-to-end")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run-all", help="Ejecuta el pipeline completo")

    monitor_parser = subparsers.add_parser("monitor", help="Ejecuta monitoreo sobre un batch")
    monitor_parser.add_argument("--input", required=True, type=Path, help="CSV con predicciones a monitorear")

    args = parser.parse_args()

    if args.command == "run-all":
        result = run_all()
    else:
        result = monitor_batch(args.input)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

