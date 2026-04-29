from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tarea2.config import ensure_directories, settings


def _to_snake_case(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def ingest_data() -> Path:
    ensure_directories()
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df.columns = [_to_snake_case(column) for column in df.columns]
    df["target_label"] = df["target"].map({0: "malignant", 1: "benign"})
    output_path = settings.raw_dir / "breast_cancer.csv"
    df.to_csv(output_path, index=False)
    return output_path


def preprocess_data(raw_path: Path) -> dict[str, Path]:
    ensure_directories()
    df = pd.read_csv(raw_path)

    train_df, test_df = train_test_split(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=df["target"],
    )

    train_path = settings.processed_dir / "train.csv"
    test_path = settings.processed_dir / "test.csv"
    reference_path = settings.reference_dir / "reference_features.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_df.drop(columns=["target", "target_label"]).to_csv(reference_path, index=False)

    return {
        "train": train_path,
        "test": test_path,
        "reference": reference_path,
    }
