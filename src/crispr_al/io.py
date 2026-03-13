"""I/O utilities for Design A."""
import json
from pathlib import Path
import pandas as pd


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file."""
    return pd.read_parquet(path)


def save_metrics_json(record: dict, path: str) -> None:
    """Save a metrics record as JSON, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def load_metrics_json(path: str) -> dict:
    """Load a metrics JSON file."""
    with open(path) as f:
        return json.load(f)
