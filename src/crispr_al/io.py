"""I/O utilities for CRISPR screen transfer analyses."""
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


def save_split_manifest(splits: list, path: str) -> None:
    """Save split manifest CSV (omits train_genes / test_genes gene lists)."""
    records = [
        {k: v for k, v in s.items() if k not in ("train_genes", "test_genes")}
        for s in splits
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def save_split_files(splits: list, directory: str) -> None:
    """Save each split as a JSON file under directory."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    for s in splits:
        with open(dir_path / f"{s['split_id']}.json", "w") as f:
            json.dump(s, f)


def get_code_commit(cwd: str = None) -> str:
    """Return the current git commit short hash, or 'unknown' if unavailable."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
