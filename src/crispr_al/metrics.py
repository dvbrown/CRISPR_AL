"""Metric computation for Design A and Design B."""
import json
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score

K_VALUES = [50, 100, 200, 500]

# Keys serialised from the split dict into every metrics JSON.
# Must stay aligned with metrics.schema.json $defs.split.properties.
_SPLIT_KEYS = [
    "split_id", "generator_id", "family", "aim", "metrics_profile",
    "seed", "repeat_index", "train_screen_id", "test_screen_id", "split_hash",
]

# Schema-safe keys forwarded from internal k_metrics rows to JSON output.
# Internal keys (e.g. precision_at_k_resistor) are stripped here and
# preserved only in aggregated CSVs.
_RANKING_KEYS = ("k", "n", "precision_at_k", "recall_at_k")


def compute_regression_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    pearson = float(pearsonr(y_test, y_pred).statistic)
    spearman = float(spearmanr(y_test, y_pred).statistic)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_test, y_pred))
    return {"pearson": pearson, "spearman": spearman, "r2": r2, "rmse": rmse, "mae": mae}


def compute_ranking_metrics(
    y_pred: np.ndarray,
    hit_sensitizer: np.ndarray,
    hit_resistor: np.ndarray,
    k_values: list = None,
) -> dict:
    """Compute Precision@K and Recall@K for sensitizers and resistors.

    hit_sensitizer and hit_resistor are boolean arrays aligned with y_pred.
    Returns schema-compliant dict with sensitizer-based precision/recall at each K,
    plus _resistor sub-keys for downstream aggregation.

    When K > len(y_pred), effective n = len(y_pred). Both k (requested) and
    n (effective) are stored in each row.
    """
    if k_values is None:
        k_values = K_VALUES

    sens_order = np.argsort(y_pred)          # ascending: most negative first
    res_order = sens_order[::-1]              # descending: most positive first

    n_sensitizers = int(hit_sensitizer.sum())
    n_resistors = int(hit_resistor.sum())

    n_eval = len(y_pred)
    k_metrics = []
    for k in k_values:
        n = min(k, n_eval)
        n_correct_sens = int(hit_sensitizer[sens_order[:n]].sum())
        n_correct_res = int(hit_resistor[res_order[:n]].sum())
        k_metrics.append({
            "k": k,
            "n": n,
            "precision_at_k": n_correct_sens / n,
            "recall_at_k": n_correct_sens / max(n_sensitizers, 1),
            "precision_at_k_resistor": n_correct_res / n,
            "recall_at_k_resistor": n_correct_res / max(n_resistors, 1),
        })

    return {"k_metrics": k_metrics}


def compute_classification_metrics(
    y_pred: np.ndarray,
    hit_sensitizer: np.ndarray,
    hit_resistor: np.ndarray,
) -> dict:
    """Compute AUROC and AUPRC for sensitizer and resistor classification."""
    labels = []
    for label, hit, score in [
        ("sensitizer", hit_sensitizer, -y_pred),  # negative pred → sensitizer
        ("resistor",   hit_resistor,   +y_pred),
    ]:
        hit_int = hit.astype(int)
        if 0 < hit.sum() < len(hit):
            auroc = float(roc_auc_score(hit_int, score))
            auprc = float(average_precision_score(hit_int, score))
        else:
            auroc = 0.5
            auprc = float(hit.mean())
        labels.append({
            "label": label,
            "auroc": auroc,
            "auprc": auprc,
            "positive_rate": float(hit.mean()),
        })
    return {"labels": labels}


def build_metrics_record(
    split: dict,
    data_counts: dict,
    leakage_checks: dict,
    regression: dict,
    ranking: dict,
    classification: dict,
    run_id: str,
    timestamp_utc: str,
    code_commit: str,
    notes: Optional[str] = None,
) -> dict:
    """Assemble a complete metrics record matching metrics.schema.json."""
    ranking_schema = {"k_metrics": [
        {key: km[key] for key in _RANKING_KEYS if key in km}
        for km in ranking["k_metrics"]
    ]}
    record = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "code_commit": code_commit,
        "split": {k: split[k] for k in _SPLIT_KEYS if k in split},
        "data_counts": data_counts,
        "leakage_checks": leakage_checks,
        "metrics": {
            "regression": regression,
            "ranking": ranking_schema,
            "classification": classification,
        },
    }
    if notes is not None:
        record["notes"] = notes
    return record


def flatten_metrics_row(split: dict, reg: dict, rank: dict, clf: dict) -> dict:
    """Flatten per-split metrics into a single CSV-friendly dict.

    Includes split metadata (split_id, seed, repeat_index), all regression
    metrics, precision/recall at each K for sensitizer and resistor, and
    AUROC/AUPRC per label. Safe to call with or without repeat_index.
    """
    row: dict = {
        "split_id": split["split_id"],
        "seed": split["seed"],
        "repeat_index": split.get("repeat_index"),
    }
    row.update({k: reg[k] for k in ("pearson", "spearman", "r2", "rmse", "mae") if k in reg})
    for km in rank["k_metrics"]:
        k = km["k"]
        row[f"precision_at_{k}"] = km["precision_at_k"]
        row[f"recall_at_{k}"] = km["recall_at_k"]
        if "precision_at_k_resistor" in km:
            row[f"precision_at_{k}_resistor"] = km["precision_at_k_resistor"]
            row[f"recall_at_{k}_resistor"] = km["recall_at_k_resistor"]
    for lbl in clf["labels"]:
        row[f"auroc_{lbl['label']}"] = lbl["auroc"]
        row[f"auprc_{lbl['label']}"] = lbl["auprc"]
    return row


def bootstrap_ci_bca(
    values,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Compute BCa bootstrap confidence interval for the mean.

    Returns (mean, ci_low, ci_high). Interprets the n input values as
    split-stability estimates (resamples across splits, not iid observations).
    BCa corrects for bias and skewness in the bootstrap distribution.
    """
    from scipy.stats import bootstrap as scipy_bootstrap
    values = np.asarray(values, dtype=float)
    result = scipy_bootstrap(
        (values,),
        np.mean,
        n_resamples=n_bootstrap,
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=seed,
    )
    return (
        float(np.mean(values)),
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def validate_metrics_record(record: dict, schema_path: str) -> None:
    """Validate a metrics record against the JSON schema."""
    import jsonschema
    with open(schema_path) as f:
        schema = json.load(f)
    jsonschema.validate(instance=record, schema=schema)
