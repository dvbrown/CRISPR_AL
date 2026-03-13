"""Metric computation for Design A."""
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score

K_VALUES = [50, 100, 200, 500]


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
    """
    if k_values is None:
        k_values = K_VALUES

    sens_order = np.argsort(y_pred)          # ascending: most negative first
    res_order = sens_order[::-1]              # descending: most positive first

    n_sensitizers = int(hit_sensitizer.sum())
    n_resistors = int(hit_resistor.sum())

    k_metrics = []
    for k in k_values:
        n_correct_sens = int(hit_sensitizer[sens_order[:k]].sum())
        n_correct_res = int(hit_resistor[res_order[:k]].sum())
        k_metrics.append({
            "k": k,
            "precision_at_k": n_correct_sens / k,
            "recall_at_k": n_correct_sens / max(n_sensitizers, 1),
            "precision_at_k_resistor": n_correct_res / k,
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
) -> dict:
    """Assemble a complete metrics record matching metrics.schema.json."""
    _SPLIT_KEYS = [
        "split_id", "generator_id", "family", "aim", "metrics_profile",
        "seed", "repeat_index", "train_screen_id", "test_screen_id", "split_hash",
    ]
    # Schema-safe ranking: strip any internal keys not in the schema
    ranking_schema = {"k_metrics": [
        {"k": km["k"], "precision_at_k": km["precision_at_k"], "recall_at_k": km["recall_at_k"]}
        for km in ranking["k_metrics"]
    ]}
    return {
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


def validate_metrics_record(record: dict, schema_path: str) -> None:
    """Validate a metrics record against the JSON schema."""
    import jsonschema
    with open(schema_path) as f:
        schema = json.load(f)
    jsonschema.validate(instance=record, schema=schema)
