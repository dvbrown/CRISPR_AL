"""Shared utilities for Olivieri 2020 benchmark scripts."""
import numpy as np

from crispr_al.metrics import (
    compute_classification_metrics,
    compute_ranking_metrics,
    compute_regression_metrics,
)

HIT_THRESHOLD = -3.0  # DrugZ NormZ; negative = sensitising KO
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]

FEATURE_COLS = [
    "n_reactome_pathways",
    "n_kegg_pathways",
    "n_go_bp_terms",
    "n_go_mf_terms",
    "in_hallmark_apoptosis",
    "in_hallmark_oxidative_phosphorylation",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression, ranking, and classification metrics for NormZ predictions.

    Hit threshold: NormZ < -3.0 (sensitising KO).
    Returns flat dict suitable for building a results DataFrame row.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    hit_sens = y_true < HIT_THRESHOLD
    hit_res = np.zeros(len(y_true), dtype=bool)

    reg = compute_regression_metrics(y_true, y_pred)
    rank = compute_ranking_metrics(y_pred, hit_sens, hit_res, k_values=[50, 100])
    clf = compute_classification_metrics(y_pred, hit_sens, hit_res)

    clf_sens = next((l for l in clf["labels"] if l["label"] == "sensitizer"), None)
    if clf_sens is None:
        clf_sens = {"auroc": float("nan"), "auprc": float("nan")}
    prec_by_k = {r["k"]: r["precision_at_k"] for r in rank["k_metrics"]}

    return {
        **{k: reg[k] for k in ("pearson", "spearman", "r2", "rmse")},
        "auroc": clf_sens["auroc"],
        "auprc": clf_sens["auprc"],
        "precision_at_50": prec_by_k.get(50, float("nan")),
        "precision_at_100": prec_by_k.get(100, float("nan")),
        "n_test": len(y_true),
        "n_hits_test": int(hit_sens.sum()),
    }
