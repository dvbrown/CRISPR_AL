"""Tests for crispr_al.metrics module."""
import numpy as np
import pytest
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
    build_metrics_record,
    validate_metrics_record,
    K_VALUES,
)
import os


@pytest.fixture
def perfect_pred():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    return y, y.copy()


@pytest.fixture
def random_pred():
    rng = np.random.default_rng(1)
    y_test = rng.normal(0, 1, 500)
    y_pred = rng.normal(0, 1, 500)
    hit_sens = y_test < -1.0
    hit_res = y_test > 1.5
    return y_test, y_pred, hit_sens, hit_res


def test_regression_metrics_perfect(perfect_pred):
    y_test, y_pred = perfect_pred
    metrics = compute_regression_metrics(y_test, y_pred)
    assert metrics["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_regression_metrics_keys(random_pred):
    y_test, y_pred, _, _ = random_pred
    metrics = compute_regression_metrics(y_test, y_pred)
    assert set(metrics.keys()) == {"pearson", "spearman", "r2", "rmse", "mae"}


def test_regression_metrics_bounds(random_pred):
    y_test, y_pred, _, _ = random_pred
    metrics = compute_regression_metrics(y_test, y_pred)
    assert -1.0 <= metrics["pearson"] <= 1.0
    assert -1.0 <= metrics["spearman"] <= 1.0
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_ranking_metrics_structure(random_pred):
    y_test, y_pred, hit_sens, hit_res = random_pred
    result = compute_ranking_metrics(y_pred, hit_sens, hit_res)
    assert "k_metrics" in result
    assert len(result["k_metrics"]) == len(K_VALUES)
    for row in result["k_metrics"]:
        assert "k" in row
        assert "precision_at_k" in row
        assert "recall_at_k" in row
        assert 0.0 <= row["precision_at_k"] <= 1.0
        assert 0.0 <= row["recall_at_k"] <= 1.0


def test_classification_metrics_structure(random_pred):
    y_test, y_pred, hit_sens, hit_res = random_pred
    result = compute_classification_metrics(y_pred, hit_sens, hit_res)
    assert "labels" in result
    assert len(result["labels"]) == 2
    labels = {r["label"] for r in result["labels"]}
    assert labels == {"sensitizer", "resistor"}
    for row in result["labels"]:
        assert 0.0 <= row["auroc"] <= 1.0
        assert 0.0 <= row["auprc"] <= 1.0


def test_build_metrics_record_keys():
    split = {
        "split_id": "test_001",
        "generator_id": "aim1_random_gene_holdout",
        "family": "random_gene_holdout",
        "aim": "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed": 11001,
        "repeat_index": 1,
        "train_screen_id": "chen2019_1393",
        "test_screen_id": "chen2019_1393",
        "split_hash": "abcd1234ef567890",
    }
    data_counts = {
        "train_row_count": 2000,
        "test_row_count": 17000,
        "n_unique_train_genes": 2000,
        "n_unique_test_genes": 17000,
        "n_overlap_genes_train_test": 0,
    }
    leakage_checks = {
        "disjoint_gene_label_rows": True,
        "normalization_fit_on_train_only": True,
        "split_hash_logged": True,
    }
    regression = {"pearson": 0.5, "spearman": 0.4, "r2": 0.2, "rmse": 0.9, "mae": 0.7}
    ranking = {"k_metrics": [{"k": 50, "precision_at_k": 0.3, "recall_at_k": 0.1}]}
    classification = {"labels": [
        {"label": "sensitizer", "auroc": 0.7, "auprc": 0.2, "positive_rate": 0.05},
        {"label": "resistor", "auroc": 0.6, "auprc": 0.1, "positive_rate": 0.03},
    ]}
    record = build_metrics_record(
        split, data_counts, leakage_checks, regression, ranking, classification,
        run_id="test_001_ridge",
        timestamp_utc="2026-03-13T00:00:00Z",
        code_commit="abcdef1",
    )
    assert record["schema_version"] == "1.0.0"
    assert record["split"]["split_hash"] == "abcd1234ef567890"


def test_validate_metrics_record(tmp_path):
    """Test that a valid record passes schema validation."""
    import json
    schema_path = "/vast/projects/G000448_Protein_Design/Repos/CRISPR_AL/notebooks/crispr_screen_transfer/metrics.schema.json"
    if not os.path.exists(schema_path):
        pytest.skip("Schema file not found")

    split = {
        "split_id": "aim1_random_chen2019_1393_r001",
        "generator_id": "aim1_random_gene_holdout",
        "family": "random_gene_holdout",
        "aim": "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed": 11001,
        "repeat_index": 1,
        "train_screen_id": "chen2019_1393",
        "test_screen_id": "chen2019_1393",
        "split_hash": "abcd1234ef567890",
    }
    data_counts = {
        "train_row_count": 2000,
        "test_row_count": 17000,
        "n_unique_train_genes": 2000,
        "n_unique_test_genes": 17000,
        "n_overlap_genes_train_test": 0,
    }
    leakage_checks = {
        "disjoint_gene_label_rows": True,
        "normalization_fit_on_train_only": True,
        "split_hash_logged": True,
    }
    regression = {"pearson": 0.5, "spearman": 0.4, "r2": 0.2, "rmse": 0.9, "mae": 0.7}
    ranking = {"k_metrics": [
        {"k": 50, "precision_at_k": 0.3, "recall_at_k": 0.1},
        {"k": 100, "precision_at_k": 0.25, "recall_at_k": 0.15},
        {"k": 200, "precision_at_k": 0.2, "recall_at_k": 0.2},
        {"k": 500, "precision_at_k": 0.15, "recall_at_k": 0.3},
    ]}
    classification = {"labels": [
        {"label": "sensitizer", "auroc": 0.7, "auprc": 0.2, "positive_rate": 0.05},
        {"label": "resistor", "auroc": 0.6, "auprc": 0.1, "positive_rate": 0.03},
    ]}
    record = build_metrics_record(
        split, data_counts, leakage_checks, regression, ranking, classification,
        run_id="aim1_random_chen2019_1393_r001_ridge",
        timestamp_utc="2026-03-13T00:00:00Z",
        code_commit="abcdef1",
    )
    validate_metrics_record(record, schema_path)
