"""Tests for crispr_al.metrics module."""
import numpy as np
import pandas as pd
import pytest
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
    build_metrics_record,
    validate_metrics_record,
    make_negative_control,
    make_positive_control_cross_screen,
    compute_drf,
    compute_calibration_report_with_hits,
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


def test_ranking_metrics_n_field(random_pred):
    """n field (effective evaluated count) is present in every k_metric_row."""
    y_test, y_pred, hit_sens, hit_res = random_pred
    result = compute_ranking_metrics(y_pred, hit_sens, hit_res)
    for row in result["k_metrics"]:
        assert "n" in row, "n field must be present"
        assert row["n"] <= row["k"]
        assert row["n"] >= 1


def test_ranking_metrics_k_greater_than_n_eval():
    """When K > n_eval, effective n is capped at n_eval."""
    rng = np.random.default_rng(5)
    n_eval = 30   # smaller than any K in K_VALUES
    y_pred = rng.normal(0, 1, n_eval)
    hit_sens = y_pred < -0.5
    hit_res = y_pred > 0.5
    result = compute_ranking_metrics(y_pred, hit_sens, hit_res, k_values=[50, 100])
    for row in result["k_metrics"]:
        assert row["n"] == n_eval, f"Expected n={n_eval}, got {row['n']}"
        assert row["precision_at_k"] == pytest.approx(
            hit_sens[np.argsort(y_pred)[:n_eval]].sum() / n_eval
        )


def test_build_metrics_record_n_in_schema_output():
    """build_metrics_record preserves n field in ranking k_metrics output."""
    split = {
        "split_id": "xfer_001",
        "generator_id": "aim1_cross_screen_transfer",
        "family": "context_zeroshot",
        "aim": "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed": 21001,
        "repeat_index": 1,
        "train_screen_id": "chen2019_1393",
        "test_screen_id": "sharon2019_1402",
        "split_hash": "abcd1234ef567890",
    }
    data_counts = {
        "train_row_count": 2000,
        "test_row_count": 15400,
        "n_unique_train_genes": 2000,
        "n_unique_test_genes": 15400,
        "n_overlap_genes_train_test": 0,
        "n_test_zero_imputed_features": 139,
    }
    leakage_checks = {
        "disjoint_gene_label_rows": True,
        "normalization_fit_on_train_only": True,
        "split_hash_logged": True,
    }
    regression = {"pearson": 0.3, "spearman": 0.28, "r2": 0.09, "rmse": 1.1, "mae": 0.85}
    ranking = {"k_metrics": [
        {"k": 50, "n": 50, "precision_at_k": 0.2, "recall_at_k": 0.1,
         "precision_at_k_resistor": 0.1, "recall_at_k_resistor": 0.05},
    ]}
    classification = {"labels": [
        {"label": "sensitizer", "auroc": 0.65, "auprc": 0.15, "positive_rate": 0.05},
        {"label": "resistor",   "auroc": 0.60, "auprc": 0.10, "positive_rate": 0.05},
    ]}
    record = build_metrics_record(
        split, data_counts, leakage_checks, regression, ranking, classification,
        run_id="xfer_001_ridge",
        timestamp_utc="2026-03-13T00:00:00Z",
        code_commit="abcdef1",
    )
    # n should be preserved; internal resistor keys should be stripped
    row = record["metrics"]["ranking"]["k_metrics"][0]
    assert "n" in row
    assert row["n"] == 50
    assert "precision_at_k_resistor" not in row


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


def test_validate_context_zeroshot_schema():
    """context_zeroshot family with cross-screen IDs passes schema validation."""
    schema_path = "/vast/projects/G000448_Protein_Design/Repos/CRISPR_AL/notebooks/crispr_screen_transfer/metrics.schema.json"
    if not os.path.exists(schema_path):
        pytest.skip("Schema file not found")

    split = {
        "split_id": "aim1_xfer_chen2019_1393_to_sharon2019_1402_r001",
        "generator_id": "aim1_cross_screen_transfer",
        "family": "context_zeroshot",
        "aim": "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed": 21001,
        "repeat_index": 1,
        "train_screen_id": "chen2019_1393",
        "test_screen_id": "sharon2019_1402",
        "split_hash": "abcd1234ef567890",
    }
    data_counts = {
        "train_row_count": 2000,
        "test_row_count": 15400,
        "n_unique_train_genes": 2000,
        "n_unique_test_genes": 15400,
        "n_overlap_genes_train_test": 0,
        "n_test_zero_imputed_features": 139,
    }
    leakage_checks = {
        "disjoint_gene_label_rows": True,
        "normalization_fit_on_train_only": True,
        "split_hash_logged": True,
    }
    regression = {"pearson": 0.3, "spearman": 0.28, "r2": 0.09, "rmse": 1.1, "mae": 0.85}
    ranking = {"k_metrics": [
        {"k": 50,  "n": 50,  "precision_at_k": 0.2,  "recall_at_k": 0.1},
        {"k": 100, "n": 100, "precision_at_k": 0.18, "recall_at_k": 0.18},
        {"k": 200, "n": 200, "precision_at_k": 0.15, "recall_at_k": 0.3},
        {"k": 500, "n": 500, "precision_at_k": 0.1,  "recall_at_k": 0.5},
    ]}
    classification = {"labels": [
        {"label": "sensitizer", "auroc": 0.65, "auprc": 0.15, "positive_rate": 0.05},
        {"label": "resistor",   "auroc": 0.60, "auprc": 0.10, "positive_rate": 0.05},
    ]}
    record = build_metrics_record(
        split, data_counts, leakage_checks, regression, ranking, classification,
        run_id="aim1_xfer_chen2019_1393_to_sharon2019_1402_r001_ridge",
        timestamp_utc="2026-03-13T00:00:00Z",
        code_commit="abcdef1",
    )
    validate_metrics_record(record, schema_path)


def test_validate_overlap_baseline_schema():
    """Overlap baseline record (seed=0, no repeat_index) passes schema validation."""
    schema_path = "/vast/projects/G000448_Protein_Design/Repos/CRISPR_AL/notebooks/crispr_screen_transfer/metrics.schema.json"
    if not os.path.exists(schema_path):
        pytest.skip("Schema file not found")

    split = {
        "split_id": "aim1_overlap_chen2019_1393_to_sharon2019_1402",
        "generator_id": "aim1_overlap_baseline",
        "family": "context_zeroshot",
        "aim": "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed": 0,
        "train_screen_id": "chen2019_1393",
        "test_screen_id": "sharon2019_1402",
        "split_hash": "abcd1234ef567890",
    }
    data_counts = {
        "train_row_count": 17091,
        "test_row_count": 17091,
        "n_unique_train_genes": 17091,
        "n_unique_test_genes": 17091,
        "n_overlap_genes_train_test": 17091,
    }
    leakage_checks = {
        "disjoint_gene_label_rows": True,
        "normalization_fit_on_train_only": True,
        "split_hash_logged": True,
        "details": "Only precomputed screen-wise harmonization used; no StandardScaler fitted.",
    }
    regression = {"pearson": 0.45, "spearman": 0.42, "r2": 0.2, "rmse": 0.9, "mae": 0.7}
    ranking = {"k_metrics": [
        {"k": 50,  "n": 50,  "precision_at_k": 0.25, "recall_at_k": 0.07},
        {"k": 100, "n": 100, "precision_at_k": 0.22, "recall_at_k": 0.13},
        {"k": 200, "n": 200, "precision_at_k": 0.18, "recall_at_k": 0.21},
        {"k": 500, "n": 500, "precision_at_k": 0.14, "recall_at_k": 0.41},
    ]}
    classification = {"labels": [
        {"label": "sensitizer", "auroc": 0.70, "auprc": 0.20, "positive_rate": 0.05},
        {"label": "resistor",   "auroc": 0.65, "auprc": 0.15, "positive_rate": 0.05},
    ]}
    record = build_metrics_record(
        split, data_counts, leakage_checks, regression, ranking, classification,
        run_id="aim1_overlap_chen2019_1393_to_sharon2019_1402",
        timestamp_utc="2026-03-13T00:00:00Z",
        code_commit="abcdef1",
    )
    validate_metrics_record(record, schema_path)


# ---------------------------------------------------------------------------
# Phase 0: DRF calibration tests
# ---------------------------------------------------------------------------

def test_compute_drf_perfect_predictor():
    """DRF ≈ 1.0 when y_positive_control == y_true."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 300)
    y_neg = make_negative_control(y)
    metric_fn = lambda a, b: float(spearmanr(a, b).statistic)
    drf = compute_drf(y, y, y_neg, metric_fn)
    assert drf == pytest.approx(1.0, abs=1e-3)


def test_compute_drf_null_predictor():
    """DRF ≈ 0.0 when positive control == negative control."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    y_neg = make_negative_control(y)
    metric_fn = lambda a, b: float(spearmanr(a, b).statistic)
    drf = compute_drf(y, y_neg, y_neg, metric_fn)
    assert drf == pytest.approx(0.0, abs=1e-3)


def test_compute_drf_spearman_fitness_screen():
    """DRF > 0.1 when positive control has low noise."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 300)
    y_pos = y + rng.normal(0, 0.3, 300)
    y_neg = make_negative_control(y)
    metric_fn = lambda a, b: float(spearmanr(a, b).statistic)
    drf = compute_drf(y, y_pos, y_neg, metric_fn)
    assert drf > 0.1


def test_make_negative_control():
    """Negative control is a constant array equal to mean."""
    rng = np.random.default_rng(3)
    y = rng.normal(5, 2, 200)
    ctrl = make_negative_control(y)
    assert ctrl.shape == y.shape
    assert np.all(ctrl == pytest.approx(np.mean(y)))


def test_make_positive_control_cross_screen():
    """Returns only shared genes aligned correctly."""
    chen = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    sharon = pd.Series({"B": 0.5, "C": 0.6, "D": 0.7, "E": 0.8})
    y_true, y_pred = make_positive_control_cross_screen(chen, sharon)
    assert len(y_true) == 3   # B, C, D shared
    assert len(y_pred) == 3
    # Values correspond to sharon (y_true) and chen (y_pred) in sorted order
    shared_sorted = sorted(["B", "C", "D"])
    np.testing.assert_array_almost_equal(y_true, sharon[shared_sorted].values)
    np.testing.assert_array_almost_equal(y_pred, chen[shared_sorted].values)


def test_calibration_report_with_hits_keys():
    """All 8 drf_* keys are present in the calibration report."""
    rng = np.random.default_rng(4)
    n = 500
    y = rng.normal(0, 1, n)
    y_pos = y + rng.normal(0, 0.3, n)
    y_neg = make_negative_control(y)
    hit_sens = y < -1.0
    hit_res = y > 1.5
    report = compute_calibration_report_with_hits(y, y_pos, y_neg, hit_sens, hit_res)
    expected_drf_keys = {
        "drf_spearman", "drf_pearson", "drf_neg_rmse",
        "drf_auroc_sensitizer",
        "drf_precision_at_50", "drf_precision_at_100",
        "drf_precision_at_200", "drf_precision_at_500",
    }
    assert expected_drf_keys.issubset(report.keys())
