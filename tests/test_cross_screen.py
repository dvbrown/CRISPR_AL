"""Tests for Design B cross-screen split generation and end-to-end pipeline."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from crispr_al.splits import (
    XFER_SEED_START,
    XFER_SEED_START_REVERSE,
    compute_overlap_baseline_hash,
    compute_split_hash,
    generate_cross_screen_splits,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chen_genes():
    """Synthetic Chen gene list: 60 genes."""
    return [f"CHEN{i:03d}" for i in range(60)]


@pytest.fixture
def sharon_genes(chen_genes):
    """Synthetic Sharon gene list: 50 genes, 40 shared with Chen + 10 Sharon-only."""
    shared = chen_genes[:40]
    sharon_only = [f"SHARON{i:03d}" for i in range(10)]
    return shared + sharon_only


@pytest.fixture
def fwd_splits(chen_genes, sharon_genes):
    """Chen → Sharon, 3 repeats, train_size=10."""
    return generate_cross_screen_splits(
        train_genes=chen_genes,
        test_genes=sharon_genes,
        train_screen_id="chen2019_1393",
        test_screen_id="sharon2019_1402",
        n_repeats=3,
        train_size=10,
        seed_start=XFER_SEED_START,
    )


@pytest.fixture
def rev_splits(chen_genes, sharon_genes):
    """Sharon → Chen, 3 repeats, train_size=10."""
    return generate_cross_screen_splits(
        train_genes=sharon_genes,
        test_genes=chen_genes,
        train_screen_id="sharon2019_1402",
        test_screen_id="chen2019_1393",
        n_repeats=3,
        train_size=10,
        seed_start=XFER_SEED_START_REVERSE,
    )


# ---------------------------------------------------------------------------
# Test 1: Split count correct
# ---------------------------------------------------------------------------

def test_split_count(fwd_splits):
    assert len(fwd_splits) == 3


# ---------------------------------------------------------------------------
# Test 2: No gene overlap between train and test in every split
# ---------------------------------------------------------------------------

def test_no_gene_overlap(fwd_splits):
    for split in fwd_splits:
        train_set = set(split["train_genes"])
        test_set = set(split["test_genes"])
        assert train_set & test_set == set(), (
            f"Train/test overlap in {split['split_id']}"
        )


# ---------------------------------------------------------------------------
# Test 3: Test genes come from the test screen's gene list
# ---------------------------------------------------------------------------

def test_test_genes_from_test_screen(fwd_splits, sharon_genes):
    sharon_set = set(sharon_genes)
    for split in fwd_splits:
        assert set(split["test_genes"]) <= sharon_set, (
            "Test genes must be a subset of Sharon gene list"
        )


# ---------------------------------------------------------------------------
# Test 4: Train genes come from the train screen's gene list
# ---------------------------------------------------------------------------

def test_train_genes_from_train_screen(fwd_splits, chen_genes):
    chen_set = set(chen_genes)
    for split in fwd_splits:
        assert set(split["train_genes"]) <= chen_set, (
            "Train genes must be a subset of Chen gene list"
        )


# ---------------------------------------------------------------------------
# Test 5: Deterministic regeneration with same seeds
# ---------------------------------------------------------------------------

def test_deterministic_regeneration(chen_genes, sharon_genes):
    kwargs = dict(
        train_genes=chen_genes,
        test_genes=sharon_genes,
        train_screen_id="chen2019_1393",
        test_screen_id="sharon2019_1402",
        n_repeats=4,
        train_size=10,
        seed_start=XFER_SEED_START,
    )
    splits_a = generate_cross_screen_splits(**kwargs)
    splits_b = generate_cross_screen_splits(**kwargs)
    for s_a, s_b in zip(splits_a, splits_b):
        assert s_a["train_genes"] == s_b["train_genes"]
        assert s_a["test_genes"] == s_b["test_genes"]
        assert s_a["split_hash"] == s_b["split_hash"]


# ---------------------------------------------------------------------------
# Test 6: Unique split hashes across all repeats
# ---------------------------------------------------------------------------

def test_unique_split_hashes(chen_genes, sharon_genes):
    splits = generate_cross_screen_splits(
        train_genes=chen_genes,
        test_genes=sharon_genes,
        train_screen_id="chen2019_1393",
        test_screen_id="sharon2019_1402",
        n_repeats=5,
        train_size=10,
        seed_start=XFER_SEED_START,
    )
    hashes = [s["split_hash"] for s in splits]
    assert len(set(hashes)) == len(hashes), "All split hashes must be unique"


# ---------------------------------------------------------------------------
# Test 7: Correct metadata (generator_id, family, aim, screen IDs)
# ---------------------------------------------------------------------------

def test_split_metadata(fwd_splits):
    for split in fwd_splits:
        assert split["generator_id"] == "aim1_cross_screen_transfer"
        assert split["family"] == "context_zeroshot"
        assert split["aim"] == "aim1_venetoclax"
        assert split["metrics_profile"] == "aim1_transfer"
        assert split["train_screen_id"] == "chen2019_1393"
        assert split["test_screen_id"] == "sharon2019_1402"


# ---------------------------------------------------------------------------
# Test 8: Seed ranges start at correct values
# ---------------------------------------------------------------------------

def test_forward_seed_range(fwd_splits):
    for i, split in enumerate(fwd_splits):
        assert split["seed"] == XFER_SEED_START + i, (
            f"Expected seed {XFER_SEED_START + i}, got {split['seed']}"
        )


def test_reverse_seed_range(rev_splits):
    for i, split in enumerate(rev_splits):
        assert split["seed"] == XFER_SEED_START_REVERSE + i, (
            f"Expected seed {XFER_SEED_START_REVERSE + i}, got {split['seed']}"
        )


def test_xfer_seed_start_value():
    assert XFER_SEED_START == 21001


def test_xfer_seed_start_reverse_value():
    assert XFER_SEED_START_REVERSE == 22001


# ---------------------------------------------------------------------------
# Test 9–12: Sharon screen loading / zscore / hit labels / duplicate handling
#   (covered comprehensively in test_screen.py — spot-check here in pipeline context)
# ---------------------------------------------------------------------------

def test_sharon_zscore_hit_labels_integrated():
    """z-score hit labels on a synthetic Sharon screen use ±1.645 threshold."""
    from crispr_al.screen import assign_hit_labels_zscore, zscore_normalize
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame({
        "gene_symbol": [f"G{i}" for i in range(n)],
        "lfc": rng.normal(0, 2, n),
    })
    df = zscore_normalize(df, score_col="lfc")
    df = assign_hit_labels_zscore(df)
    assert (df.loc[df["is_hit_sensitizer"], "score_norm"] < -1.645).all()
    assert (df.loc[df["is_hit_resistor"], "score_norm"] > 1.645).all()
    assert not (df["is_hit_sensitizer"] & df["is_hit_resistor"]).any()


# ---------------------------------------------------------------------------
# Test 13: Overlap baseline hash is deterministic and uses correct signature
# ---------------------------------------------------------------------------

def test_overlap_baseline_hash_deterministic(chen_genes, sharon_genes):
    shared = sorted(set(chen_genes) & set(sharon_genes))
    h1 = compute_overlap_baseline_hash(
        "aim1_overlap_baseline", "chen2019_1393", "sharon2019_1402", shared
    )
    h2 = compute_overlap_baseline_hash(
        "aim1_overlap_baseline", "chen2019_1393", "sharon2019_1402", shared
    )
    assert h1 == h2


def test_overlap_baseline_hash_direction_differs(chen_genes, sharon_genes):
    shared = sorted(set(chen_genes) & set(sharon_genes))
    h_fwd = compute_overlap_baseline_hash(
        "aim1_overlap_baseline", "chen2019_1393", "sharon2019_1402", shared
    )
    h_rev = compute_overlap_baseline_hash(
        "aim1_overlap_baseline", "sharon2019_1402", "chen2019_1393", shared
    )
    assert h_fwd != h_rev, "Hash must differ by direction"


def test_overlap_baseline_hash_distinct_from_split_hash(chen_genes, sharon_genes):
    shared = sorted(set(chen_genes) & set(sharon_genes))
    overlap_hash = compute_overlap_baseline_hash(
        "aim1_overlap_baseline", "chen2019_1393", "sharon2019_1402", shared
    )
    split_hash = compute_split_hash(
        "aim1_cross_screen_transfer", "chen2019_1393_to_sharon2019_1402", 21001, shared
    )
    assert overlap_hash != split_hash


# ---------------------------------------------------------------------------
# Test 16: End-to-end integration — mini screens, 2 splits, Ridge, schema validate
# ---------------------------------------------------------------------------

def test_end_to_end_mini(tmp_path):
    """End-to-end: synthetic mini screens, 2 cross-screen splits, Ridge, schema."""
    schema_path = (
        "/vast/projects/G000448_Protein_Design/Repos/CRISPR_AL/"
        "notebooks/crispr_screen_transfer/metrics.schema.json"
    )
    if not os.path.exists(schema_path):
        pytest.skip("Schema file not found")

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from crispr_al.metrics import (
        build_metrics_record,
        compute_classification_metrics,
        compute_ranking_metrics,
        compute_regression_metrics,
        validate_metrics_record,
    )
    from crispr_al.screen import assign_hit_labels_zscore, zscore_normalize
    from crispr_al.splits import generate_cross_screen_splits

    rng = np.random.default_rng(0)
    n_chen, n_sharon, n_shared = 60, 50, 40
    n_features = 9

    chen_genes = [f"G{i:03d}" for i in range(n_chen)]
    sharon_genes = chen_genes[:n_shared] + [f"SH{i:03d}" for i in range(n_sharon - n_shared)]
    sharon_only = set(sharon_genes) - set(chen_genes)

    # Synthetic scores
    def make_screen(genes, score_col):
        return pd.DataFrame({
            "gene_symbol": genes,
            score_col: rng.normal(0, 2, len(genes)),
        })

    chen_df = zscore_normalize(make_screen(chen_genes, "cs"), score_col="cs")
    chen_df = assign_hit_labels_zscore(chen_df)
    sharon_df = zscore_normalize(make_screen(sharon_genes, "lfc"), score_col="lfc")
    sharon_df = assign_hit_labels_zscore(sharon_df)

    # Synthetic feature matrix for Chen genes; Sharon-only genes → 0.0
    feat_index = pd.Index(chen_genes, name="gene_symbol")
    gene_features = pd.DataFrame(rng.normal(0, 1, (n_chen, n_features)), index=feat_index)
    gene_features = gene_features.reindex(sharon_genes, fill_value=0.0)

    # 2 splits, train_size=10
    splits = generate_cross_screen_splits(
        train_genes=chen_genes,
        test_genes=sharon_genes,
        train_screen_id="chen2019_1393",
        test_screen_id="sharon2019_1402",
        n_repeats=2,
        train_size=10,
        seed_start=XFER_SEED_START,
    )

    sharon_scores = sharon_df.set_index("gene_symbol")

    for split in splits:
        train_genes = split["train_genes"]
        test_genes = split["test_genes"]
        assert set(train_genes) & set(test_genes) == set()

        X_train = gene_features.reindex(train_genes, fill_value=0.0).values
        # Synthetic E2E: Sharon scores are used as labels for both train and test
        # genes (train_genes come from Chen, but Sharon scores exist for the 40
        # shared genes). This intentional simplification keeps the fixture small
        # and tests pipeline mechanics, not real cross-screen semantics.
        y_train = sharon_scores.reindex(train_genes)["score_norm"].fillna(0.0).values
        X_test = gene_features.reindex(test_genes, fill_value=0.0).values
        y_test = sharon_scores.reindex(test_genes)["score_norm"].fillna(0.0).values
        hit_sens = sharon_scores.reindex(test_genes)["is_hit_sensitizer"].fillna(False).values
        hit_res = sharon_scores.reindex(test_genes)["is_hit_resistor"].fillna(False).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        n_zero_imputed = sum(g in sharon_only for g in test_genes)
        K_VALUES = [5, 10, 20, 30]
        regression = compute_regression_metrics(y_test, y_pred)
        ranking = compute_ranking_metrics(y_pred, hit_sens, hit_res, k_values=K_VALUES)
        classification = compute_classification_metrics(y_pred, hit_sens, hit_res)

        # Verify effective n is capped at n_test when K > n_test
        n_test = len(test_genes)
        for row in ranking["k_metrics"]:
            assert row["n"] == min(row["k"], n_test)

        data_counts = {
            "train_row_count": len(train_genes),
            "test_row_count": len(test_genes),
            "n_unique_train_genes": len(train_genes),
            "n_unique_test_genes": len(test_genes),
            "n_overlap_genes_train_test": 0,
            "n_test_zero_imputed_features": n_zero_imputed,
        }
        leakage_checks = {
            "disjoint_gene_label_rows": True,
            "normalization_fit_on_train_only": True,
            "split_hash_logged": True,
        }
        record = build_metrics_record(
            split, data_counts, leakage_checks, regression, ranking, classification,
            run_id=f"{split['split_id']}_ridge",
            timestamp_utc="2026-03-13T00:00:00Z",
            code_commit="abcdef1",
            notes=f"ridge_alpha={model.alpha_}",
        )
        validate_metrics_record(record, schema_path)
