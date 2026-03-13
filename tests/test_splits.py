"""Tests for crispr_al.splits module."""
import pytest
from crispr_al.splits import generate_splits, compute_split_hash, SEED_START, SCREEN_ID


@pytest.fixture
def all_genes():
    return [f"GENE{i:04d}" for i in range(5000)]


def test_generate_splits_count(all_genes):
    splits = generate_splits(all_genes, n_repeats=3, train_size=100)
    assert len(splits) == 3


def test_generate_splits_train_size(all_genes):
    splits = generate_splits(all_genes, n_repeats=3, train_size=100)
    for split in splits:
        assert len(split["train_genes"]) == 100


def test_generate_splits_no_overlap(all_genes):
    splits = generate_splits(all_genes, n_repeats=3, train_size=100)
    for split in splits:
        train_set = set(split["train_genes"])
        test_set = set(split["test_genes"])
        assert len(train_set & test_set) == 0, "Train and test must be disjoint"


def test_generate_splits_complement(all_genes):
    splits = generate_splits(all_genes, n_repeats=2, train_size=100)
    for split in splits:
        combined = set(split["train_genes"]) | set(split["test_genes"])
        assert combined == set(all_genes), "Train + test should cover all genes"


def test_generate_splits_deterministic(all_genes):
    splits1 = generate_splits(all_genes, n_repeats=5, train_size=100)
    splits2 = generate_splits(all_genes, n_repeats=5, train_size=100)
    for s1, s2 in zip(splits1, splits2):
        assert s1["train_genes"] == s2["train_genes"]
        assert s1["split_hash"] == s2["split_hash"]


def test_generate_splits_unique_hashes(all_genes):
    splits = generate_splits(all_genes, n_repeats=10, train_size=100)
    hashes = [s["split_hash"] for s in splits]
    assert len(set(hashes)) == 10, "All split hashes must be unique"


def test_generate_splits_seed_increment(all_genes):
    splits = generate_splits(all_genes, n_repeats=3, train_size=100)
    for i, split in enumerate(splits):
        assert split["seed"] == SEED_START + i


def test_generate_splits_metadata(all_genes):
    splits = generate_splits(all_genes, n_repeats=2, train_size=100)
    for i, split in enumerate(splits):
        assert split["generator_id"] == "aim1_random_gene_holdout"
        assert split["family"] == "random_gene_holdout"
        assert split["aim"] == "aim1_venetoclax"
        assert split["metrics_profile"] == "aim1_transfer"
        assert split["repeat_index"] == i + 1
        assert split["train_screen_id"] == SCREEN_ID
        assert split["test_screen_id"] == SCREEN_ID


def test_compute_split_hash_stability(all_genes):
    splits = generate_splits(all_genes, n_repeats=1, train_size=100)
    split = splits[0]
    recomputed = compute_split_hash(
        split["generator_id"],
        split["train_screen_id"],
        split["seed"],
        split["train_genes"],
    )
    assert recomputed == split["split_hash"]
