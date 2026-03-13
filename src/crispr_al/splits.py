"""Split generation for Design A (within-screen) and Design B (cross-screen)."""
import hashlib
import json

import numpy as np

# Design A constants
SEED_START = 11001
SCREEN_ID = "chen2019_1393"

# Design B constants
XFER_SEED_START = 21001
XFER_SEED_START_REVERSE = 22001

# Shared metadata constants used in split records and schema validation.
_AIM1_VENETOCLAX = "aim1_venetoclax"
_AIM1_TRANSFER = "aim1_transfer"


def _sha256_hex16(payload: dict) -> str:
    """Compute SHA-256 hash (first 16 hex chars) of a JSON-serialised payload."""
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def generate_splits(all_genes: list, n_repeats: int = 25, train_size: int = 2000) -> list:
    """Generate n_repeats random gene holdout splits (Design A, within-screen).

    Returns list of dicts with:
      split_id, generator_id, family, aim, metrics_profile, seed,
      repeat_index, train_screen_id, test_screen_id, split_hash,
      train_genes, test_genes
    """
    splits = []
    for i in range(n_repeats):
        seed = SEED_START + i
        rng = np.random.default_rng(seed)
        train_genes = set(rng.choice(all_genes, size=train_size, replace=False))
        test_genes = [g for g in all_genes if g not in train_genes]

        split_hash = compute_split_hash("aim1_random_gene_holdout", SCREEN_ID, seed, train_genes)

        splits.append({
            "split_id": f"aim1_random_{SCREEN_ID}_r{i+1:03d}",
            "generator_id": "aim1_random_gene_holdout",
            "family": "random_gene_holdout",
            "aim": _AIM1_VENETOCLAX,
            "metrics_profile": _AIM1_TRANSFER,
            "seed": seed,
            "repeat_index": i + 1,
            "train_screen_id": SCREEN_ID,
            "test_screen_id": SCREEN_ID,
            "split_hash": split_hash,
            "train_genes": sorted(train_genes),
            "test_genes": sorted(test_genes),
        })
    return splits


def generate_cross_screen_splits(
    train_genes: list,
    test_genes: list,
    train_screen_id: str,
    test_screen_id: str,
    n_repeats: int = 30,
    train_size: int = 2000,
    seed_start: int = XFER_SEED_START,
) -> list:
    """Generate n_repeats cross-screen transfer splits (Design B).

    Samples train_size genes from train_genes. Test set is all test_genes
    not in the sampled train set.

    Returns list of dicts with:
      split_id, generator_id, family, aim, metrics_profile, seed,
      repeat_index, train_screen_id, test_screen_id, split_hash,
      train_genes, test_genes
    """
    train_genes = list(train_genes)
    test_genes = list(test_genes)
    splits = []
    for i in range(n_repeats):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        sampled_train = set(rng.choice(train_genes, size=train_size, replace=False))
        split_test = [g for g in test_genes if g not in sampled_train]

        assert sampled_train & set(split_test) == set(), (
            f"Leakage detected in split {i+1}: train/test gene overlap"
        )

        split_hash = compute_split_hash(
            "aim1_cross_screen_transfer",
            f"{train_screen_id}_to_{test_screen_id}",
            seed,
            sampled_train,
        )

        splits.append({
            "split_id": f"aim1_xfer_{train_screen_id}_to_{test_screen_id}_r{i+1:03d}",
            "generator_id": "aim1_cross_screen_transfer",
            "family": "context_zeroshot",
            "aim": _AIM1_VENETOCLAX,
            "metrics_profile": _AIM1_TRANSFER,
            "seed": seed,
            "repeat_index": i + 1,
            "train_screen_id": train_screen_id,
            "test_screen_id": test_screen_id,
            "split_hash": split_hash,
            "train_genes": sorted(sampled_train),
            "test_genes": sorted(split_test),
        })
    return splits


def compute_split_hash(generator_id: str, screen_id: str, seed: int, train_genes) -> str:
    """Compute reproducible SHA-256 hash (first 16 hex chars) for a split."""
    return _sha256_hex16({
        "generator_id": generator_id,
        "screen_id": screen_id,
        "seed": seed,
        "train_genes": sorted(train_genes),
    })


def compute_overlap_baseline_hash(
    generator_id: str,
    train_screen_id: str,
    test_screen_id: str,
    shared_genes: list,
) -> str:
    """Compute deterministic SHA-256 hash for the overlap-only baseline record.

    Encodes generator_id, ordered screen IDs, and sorted shared gene list.
    Uses seed=0 by convention.
    """
    return _sha256_hex16({
        "generator_id": generator_id,
        "train_screen_id": train_screen_id,
        "test_screen_id": test_screen_id,
        "seed": 0,
        "shared_genes": sorted(shared_genes),
    })
