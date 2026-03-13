"""Split generation for Design A."""
import hashlib
import json
import numpy as np

SEED_START = 11001
SCREEN_ID = "chen2019_1393"


def generate_splits(all_genes: list, n_repeats: int = 25, train_size: int = 2000) -> list:
    """Generate n_repeats random gene holdout splits.

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
            "aim": "aim1_venetoclax",
            "metrics_profile": "aim1_transfer",
            "seed": seed,
            "repeat_index": i + 1,
            "train_screen_id": SCREEN_ID,
            "test_screen_id": SCREEN_ID,
            "split_hash": split_hash,
            "train_genes": sorted(train_genes),
            "test_genes": sorted(test_genes),
        })
    return splits


def compute_split_hash(generator_id: str, screen_id: str, seed: int, train_genes: list) -> str:
    """Compute reproducible SHA-256 hash (first 16 hex chars) for a split."""
    payload = json.dumps({
        "generator_id": generator_id,
        "screen_id": screen_id,
        "seed": seed,
        "train_genes": sorted(train_genes),
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
