"""Shared fixtures for Design A tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_screen_df():
    """Synthetic screen DataFrame with 50 genes."""
    rng = np.random.default_rng(42)
    n = 50
    cs_values = rng.normal(0, 2, size=n)
    symbols = [f"GENE{i:03d}" for i in range(n)]
    entrez_ids = list(range(1000, 1000 + n))
    return pd.DataFrame({
        "gene_symbol": symbols,
        "entrez_id": entrez_ids,
        "cs": cs_values,
        "pvalue": rng.uniform(0, 1, size=n),
    })


@pytest.fixture
def tiny_screen_normalized(tiny_screen_df):
    """Screen DataFrame with z-score normalization."""
    from crispr_al.screen import zscore_normalize, assign_hit_labels
    df = zscore_normalize(tiny_screen_df)
    df = assign_hit_labels(df)
    return df


@pytest.fixture
def tiny_features_df(tiny_screen_df):
    """Synthetic gene features DataFrame (9 columns)."""
    rng = np.random.default_rng(42)
    genes = tiny_screen_df["gene_symbol"].tolist()
    n = len(genes)
    return pd.DataFrame({
        "molm13_log_tpm": rng.uniform(0, 10, n),
        "coessential_mean_r_top50": rng.uniform(-0.5, 0.9, n),
        "coessential_molm13_chronos": rng.normal(0, 0.5, n),
        "n_reactome_pathways": rng.integers(0, 50, n),
        "n_go_bp_terms": rng.integers(0, 100, n),
        "n_go_mf_terms": rng.integers(0, 30, n),
        "in_hallmark_apoptosis": rng.integers(0, 2, n),
        "in_hallmark_oxidative_phosphorylation": rng.integers(0, 2, n),
        "n_kegg_pathways": rng.integers(0, 30, n),
    }, index=pd.Index(genes, name="gene_symbol")).astype(float)
