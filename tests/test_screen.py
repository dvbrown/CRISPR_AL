"""Tests for crispr_al.screen module."""
import io

import numpy as np
import pandas as pd
import pytest

from crispr_al.screen import (
    assign_hit_labels,
    assign_hit_labels_zscore,
    load_screen_scores,
    load_sharon_screen_scores,
    zscore_normalize,
)

SHARON_HEADER = (
    "SCREEN_ID\tIDENTIFIER_ID\tIDENTIFIER_TYPE\tOFFICIAL_SYMBOL\tALIASES\t"
    "ORGANISM_ID\tORGANISM_OFFICIAL\tSCORE.1\tSCORE.2\tSCORE.3\tSCORE.4\tSCORE.5\n"
)


def _make_sharon_file(tmp_path, rows):
    """Write a minimal Sharon-format screen file and return its path."""
    content = SHARON_HEADER + "".join(
        f"1402\t{eid}\tENTREZ_GENE\t{sym}\t\t9606\tHomo sapiens\t"
        f"0.1\t0.05\t0.1\t0.05\t{lfc}\n"
        for sym, eid, lfc in rows
    )
    p = tmp_path / "sharon.tab.txt"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# zscore_normalize
# ---------------------------------------------------------------------------

def test_zscore_normalize_mean_zero(tiny_screen_df):
    df = zscore_normalize(tiny_screen_df)
    assert "score_norm" in df.columns
    assert abs(df["score_norm"].mean()) < 1e-10
    assert abs(df["score_norm"].std() - 1.0) < 0.02


def test_zscore_normalize_preserves_rows(tiny_screen_df):
    df = zscore_normalize(tiny_screen_df)
    assert len(df) == len(tiny_screen_df)


def test_zscore_normalize_lfc_column(tiny_screen_df):
    """score_col parameter should work with 'lfc' for Sharon data."""
    df = tiny_screen_df.rename(columns={"cs": "lfc"})
    df = zscore_normalize(df, score_col="lfc")
    assert "score_norm" in df.columns
    assert abs(df["score_norm"].mean()) < 1e-10


# ---------------------------------------------------------------------------
# assign_hit_labels / assign_hit_labels_zscore
# ---------------------------------------------------------------------------

def test_assign_hit_labels_columns(tiny_screen_normalized):
    assert "is_hit_sensitizer" in tiny_screen_normalized.columns
    assert "is_hit_resistor" in tiny_screen_normalized.columns


def test_assign_hit_labels_zscore_threshold(tiny_screen_normalized):
    """Hit labels must be based on z-score ±1.645 on score_norm."""
    df = tiny_screen_normalized
    assert (df.loc[df["is_hit_sensitizer"], "score_norm"] < -1.645).all()
    assert (df.loc[df["is_hit_resistor"], "score_norm"] > 1.645).all()


def test_assign_hit_labels_no_overlap(tiny_screen_normalized):
    overlap = tiny_screen_normalized["is_hit_sensitizer"] & tiny_screen_normalized["is_hit_resistor"]
    assert not overlap.any(), "No gene should be both sensitizer and resistor"


def test_assign_hit_labels_delegates_to_zscore(tiny_screen_normalized):
    """assign_hit_labels (alias) produces same result as assign_hit_labels_zscore."""
    df1 = assign_hit_labels(tiny_screen_normalized)
    df2 = assign_hit_labels_zscore(tiny_screen_normalized)
    pd.testing.assert_series_equal(df1["is_hit_sensitizer"], df2["is_hit_sensitizer"])
    pd.testing.assert_series_equal(df1["is_hit_resistor"], df2["is_hit_resistor"])


def test_assign_hit_labels_zscore_custom_threshold():
    """Custom threshold is respected."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"score_norm": rng.normal(0, 1, 200)})
    df = assign_hit_labels_zscore(df, threshold=2.0)
    assert (df.loc[df["is_hit_sensitizer"], "score_norm"] < -2.0).all()
    assert (df.loc[df["is_hit_resistor"], "score_norm"] > 2.0).all()


# ---------------------------------------------------------------------------
# load_screen_scores (Chen)
# ---------------------------------------------------------------------------

def test_load_screen_scores_smoke(tmp_path):
    screen_file = tmp_path / "screen.tab.txt"
    content = (
        "SCREEN_ID\tIDENTIFIER_ID\tIDENTIFIER_TYPE\tOFFICIAL_SYMBOL\t"
        "ALIASES\tORGANISM_ID\tORGANISM_OFFICIAL\tSCORE.1\tSCORE.2\n"
        "1393\t596\tENTREZ_GENE\tBCL2\t\t9606\tHomo sapiens\t-2.5\t0.001\n"
        "1393\t598\tENTREZ_GENE\tBCL2L1\t\t9606\tHomo sapiens\t0.5\t0.05\n"
    )
    screen_file.write_text(content)
    df = load_screen_scores(str(screen_file))
    assert len(df) == 2
    assert "gene_symbol" in df.columns
    assert "cs" in df.columns
    assert df.loc[df["gene_symbol"] == "BCL2", "cs"].values[0] == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# load_sharon_screen_scores
# ---------------------------------------------------------------------------

def test_load_sharon_basic(tmp_path):
    path = _make_sharon_file(tmp_path, [
        ("BCL2", 596, -1.5),
        ("TP53", 7157, 0.8),
    ])
    df = load_sharon_screen_scores(path)
    assert len(df) == 2
    assert set(df.columns) >= {"gene_symbol", "entrez_id", "lfc", "neg_fdr", "pos_fdr"}
    assert df.loc[df["gene_symbol"] == "BCL2", "lfc"].values[0] == pytest.approx(-1.5)


def test_load_sharon_drops_empty_symbol(tmp_path):
    path = _make_sharon_file(tmp_path, [
        ("BCL2", 596, -1.5),
        ("", 0, 0.0),        # empty symbol — should be dropped
    ])
    df = load_sharon_screen_scores(path)
    assert len(df) == 1
    assert df["gene_symbol"].iloc[0] == "BCL2"


def test_load_sharon_drops_nonfinite_lfc(tmp_path):
    content = (
        SHARON_HEADER
        + "1402\t596\tENTREZ_GENE\tBCL2\t\t9606\tHomo sapiens\t0.1\t0.05\t0.1\t0.05\t-1.5\n"
        + "1402\t999\tENTREZ_GENE\tBADGENE\t\t9606\tHomo sapiens\t0.1\t0.05\t0.1\t0.05\tinf\n"
    )
    p = tmp_path / "sharon.tab.txt"
    p.write_text(content)
    df = load_sharon_screen_scores(str(p))
    assert len(df) == 1
    assert df["gene_symbol"].iloc[0] == "BCL2"


def test_load_sharon_duplicate_keeps_first(tmp_path):
    """Duplicate gene_symbol rows: first occurrence in file order is kept."""
    path = _make_sharon_file(tmp_path, [
        ("BCL2", 596, -1.5),
        ("BCL2", 596, 9.9),   # duplicate — should be dropped
        ("TP53", 7157, 0.8),
    ])
    df = load_sharon_screen_scores(path)
    assert len(df) == 2
    bcl2_lfc = df.loc[df["gene_symbol"] == "BCL2", "lfc"].values[0]
    assert bcl2_lfc == pytest.approx(-1.5), "First occurrence should be kept"


def test_load_sharon_zscore_on_lfc(tmp_path):
    """zscore_normalize with score_col='lfc' produces correct score_norm."""
    rng = np.random.default_rng(42)
    n = 40
    rows = [(f"G{i:03d}", i, float(v)) for i, v in enumerate(rng.normal(0, 2, n))]
    path = _make_sharon_file(tmp_path, rows)
    df = load_sharon_screen_scores(path)
    df = zscore_normalize(df, score_col="lfc")
    assert abs(df["score_norm"].mean()) < 1e-10
    assert abs(df["score_norm"].std() - 1.0) < 0.02
