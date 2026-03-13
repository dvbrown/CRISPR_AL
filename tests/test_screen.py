"""Tests for crispr_al.screen module."""
import numpy as np
import pandas as pd
import pytest
from crispr_al.screen import load_screen_scores, zscore_normalize, assign_hit_labels


def test_zscore_normalize_mean_zero(tiny_screen_df):
    df = zscore_normalize(tiny_screen_df)
    assert "score_norm" in df.columns
    assert abs(df["score_norm"].mean()) < 1e-10
    assert abs(df["score_norm"].std() - 1.0) < 0.02


def test_zscore_normalize_preserves_rows(tiny_screen_df):
    df = zscore_normalize(tiny_screen_df)
    assert len(df) == len(tiny_screen_df)


def test_assign_hit_labels_columns(tiny_screen_df):
    df = assign_hit_labels(tiny_screen_df)
    assert "is_hit_sensitizer" in df.columns
    assert "is_hit_resistor" in df.columns


def test_assign_hit_labels_thresholds(tiny_screen_df):
    df = assign_hit_labels(tiny_screen_df)
    # Sensitizers should have cs < -1.0
    assert (df.loc[df["is_hit_sensitizer"], "cs"] < -1.0).all()
    # Resistors should have cs > 3.0
    assert (df.loc[df["is_hit_resistor"], "cs"] > 3.0).all()


def test_assign_hit_labels_no_overlap(tiny_screen_df):
    df = assign_hit_labels(tiny_screen_df)
    overlap = df["is_hit_sensitizer"] & df["is_hit_resistor"]
    assert not overlap.any(), "No gene should be both sensitizer and resistor"


def test_load_screen_scores_smoke(tmp_path):
    """Test load_screen_scores with a minimal synthetic file."""
    screen_file = tmp_path / "screen.tab.txt"
    content = "SCREEN_ID\tIDENTIFIER_ID\tIDENTIFIER_TYPE\tOFFICIAL_SYMBOL\tALIASES\tORGANISM_ID\tORGANISM_OFFICIAL\tSCORE.1\tSCORE.2\n"
    content += "1393\t596\tENTREZ_GENE\tBCL2\t\t9606\tHomo sapiens\t-2.5\t0.001\n"
    content += "1393\t598\tENTREZ_GENE\tBCL2L1\t\t9606\tHomo sapiens\t0.5\t0.05\n"
    screen_file.write_text(content)

    df = load_screen_scores(str(screen_file))
    assert len(df) == 2
    assert "gene_symbol" in df.columns
    assert "cs" in df.columns
    assert df.loc[df["gene_symbol"] == "BCL2", "cs"].values[0] == pytest.approx(-2.5)
