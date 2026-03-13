"""Tests for crispr_al.features module."""
import numpy as np
import pandas as pd
import pytest
from crispr_al.features import assemble_gene_features, _strip_entrez, _parse_gmt
import tempfile
import os


def test_strip_entrez():
    assert _strip_entrez("BCL2 (596)") == "BCL2"
    assert _strip_entrez("TP53 (7157)") == "TP53"
    assert _strip_entrez("GENE") == "GENE"


def test_parse_gmt(tmp_path):
    gmt_file = tmp_path / "test.gmt"
    gmt_file.write_text("SET1\tdesc\tGENEA\tGENEB\tGENEC\nSET2\tdesc2\tGENED\n")
    result = _parse_gmt(str(gmt_file))
    assert "SET1" in result
    assert "GENEA" in result["SET1"]
    assert len(result["SET2"]) == 1


def test_assemble_gene_features_shape(tiny_screen_df, tiny_features_df):
    genes = tiny_screen_df["gene_symbol"].tolist()
    expr_series = tiny_features_df["molm13_log_tpm"]
    coess_df = tiny_features_df[["coessential_mean_r_top50", "coessential_molm13_chronos"]]
    pathway_df = tiny_features_df[[
        "n_reactome_pathways", "n_go_bp_terms", "n_go_mf_terms",
        "in_hallmark_apoptosis", "in_hallmark_oxidative_phosphorylation", "n_kegg_pathways"
    ]]
    result = assemble_gene_features(genes, expr_series, coess_df, pathway_df)
    assert result.shape == (len(genes), 9)
    assert result.columns.tolist() == [
        "molm13_log_tpm",
        "coessential_mean_r_top50",
        "coessential_molm13_chronos",
        "n_reactome_pathways",
        "n_go_bp_terms",
        "n_go_mf_terms",
        "in_hallmark_apoptosis",
        "in_hallmark_oxidative_phosphorylation",
        "n_kegg_pathways",
    ]


def test_assemble_gene_features_missing_filled(tiny_screen_df, tiny_features_df):
    """Missing genes should be filled with 0.0."""
    genes = tiny_screen_df["gene_symbol"].tolist()
    # Provide expr for only half the genes
    partial_expr = tiny_features_df["molm13_log_tpm"].iloc[:25]
    coess_df = tiny_features_df[["coessential_mean_r_top50", "coessential_molm13_chronos"]]
    pathway_df = tiny_features_df[[
        "n_reactome_pathways", "n_go_bp_terms", "n_go_mf_terms",
        "in_hallmark_apoptosis", "in_hallmark_oxidative_phosphorylation", "n_kegg_pathways"
    ]]
    result = assemble_gene_features(genes, partial_expr, coess_df, pathway_df)
    # Missing genes should have 0.0 for molm13_log_tpm
    missing_genes = genes[25:]
    assert (result.loc[missing_genes, "molm13_log_tpm"] == 0.0).all()
