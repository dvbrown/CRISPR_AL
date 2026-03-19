"""Build the 9-feature gene matrix for Design A.

Features:
  molm13_log_tpm, coessential_mean_r_top50, coessential_molm13_chronos,
  n_reactome_pathways, n_go_bp_terms, n_go_mf_terms,
  in_hallmark_apoptosis, in_hallmark_oxidative_phosphorylation, n_kegg_pathways

Pass --quick to zero-impute DepMap features (fast testing mode).
"""
import argparse

import pandas as pd

from crispr_al.features import (
    build_expression_feature,
    build_coessentiality_features,
    build_pathway_features,
    assemble_gene_features,
)
from crispr_al.io import load_parquet, save_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-parquet", required=True, help="screen_scores.parquet path")
    parser.add_argument("--ccle-path",      required=True, help="CCLE expression CSV.gz path")
    parser.add_argument("--depmap-path",    required=True, help="DepMap CRISPRGeneEffect CSV.gz")
    parser.add_argument("--reactome-path",  required=True, help="NCBI2Reactome_PE_Pathway file")
    parser.add_argument("--goa-path",       required=True, help="goa_human.gaf.gz path")
    parser.add_argument("--hallmarks-path", required=True, help="Hallmarks GMT(.gz) path")
    parser.add_argument("--kegg-path",      required=True, help="KEGG GMT(.gz) path")
    parser.add_argument("--output",         default="gene_features.parquet", help="Output parquet")
    parser.add_argument("--quick",          action="store_true",
                        help="Zero-impute DepMap features (fast test mode)")
    args = parser.parse_args()

    screen_df = load_parquet(args.screen_parquet).reset_index()
    screen_genes = screen_df["gene_symbol"].tolist()

    print(f"Building features for {len(screen_genes):,} genes…")

    expr_series = build_expression_feature(args.ccle_path)
    print("  ✓ CCLE expression loaded")

    if args.quick:
        import numpy as np
        coess_df = pd.DataFrame(
            {"coessential_mean_r_top50": 0.0, "coessential_molm13_chronos": 0.0},
            index=pd.Index(screen_genes, name="gene_symbol"),
        )
        print("  ✓ DepMap co-essentiality zero-imputed (--quick mode)")
    else:
        coess_df = build_coessentiality_features(args.depmap_path, screen_genes)
        print("  ✓ DepMap co-essentiality computed")

    pathway_df = build_pathway_features(
        args.reactome_path, args.goa_path, args.hallmarks_path, args.kegg_path, screen_df
    )
    print("  ✓ Pathway features built")

    features_df = assemble_gene_features(screen_genes, expr_series, coess_df, pathway_df)
    save_parquet(features_df, args.output)
    print(f"Saved {features_df.shape[0]:,} genes × {features_df.shape[1]} features → {args.output}")


if __name__ == "__main__":
    main()
