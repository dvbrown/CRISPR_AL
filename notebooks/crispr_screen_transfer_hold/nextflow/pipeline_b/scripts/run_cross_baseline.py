"""Compute overlap-only baseline for Design B cross-screen transfer.

Trains on genes shared by both screens, tests on each full screen.
Produces two metric JSONs:
  baseline_chen_to_sharon.json  — train on overlap, test on all Sharon genes
  baseline_sharon_to_chen.json  — train on overlap, test on all Chen genes
"""
import argparse
import datetime
import json

import numpy as np
import pandas as pd

from crispr_al.io import load_parquet, save_metrics_json, get_code_commit
from crispr_al.models import scale_features, train_ridge, predict
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
    build_metrics_record,
    validate_metrics_record,
    flatten_metrics_row,
)
from crispr_al.splits import compute_overlap_baseline_hash

CHEN_SCREEN_ID   = "chen2019_1393"
SHARON_SCREEN_ID = "sharon2019_1402"

ALL_FEATURE_NAMES = [
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


def run_baseline(
    train_screen_df: pd.DataFrame,
    test_screen_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: list,
    train_screen_id: str,
    test_screen_id: str,
    shared_genes: list,
    schema_json: str,
) -> tuple[dict, dict]:
    """Train Ridge on shared genes; test on all test screen genes.

    Returns (metrics_record, flat_row_dict).
    """
    # Train on shared genes using train screen scores
    train_score_idx = train_screen_df.set_index("gene_symbol")["score_norm"]
    test_score_idx  = test_screen_df.set_index("gene_symbol")["score_norm"]
    test_hit_sens   = test_screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
    test_hit_res    = test_screen_df.set_index("gene_symbol")["is_hit_resistor"]

    # Test genes = all test screen genes (overlap is legitimate; zero-predictor-like baseline)
    test_genes  = test_screen_df["gene_symbol"].tolist()
    train_genes = shared_genes

    X_train = features_df.reindex(train_genes, fill_value=0.0)[feature_cols].values.astype(float)
    y_train = train_score_idx.loc[train_genes].values
    X_test  = features_df.reindex(test_genes,  fill_value=0.0)[feature_cols].values.astype(float)
    y_test  = test_score_idx.loc[test_genes].values
    hit_sens = test_hit_sens.loc[test_genes].values
    hit_res  = test_hit_res.loc[test_genes].values

    X_train_s, X_test_s = scale_features(X_train, X_test)
    model_obj = train_ridge(X_train_s, y_train)
    y_pred = predict(model_obj, X_test_s)

    reg  = compute_regression_metrics(y_test, y_pred)
    rank = compute_ranking_metrics(y_pred, hit_sens, hit_res)
    clf  = compute_classification_metrics(y_pred, hit_sens, hit_res)

    split_hash = compute_overlap_baseline_hash(
        generator_id="aim1_overlap_baseline",
        train_screen_id=train_screen_id,
        test_screen_id=test_screen_id,
        shared_genes=shared_genes,
    )

    split_record = {
        "split_id":        f"aim1_overlap_baseline_{train_screen_id}_to_{test_screen_id}",
        "generator_id":    "aim1_overlap_baseline",
        "family":          "context_zeroshot",
        "aim":             "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed":            0,
        "repeat_index":    1,
        "train_screen_id": train_screen_id,
        "test_screen_id":  test_screen_id,
        "split_hash":      split_hash,
        "train_genes":     sorted(train_genes),
        "test_genes":      sorted(test_genes),
    }

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    code_commit = get_code_commit()
    data_counts = {
        "train_row_count":            len(train_genes),
        "test_row_count":             len(test_genes),
        "n_unique_train_genes":       len(set(train_genes)),
        "n_unique_test_genes":        len(set(test_genes)),
        "n_overlap_genes_train_test": len(set(train_genes) & set(test_genes)),
    }
    leakage_checks = {
        "disjoint_gene_label_rows":       False,  # overlap baseline: shared genes in test
        "normalization_fit_on_train_only": True,
        "split_hash_logged":              True,
    }

    record = build_metrics_record(
        split=split_record,
        data_counts=data_counts,
        leakage_checks=leakage_checks,
        regression=reg,
        ranking=rank,
        classification=clf,
        run_id=f"aim1_overlap_baseline_{train_screen_id}_to_{test_screen_id}",
        timestamp_utc=timestamp,
        code_commit=code_commit,
        notes=f"Overlap-only baseline: train on {len(train_genes)} shared genes, "
              f"test on all {len(test_genes)} {test_screen_id} genes",
    )

    validate_metrics_record(record, schema_json)

    direction = "chen_to_sharon" if "chen" in train_screen_id else "sharon_to_chen"
    row = {"model": "overlap_baseline", "direction": direction,
           **flatten_metrics_row(split_record, reg, rank, clf)}
    return record, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chen-screen-parquet",   required=True)
    parser.add_argument("--sharon-screen-parquet",  required=True)
    parser.add_argument("--features-parquet",       required=True)
    parser.add_argument("--schema-json",            required=True)
    parser.add_argument("--features-subset-file",   default=None,
                        help="Text file with one feature name per line (optimal feature set)")
    args = parser.parse_args()

    chen_df   = load_parquet(args.chen_screen_parquet).reset_index()
    sharon_df = load_parquet(args.sharon_screen_parquet).reset_index()
    features_df = load_parquet(args.features_parquet)

    feature_cols = ALL_FEATURE_NAMES
    if args.features_subset_file:
        import re
        from pathlib import Path
        p = Path(args.features_subset_file)
        if p.exists():
            feature_cols = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            print(f"Using feature subset ({len(feature_cols)}): {feature_cols}")

    chen_genes   = set(chen_df["gene_symbol"].tolist())
    sharon_genes = set(sharon_df["gene_symbol"].tolist())
    shared_genes = sorted(chen_genes & sharon_genes)
    print(f"Shared genes: {len(shared_genes):,}  "
          f"(Chen {len(chen_genes):,}  Sharon {len(sharon_genes):,})")

    # Extend features index to cover all genes
    all_genes_needed = list(chen_genes | sharon_genes)
    features_df = features_df.reindex(all_genes_needed, fill_value=0.0)

    # Chen → Sharon baseline
    rec_c2s, row_c2s = run_baseline(
        train_screen_df=chen_df,
        test_screen_df=sharon_df,
        features_df=features_df,
        feature_cols=feature_cols,
        train_screen_id=CHEN_SCREEN_ID,
        test_screen_id=SHARON_SCREEN_ID,
        shared_genes=shared_genes,
        schema_json=args.schema_json,
    )
    save_metrics_json(rec_c2s, "baseline_chen_to_sharon.json")
    pd.DataFrame([row_c2s]).to_csv("baseline_chen_to_sharon_row.csv", index=False)

    # Sharon → Chen baseline
    rec_s2c, row_s2c = run_baseline(
        train_screen_df=sharon_df,
        test_screen_df=chen_df,
        features_df=features_df,
        feature_cols=feature_cols,
        train_screen_id=SHARON_SCREEN_ID,
        test_screen_id=CHEN_SCREEN_ID,
        shared_genes=shared_genes,
        schema_json=args.schema_json,
    )
    save_metrics_json(rec_s2c, "baseline_sharon_to_chen.json")
    pd.DataFrame([row_s2c]).to_csv("baseline_sharon_to_chen_row.csv", index=False)

    for rec in [rec_c2s, rec_s2c]:
        sp = rec["split"]
        m = rec["metrics"]
        rank_k = {x["k"]: x for x in m["ranking"]["k_metrics"]}
        print(f"{sp['split_id']}  |  "
              f"Pearson={m['regression']['pearson']:.3f}  "
              f"AUROC_sens={m['classification']['labels'][0]['auroc']:.3f}  "
              f"P@50={rank_k.get(50, {}).get('precision_at_k', float('nan')):.3f}")


if __name__ == "__main__":
    main()
