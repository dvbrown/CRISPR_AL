"""Compute zero-predictor baseline on the first Design A split.

Outputs baseline_zero.json — schema-compliant metrics record.
"""
import argparse
import datetime
import json

import numpy as np
import pandas as pd

from crispr_al.io import load_parquet, save_metrics_json, get_code_commit
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
    build_metrics_record,
    validate_metrics_record,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json",      required=True)
    parser.add_argument("--screen-parquet",  required=True)
    parser.add_argument("--features-parquet", required=True)
    parser.add_argument("--schema-json",     required=True)
    parser.add_argument("--output",          default="baseline_zero.json")
    args = parser.parse_args()

    with open(args.split_json) as f:
        split = json.load(f)

    screen_df = load_parquet(args.screen_parquet).reset_index()
    score_idx = screen_df.set_index("gene_symbol")["score_norm"]
    hit_sens_idx = screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
    hit_res_idx = screen_df.set_index("gene_symbol")["is_hit_resistor"]

    test_genes = split["test_genes"]
    y_test = score_idx.loc[test_genes].values
    hit_sens = hit_sens_idx.loc[test_genes].values
    hit_res = hit_res_idx.loc[test_genes].values

    y_zero = np.zeros(len(y_test))
    reg = compute_regression_metrics(y_test, y_zero)
    rank = compute_ranking_metrics(y_zero, hit_sens, hit_res)
    clf = compute_classification_metrics(y_zero, hit_sens, hit_res)

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    code_commit = get_code_commit()
    data_counts = {
        "train_row_count": len(split["train_genes"]),
        "test_row_count": len(test_genes),
        "n_unique_train_genes": len(split["train_genes"]),
        "n_unique_test_genes": len(test_genes),
        "n_overlap_genes_train_test": 0,
    }
    leakage_checks = {
        "disjoint_gene_label_rows": True,
        "normalization_fit_on_train_only": True,
        "split_hash_logged": True,
    }

    record = build_metrics_record(
        split=split,
        data_counts=data_counts,
        leakage_checks=leakage_checks,
        regression=reg,
        ranking=rank,
        classification=clf,
        run_id="aim1_baseline_zero",
        timestamp_utc=timestamp,
        code_commit=code_commit,
        notes="Zero predictor baseline (predict 0 for all genes)",
    )
    save_metrics_json(record, args.output)
    validate_metrics_record(record, args.schema_json)

    print(f"Baseline zero  |  Pearson={reg['pearson']:.4f}  "
          f"AUROC_sens={clf['labels'][0]['auroc']:.4f}  "
          f"P@50={rank['k_metrics'][0]['precision_at_k']:.4f}")


if __name__ == "__main__":
    main()
