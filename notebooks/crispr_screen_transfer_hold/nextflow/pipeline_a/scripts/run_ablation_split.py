"""Leave-one-out feature ablation: train Ridge with one feature dropped.

For one split × one dropped feature, trains Ridge on the remaining 8 features
and records Precision@K metrics. Part of Loop 2 (225 total runs: 9 × 25).

Outputs:
  {split_id}_{dropped_feature}_ablation_row.json — single-row ablation result
"""
import argparse
import json

import numpy as np
import pandas as pd

from crispr_al.io import load_parquet
from crispr_al.models import scale_features, train_ridge, predict
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
)

FEATURE_NAMES = [
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json",       required=True)
    parser.add_argument("--screen-parquet",   required=True)
    parser.add_argument("--features-parquet", required=True)
    parser.add_argument("--drop-feature",     required=True, help="Feature name to drop")
    parser.add_argument("--split-id",         required=True)
    args = parser.parse_args()

    if args.drop_feature not in FEATURE_NAMES:
        raise ValueError(f"Unknown feature: {args.drop_feature}. Must be one of {FEATURE_NAMES}")

    reduced_features = [f for f in FEATURE_NAMES if f != args.drop_feature]

    with open(args.split_json) as f:
        split = json.load(f)

    screen_df = load_parquet(args.screen_parquet).reset_index()
    features_df = load_parquet(args.features_parquet)

    score_idx = screen_df.set_index("gene_symbol")["score_norm"]
    hit_sens_idx = screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
    hit_res_idx = screen_df.set_index("gene_symbol")["is_hit_resistor"]

    train_genes = split["train_genes"]
    test_genes = split["test_genes"]

    X_train = features_df.loc[train_genes, reduced_features].values.astype(np.float64)
    y_train = score_idx.loc[train_genes].values
    X_test = features_df.loc[test_genes, reduced_features].values.astype(np.float64)
    y_test = score_idx.loc[test_genes].values
    hit_sens = hit_sens_idx.loc[test_genes].values
    hit_res = hit_res_idx.loc[test_genes].values

    X_train_s, X_test_s = scale_features(X_train, X_test)
    model = train_ridge(X_train_s, y_train)
    y_pred = predict(model, X_test_s)

    reg = compute_regression_metrics(y_test, y_pred)
    rank = compute_ranking_metrics(y_pred, hit_sens, hit_res)
    clf = compute_classification_metrics(y_pred, hit_sens, hit_res)

    k_map = {km["k"]: km for km in rank["k_metrics"]}
    result = {
        "split_id": split["split_id"],
        "dropped_feature": args.drop_feature,
        "n_features_used": len(reduced_features),
        "pearson": reg["pearson"],
        "auroc_sensitizer": clf["labels"][0]["auroc"],
        "auprc_sensitizer": clf["labels"][0]["auprc"],
        "precision_at_50":  k_map[50]["precision_at_k"],
        "recall_at_50":     k_map[50]["recall_at_k"],
        "precision_at_100": k_map[100]["precision_at_k"],
        "precision_at_200": k_map[200]["precision_at_k"],
        "precision_at_500": k_map[500]["precision_at_k"],
    }

    out_name = f"{args.split_id}_{args.drop_feature}_ablation_row.json"
    with open(out_name, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{args.split_id} / drop={args.drop_feature}  |  "
          f"Pearson={reg['pearson']:.3f}  P@50={k_map[50]['precision_at_k']:.3f}")


if __name__ == "__main__":
    main()
