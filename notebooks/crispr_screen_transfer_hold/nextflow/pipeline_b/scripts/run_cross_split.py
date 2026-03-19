"""Train Ridge or RF on one Design B cross-screen split and compute all metrics.

Key difference from run_split.py: takes separate train and test screen parquets
and uses features_df.reindex(genes, fill_value=0.0) to handle Sharon-only genes
that are absent from the feature matrix.

Outputs:
  {split_id}_{model}.json        — schema-compliant metrics record
  {split_id}_{model}_row.csv     — flat row for aggregation (includes direction column)
  {split_id}_{model}_preds.csv   — per-gene predictions for calibration analysis
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crispr_al.io import load_parquet, save_metrics_json, get_code_commit
from crispr_al.models import scale_features, train_ridge, train_rf, predict
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
    build_metrics_record,
    flatten_metrics_row,
    validate_metrics_record,
)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json",           required=True)
    parser.add_argument("--train-screen-parquet",  required=True,
                        help="Parquet for the train screen (chen or sharon)")
    parser.add_argument("--test-screen-parquet",   required=True,
                        help="Parquet for the test screen (sharon or chen)")
    parser.add_argument("--features-parquet",      required=True)
    parser.add_argument("--schema-json",           required=True)
    parser.add_argument("--model",                 required=True, choices=["ridge", "rf"])
    parser.add_argument("--split-id",              required=True,
                        help="Split ID prefix for output file names")
    parser.add_argument("--n-estimators",          type=int, default=200)
    parser.add_argument("--features-subset-file",  default=None,
                        help="Text file with one feature name per line (optimal feature set)")
    args = parser.parse_args()

    # Load split
    with open(args.split_json) as f:
        split = json.load(f)

    # Load screens
    train_screen_df = load_parquet(args.train_screen_parquet).reset_index()
    test_screen_df  = load_parquet(args.test_screen_parquet).reset_index()

    train_score_idx    = train_screen_df.set_index("gene_symbol")["score_norm"]
    test_score_idx     = test_screen_df.set_index("gene_symbol")["score_norm"]
    test_hit_sens_idx  = test_screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
    test_hit_res_idx   = test_screen_df.set_index("gene_symbol")["is_hit_resistor"]

    # Load features; reindex to cover all genes (zero-impute Sharon-only)
    features_df = load_parquet(args.features_parquet)

    # Feature subset
    feature_cols = ALL_FEATURE_NAMES
    if args.features_subset_file and Path(args.features_subset_file).exists():
        with open(args.features_subset_file) as f:
            feature_cols = [ln.strip() for ln in f if ln.strip()]
        print(f"Using feature subset ({len(feature_cols)}): {feature_cols}")

    train_genes = split["train_genes"]
    test_genes  = split["test_genes"]

    # Zero-impute missing genes in features
    all_needed = list(set(train_genes) | set(test_genes))
    features_df = features_df.reindex(all_needed, fill_value=0.0)

    n_imputed = sum(1 for g in all_needed if g not in features_df.index or
                    features_df.loc[g, feature_cols].isna().any())

    X_train = features_df.reindex(train_genes, fill_value=0.0)[feature_cols].values.astype(np.float64)
    y_train = train_score_idx.loc[train_genes].values
    X_test  = features_df.reindex(test_genes,  fill_value=0.0)[feature_cols].values.astype(np.float64)
    y_test  = test_score_idx.loc[test_genes].values
    hit_sens = test_hit_sens_idx.loc[test_genes].values
    hit_res  = test_hit_res_idx.loc[test_genes].values

    assert len(set(train_genes) & set(test_genes)) == 0, "Train/test gene overlap!"

    X_train_s, X_test_s = scale_features(X_train, X_test)

    if args.model == "ridge":
        model_obj = train_ridge(X_train_s, y_train)
    else:
        model_obj = train_rf(X_train_s, y_train, seed=split["seed"],
                             n_estimators=args.n_estimators)

    y_pred = predict(model_obj, X_test_s)

    reg  = compute_regression_metrics(y_test, y_pred)
    rank = compute_ranking_metrics(y_pred, hit_sens, hit_res)
    clf  = compute_classification_metrics(y_pred, hit_sens, hit_res)

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    code_commit = get_code_commit()
    data_counts = {
        "train_row_count":        len(train_genes),
        "test_row_count":         len(test_genes),
        "n_unique_train_genes":   len(set(train_genes)),
        "n_unique_test_genes":    len(set(test_genes)),
        "n_overlap_genes_train_test": 0,
    }
    leakage_checks = {
        "disjoint_gene_label_rows":        True,
        "normalization_fit_on_train_only":  True,
        "split_hash_logged":               True,
    }

    record = build_metrics_record(
        split=split,
        data_counts=data_counts,
        leakage_checks=leakage_checks,
        regression=reg,
        ranking=rank,
        classification=clf,
        run_id=f"{args.split_id}_{args.model}",
        timestamp_utc=timestamp,
        code_commit=code_commit,
    )
    out_json = f"{args.split_id}_{args.model}.json"
    save_metrics_json(record, out_json)
    validate_metrics_record(record, args.schema_json)

    # Flat row CSV — adds direction column for cross-results aggregation
    # Normalise to short names used by aggregate_cross_results.py
    _train_id = split["train_screen_id"]
    _test_id  = split["test_screen_id"]
    direction = "chen_to_sharon" if "chen" in _train_id else "sharon_to_chen"
    row = {"model": args.model, "direction": direction,
           **flatten_metrics_row(split, reg, rank, clf)}
    out_row = f"{args.split_id}_{args.model}_row.csv"
    pd.DataFrame([row]).to_csv(out_row, index=False)

    # Per-gene predictions CSV
    preds_df = pd.DataFrame({
        "gene_symbol":       test_genes,
        "y_test":            y_test,
        "y_pred":            y_pred,
        "is_hit_sensitizer": hit_sens,
        "is_hit_resistor":   hit_res,
        "split_id":          split["split_id"],
        "model":             args.model,
        "direction":          direction,
    })
    out_preds = f"{args.split_id}_{args.model}_preds.csv"
    preds_df.to_csv(out_preds, index=False)

    print(f"{args.split_id} / {args.model} ({direction})  |  "
          f"Pearson={reg['pearson']:.3f}  "
          f"AUROC_sens={clf['labels'][0]['auroc']:.3f}  "
          f"P@50={rank['k_metrics'][0]['precision_at_k']:.3f}")


if __name__ == "__main__":
    main()
