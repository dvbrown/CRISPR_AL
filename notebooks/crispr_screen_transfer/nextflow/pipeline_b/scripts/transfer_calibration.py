"""Transfer calibration analysis for Design A (Loop 4) and Design B (Loop 2).

Runs RF across all splits to extract feature importances and per-gene predictions,
then computes:
  - Predicted vs actual score distribution statistics
  - Rank correlation stratified by hit/non-hit, expression quartile, co-essentiality quartile
  - RF feature_importances_ averaged across all splits

Outputs (suffix defaults to "" for Design A; pass e.g. "_c2s" for Design B):
  transfer_calibration_design_a{suffix}.csv  — stratified rank correlations
  figure_score_dist{suffix}.png              — predicted vs actual score distribution
  figure_stratified_spearman{suffix}.png     — Spearman r by stratum
  figure_feature_importance{suffix}.png      — mean RF feature importances

Design B cross-screen usage: pass --train-screen-parquet for the train screen and
--screen-parquet for the test screen.  Feature zero-imputation is applied
automatically for genes absent from the feature matrix.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crispr_al.io import load_parquet
from crispr_al.models import scale_features, train_rf, predict

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
    parser.add_argument("--split-jsons",         required=True, nargs="+",
                        help="Split JSON files")
    parser.add_argument("--screen-parquet",       required=True,
                        help="Test screen parquet (also used as train screen if "
                             "--train-screen-parquet is not supplied)")
    parser.add_argument("--train-screen-parquet", default=None,
                        help="Train screen parquet (Design B cross-screen only)")
    parser.add_argument("--features-parquet",     required=True)
    parser.add_argument("--n-estimators",         type=int, default=200)
    parser.add_argument("--output-csv",           default="transfer_calibration_design_a.csv")
    parser.add_argument("--figure-suffix",        default="",
                        help="Suffix appended to figure filenames (e.g. '_c2s')")
    parser.add_argument("--features-subset-file", default=None,
                        help="Text file with one feature name per line (optimal feature set)")
    args = parser.parse_args()

    test_screen_df = load_parquet(args.screen_parquet).reset_index()
    if args.train_screen_parquet:
        train_screen_df = load_parquet(args.train_screen_parquet).reset_index()
    else:
        train_screen_df = test_screen_df

    features_df = load_parquet(args.features_parquet)

    feature_cols = ALL_FEATURE_NAMES
    if args.features_subset_file and Path(args.features_subset_file).exists():
        feature_cols = [ln.strip() for ln in
                        Path(args.features_subset_file).read_text().splitlines()
                        if ln.strip()]
        print(f"Using feature subset ({len(feature_cols)}): {feature_cols}")

    # Extend feature index to cover all genes (zero-impute Sharon-only genes)
    train_genes_all = train_screen_df["gene_symbol"].tolist()
    test_genes_all  = test_screen_df["gene_symbol"].tolist()
    all_needed = list(set(train_genes_all) | set(test_genes_all))
    features_df = features_df.reindex(all_needed, fill_value=0.0)

    train_score_idx = train_screen_df.set_index("gene_symbol")["score_norm"]
    test_score_idx  = test_screen_df.set_index("gene_symbol")["score_norm"]
    hit_sens_idx    = test_screen_df.set_index("gene_symbol")["is_hit_sensitizer"]

    all_importances = []
    stratified_rows = []
    all_y_test, all_y_pred = [], []

    # Hoist loop-invariant feature availability checks
    stratify_by_expr  = "molm13_log_tpm" in feature_cols
    stratify_by_coess = "coessential_mean_r_top50" in feature_cols

    split_files = sorted(args.split_jsons)
    print(f"Running RF calibration across {len(split_files)} splits…")

    for split_path in split_files:
        with open(split_path) as f:
            split = json.load(f)

        train_genes = split["train_genes"]
        test_genes  = split["test_genes"]

        # features_df is already reindexed to cover all needed genes; use .loc for subset
        X_train = features_df.loc[train_genes, feature_cols].values.astype(np.float64)
        y_train = train_score_idx.loc[train_genes].values
        X_test  = features_df.loc[test_genes,  feature_cols].values.astype(np.float64)
        y_test  = test_score_idx.loc[test_genes].values
        hit_sens = hit_sens_idx.loc[test_genes].values

        X_train_s, X_test_s = scale_features(X_train, X_test)
        rf = train_rf(X_train_s, y_train, seed=split["seed"], n_estimators=args.n_estimators)
        y_pred = predict(rf, X_test_s)

        all_importances.append(rf.feature_importances_)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        # Stratify by hit/non-hit
        for stratum_name, mask in [("hit_sensitizer", hit_sens), ("non_hit", ~hit_sens)]:
            if mask.sum() > 10:
                rho = float(spearmanr(y_test[mask], y_pred[mask]).statistic)
                stratified_rows.append({
                    "split_id": split["split_id"],
                    "stratum": stratum_name,
                    "n": int(mask.sum()),
                    "spearman_r": rho,
                })

        # Stratify by expression quartile
        if stratify_by_expr:
            expr_vals = features_df.loc[test_genes, "molm13_log_tpm"].values
            expr_q = pd.qcut(expr_vals, 4, labels=["Q1", "Q2", "Q3", "Q4"])
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                mask_q = expr_q == q
                if mask_q.sum() > 10:
                    rho = float(spearmanr(y_test[mask_q], y_pred[mask_q]).statistic)
                    stratified_rows.append({
                        "split_id": split["split_id"],
                        "stratum": f"expr_{q}",
                        "n": int(mask_q.sum()),
                        "spearman_r": rho,
                    })

        # Stratify by co-essentiality quartile
        if stratify_by_coess:
            coess_vals = features_df.loc[test_genes, "coessential_mean_r_top50"].values
            if coess_vals.std() > 0:
                coess_q = pd.qcut(coess_vals, 4, labels=["Q1", "Q2", "Q3", "Q4"],
                                  duplicates="drop")
                for q in coess_q.cat.categories:
                    mask_q = coess_q == q
                    if mask_q.sum() > 10:
                        rho = float(spearmanr(y_test[mask_q], y_pred[mask_q]).statistic)
                        stratified_rows.append({
                            "split_id": split["split_id"],
                            "stratum": f"coess_{q}",
                            "n": int(mask_q.sum()),
                            "spearman_r": rho,
                        })

        print(f"  ✓ {split['split_id']}")

    sfx = args.figure_suffix

    # -------------------------------------------------------------------------
    # Save stratified calibration CSV
    strat_df = pd.DataFrame(stratified_rows)
    strat_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(strat_df)} stratified rows → {args.output_csv}")

    # -------------------------------------------------------------------------
    # Figure 1: score distribution
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(all_y_test, all_y_pred, alpha=0.05, s=1, c="#1f77b4", rasterized=True)
    ax1.set_xlabel("Actual score_norm")
    ax1.set_ylabel("Predicted score_norm")
    ax1.set_title("Predicted vs Actual Scores (all splits, RF)")
    corr = np.corrcoef(all_y_test, all_y_pred)[0, 1]
    ax1.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax1.transAxes, va="top")
    fig1.tight_layout()
    fig1.savefig(f"figure_score_dist{sfx}.png", dpi=150)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: stratified Spearman r
    strat_mean = strat_df.groupby("stratum")["spearman_r"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.barh(strat_mean["stratum"], strat_mean["spearman_r"], color="#2ca02c")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Mean Spearman ρ across splits")
    ax2.set_title("Rank Correlation by Stratum (RF)")
    fig2.tight_layout()
    fig2.savefig(f"figure_stratified_spearman{sfx}.png", dpi=150)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Figure 3: mean RF feature importances
    mean_importance = np.mean(all_importances, axis=0)
    order = np.argsort(mean_importance)[::-1]
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.barh(
        [feature_cols[i] for i in order[::-1]],
        mean_importance[order[::-1]],
        color="#ff7f0e",
    )
    ax3.set_xlabel("Mean RF feature importance (MDI)")
    ax3.set_title(f"RF Feature Importances (averaged over {len(split_files)} splits)")
    fig3.tight_layout()
    fig3.savefig(f"figure_feature_importance{sfx}.png", dpi=150)
    plt.close(fig3)

    # Print importance summary
    print("\nRF Feature Importances (mean MDI):")
    for i in order:
        flag = " ← near-zero" if mean_importance[i] < 0.02 else ""
        print(f"  {feature_cols[i]:<45} {mean_importance[i]:.4f}{flag}")


if __name__ == "__main__":
    main()
