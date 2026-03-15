"""Transfer calibration analysis for Design A (Loop 4).

Runs RF across all 25 splits to extract feature importances and per-gene predictions,
then computes:
  - Predicted vs actual score distribution statistics
  - Rank correlation stratified by hit/non-hit, expression quartile, co-essentiality quartile
  - RF feature_importances_ averaged across all splits

Outputs:
  transfer_calibration_design_a.csv  — stratified rank correlations
  figure_score_dist.png              — predicted vs actual score distribution
  figure_stratified_spearman.png     — Spearman r by stratum
  figure_feature_importance.png      — mean RF feature importances
"""
import argparse
import glob
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
    parser.add_argument("--split-jsons",     required=True, nargs="+",
                        help="Split JSON files (all 25)")
    parser.add_argument("--screen-parquet",  required=True)
    parser.add_argument("--features-parquet", required=True)
    parser.add_argument("--n-estimators",    type=int, default=200)
    parser.add_argument("--output-csv",      default="transfer_calibration_design_a.csv")
    args = parser.parse_args()

    screen_df = load_parquet(args.screen_parquet).reset_index()
    features_df = load_parquet(args.features_parquet)

    score_idx = screen_df.set_index("gene_symbol")["score_norm"]
    hit_sens_idx = screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
    hit_res_idx = screen_df.set_index("gene_symbol")["is_hit_resistor"]

    all_importances = []
    stratified_rows = []
    all_y_test, all_y_pred = [], []

    split_files = sorted(args.split_jsons)
    print(f"Running RF calibration across {len(split_files)} splits…")

    for split_path in split_files:
        with open(split_path) as f:
            split = json.load(f)

        train_genes = split["train_genes"]
        test_genes = split["test_genes"]

        X_train = features_df.loc[train_genes, FEATURE_NAMES].values.astype(np.float64)
        y_train = score_idx.loc[train_genes].values
        X_test = features_df.loc[test_genes, FEATURE_NAMES].values.astype(np.float64)
        y_test = score_idx.loc[test_genes].values
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
        coess_vals = features_df.loc[test_genes, "coessential_mean_r_top50"].values
        if coess_vals.std() > 0:
            coess_q = pd.qcut(coess_vals, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
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
    ax1.set_title("Design A: Predicted vs Actual Scores (all splits, RF)")
    corr = np.corrcoef(all_y_test, all_y_pred)[0, 1]
    ax1.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax1.transAxes, va="top")
    fig1.tight_layout()
    fig1.savefig("figure_score_dist.png", dpi=150)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: stratified Spearman r
    strat_mean = strat_df.groupby("stratum")["spearman_r"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.barh(strat_mean["stratum"], strat_mean["spearman_r"], color="#2ca02c")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Mean Spearman ρ across splits")
    ax2.set_title("Design A: Rank Correlation by Stratum (RF)")
    fig2.tight_layout()
    fig2.savefig("figure_stratified_spearman.png", dpi=150)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Figure 3: mean RF feature importances
    mean_importance = np.mean(all_importances, axis=0)
    order = np.argsort(mean_importance)[::-1]
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.barh(
        [FEATURE_NAMES[i] for i in reversed(order)],
        mean_importance[reversed(order)],
        color="#ff7f0e",
    )
    ax3.set_xlabel("Mean RF feature importance (MDI)")
    ax3.set_title("Design A: RF Feature Importances (averaged over 25 splits)")
    fig3.tight_layout()
    fig3.savefig("figure_feature_importance.png", dpi=150)
    plt.close(fig3)

    # Print importance summary
    print("\nRF Feature Importances (mean MDI):")
    for i in order:
        flag = " ← near-zero" if mean_importance[i] < 0.02 else ""
        print(f"  {FEATURE_NAMES[i]:<45} {mean_importance[i]:.4f}{flag}")


if __name__ == "__main__":
    main()
