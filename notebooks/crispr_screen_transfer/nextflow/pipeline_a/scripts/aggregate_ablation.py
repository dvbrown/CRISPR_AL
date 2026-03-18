"""Aggregate feature ablation rows and identify impactful features.

Reads all *_ablation_row.json files in the current directory.
Identifies features where dropping them reduces Precision@50 by > 5% relative
to the full-feature baseline (computed from Loop 1 summary CSV).

Outputs:
  feature_ablation_ridge.csv   — per-feature × per-split ablation metrics
  top_features.txt             — features to KEEP in the reduced model
  ablation_summary.csv         — feature × mean Precision@50 + delta vs baseline
"""
import argparse
import glob
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary-csv", default=None,
                        help="design_a_summary.csv from Loop 1 to compute deltas")
    parser.add_argument("--top-k",      type=int, default=5,
                        help="Number of top features to keep in reduced model")
    parser.add_argument("--drop-threshold", type=float, default=0.05,
                        help="Minimum relative Precision@50 drop to flag a feature as important")
    args = parser.parse_args()

    ablation_files = sorted(glob.glob("*_ablation_row.json"))
    if not ablation_files:
        raise FileNotFoundError("No *_ablation_row.json files found")

    rows = []
    for path in ablation_files:
        with open(path) as f:
            rows.append(json.load(f))

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv("feature_ablation_ridge.csv", index=False)
    print(f"Loaded {len(abl_df)} ablation rows  ({abl_df['dropped_feature'].nunique()} features × "
          f"{abl_df['split_id'].nunique()} splits)")

    # Per-feature mean metrics
    feature_summary = (
        abl_df.groupby("dropped_feature")[["pearson", "precision_at_50", "auroc_sensitizer"]]
        .mean()
        .reset_index()
        .rename(columns={
            "pearson": "mean_pearson",
            "precision_at_50": "mean_precision_at_50",
            "auroc_sensitizer": "mean_auroc_sensitizer",
        })
    )

    # Load baseline P@50 if available
    if args.baseline_summary_csv:
        try:
            baseline = pd.read_csv(args.baseline_summary_csv)
            ridge_row = baseline[baseline["model"] == "ridge"]
            if not ridge_row.empty and "precision_at_50_mean" in ridge_row.columns:
                baseline_p50 = float(ridge_row["precision_at_50_mean"].iloc[0])
                feature_summary["delta_precision_at_50"] = (
                    feature_summary["mean_precision_at_50"] - baseline_p50
                )
                feature_summary["relative_drop"] = (
                    -feature_summary["delta_precision_at_50"] / max(baseline_p50, 1e-8)
                )
                print(f"\nBaseline Ridge P@50: {baseline_p50:.4f}")
            else:
                baseline_p50 = None
        except Exception as e:
            print(f"Warning: could not load baseline summary ({e})")
            baseline_p50 = None
    else:
        baseline_p50 = None
        feature_summary["delta_precision_at_50"] = float("nan")
        feature_summary["relative_drop"] = float("nan")

    # Sort by impact: features that hurt most when dropped → most important
    if "relative_drop" in feature_summary.columns and feature_summary["relative_drop"].notna().any():
        feature_summary = feature_summary.sort_values("relative_drop", ascending=False)
    else:
        feature_summary = feature_summary.sort_values("mean_precision_at_50", ascending=True)

    feature_summary.to_csv("ablation_summary.csv", index=False)

    # Select top features to KEEP
    all_features = [
        "molm13_log_tpm", "coessential_mean_r_top50", "coessential_molm13_chronos",
        "n_reactome_pathways", "n_go_bp_terms", "n_go_mf_terms",
        "in_hallmark_apoptosis", "in_hallmark_oxidative_phosphorylation", "n_kegg_pathways",
    ]
    top_dropped = feature_summary.head(args.top_k)["dropped_feature"].tolist()
    # Top features = features that matter most when dropped (keep them)
    top_features = top_dropped  # these are most important — keep them
    remaining = [f for f in all_features if f not in top_features]
    # Also keep features with relative_drop > threshold
    if baseline_p50 is not None:
        impactful = feature_summary[feature_summary["relative_drop"] > args.drop_threshold][
            "dropped_feature"
        ].tolist()
        top_features = sorted(set(top_features) | set(impactful))

    if not top_features:
        top_features = all_features[:args.top_k]

    with open("top_features.txt", "w") as f:
        f.write("\n".join(top_features) + "\n")

    print("\nFeature ablation summary (sorted by impact):")
    print(feature_summary.to_string(index=False))
    print(f"\nTop features to KEEP in reduced model ({len(top_features)}): {top_features}")
    print(f"Written to top_features.txt")


if __name__ == "__main__":
    main()
