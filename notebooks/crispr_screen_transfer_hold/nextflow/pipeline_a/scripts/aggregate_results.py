"""Aggregate per-split metrics row CSVs into summary tables with BCa bootstrap CIs.

Reads all *_row.csv files in the current directory (collected by Nextflow).
Outputs:
  {tag}_results_ridge.csv   — per-split Ridge metrics
  {tag}_results_rf.csv      — per-split RF metrics
  {tag}_results_all.csv     — all models combined
  {tag}_summary.csv         — mean ± BCa 95% CI per model
"""
import argparse
import glob

import pandas as pd

from crispr_al.metrics import bootstrap_ci_bca

METRIC_COLS = [
    "pearson", "spearman", "r2", "rmse", "mae",
    "auroc_sensitizer", "auroc_resistor",
    "auprc_sensitizer", "auprc_resistor",
    "precision_at_50", "precision_at_100", "precision_at_200", "precision_at_500",
    "recall_at_50",    "recall_at_100",    "recall_at_200",    "recall_at_500",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="design_a", help="Output filename prefix")
    args = parser.parse_args()

    row_files = sorted(glob.glob("*_row.csv"))
    if not row_files:
        raise FileNotFoundError("No *_row.csv files found in current directory")

    all_df = pd.concat([pd.read_csv(f) for f in row_files], ignore_index=True)
    print(f"Loaded {len(row_files)} row files  →  {len(all_df)} total rows")

    # Per-model CSVs
    for model_name in ["ridge", "rf"]:
        subset = all_df[all_df["model"] == model_name].copy()
        subset.to_csv(f"{args.tag}_results_{model_name}.csv", index=False)
        print(f"  {model_name}: {len(subset)} rows")

    all_df.to_csv(f"{args.tag}_results_all.csv", index=False)

    # Summary with BCa CIs
    summary_rows = []
    for model_name in ["ridge", "rf"]:
        model_df = all_df[all_df["model"] == model_name]
        row = {"model": model_name}
        for col in METRIC_COLS:
            if col in model_df.columns and model_df[col].notna().any():
                vals = model_df[col].dropna().values
                mean, lo, hi = bootstrap_ci_bca(vals)
                row[f"{col}_mean"] = round(mean, 4)
                row[f"{col}_ci_lo"] = round(lo, 4)
                row[f"{col}_ci_hi"] = round(hi, 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.tag}_summary.csv", index=False)

    # Print key metrics
    print(f"\n{'Model':<8} {'Pearson':>8} {'AUROC_sens':>11} {'P@50':>7}")
    print("-" * 38)
    for _, r in summary_df.iterrows():
        print(f"{r['model']:<8} {r.get('pearson_mean', float('nan')):>8.4f} "
              f"{r.get('auroc_sensitizer_mean', float('nan')):>11.4f} "
              f"{r.get('precision_at_50_mean', float('nan')):>7.4f}")


if __name__ == "__main__":
    main()
