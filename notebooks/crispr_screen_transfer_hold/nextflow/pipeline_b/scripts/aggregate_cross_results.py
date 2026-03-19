"""Aggregate Design B cross-screen metrics row CSVs into per-direction summary tables.

Reads all *_row.csv files in the current directory (collected by Nextflow).
Partitions by direction (chen_to_sharon / sharon_to_chen) and model.

Outputs:
  design_b_results_chen_to_sharon.csv     — all rows, chen→sharon direction
  design_b_results_sharon_to_chen.csv     — all rows, sharon→chen direction
  design_b_results_all.csv                — all rows combined
  design_b_summary.csv                    — mean ± BCa 95% CI per direction × model
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

DIRECTIONS = ["chen_to_sharon", "sharon_to_chen"]
MODELS     = ["ridge", "rf", "overlap_baseline"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="design_b", help="Output filename prefix")
    args = parser.parse_args()

    row_files = sorted(glob.glob("*_row.csv"))
    if not row_files:
        raise FileNotFoundError("No *_row.csv files found in current directory")

    all_df = pd.concat([pd.read_csv(f) for f in row_files], ignore_index=True)
    print(f"Loaded {len(row_files)} row files  →  {len(all_df)} total rows")

    # Infer direction from split_id if direction column absent
    if "direction" not in all_df.columns:
        def _infer_direction(split_id: str) -> str:
            if "chen2019" in split_id and "sharon2019" in split_id:
                parts = split_id.split("_to_")
                if len(parts) == 2:
                    return ("chen_to_sharon" if "chen" in parts[0] else "sharon_to_chen")
            return "unknown"
        all_df["direction"] = all_df["split_id"].apply(_infer_direction)

    # Per-direction CSVs
    for direction in DIRECTIONS:
        subset = all_df[all_df["direction"] == direction].copy()
        subset.to_csv(f"{args.tag}_results_{direction}.csv", index=False)
        print(f"  {direction}: {len(subset)} rows")

    all_df.to_csv(f"{args.tag}_results_all.csv", index=False)

    # Summary with BCa CIs — per direction × model
    summary_rows = []
    for direction in DIRECTIONS:
        dir_df = all_df[all_df["direction"] == direction]
        for model_name in MODELS:
            model_df = dir_df[dir_df["model"] == model_name]
            if model_df.empty:
                continue
            row = {"direction": direction, "model": model_name}
            for col in METRIC_COLS:
                if col in model_df.columns and model_df[col].notna().any():
                    vals = model_df[col].dropna().values
                    if len(vals) >= 2:
                        mean, lo, hi = bootstrap_ci_bca(vals)
                        row[f"{col}_mean"] = round(mean, 4)
                        row[f"{col}_ci_lo"] = round(lo, 4)
                        row[f"{col}_ci_hi"] = round(hi, 4)
                    else:
                        row[f"{col}_mean"] = round(float(vals[0]), 4)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.tag}_summary.csv", index=False)

    # Print key metrics
    print(f"\n{'Direction':<20} {'Model':<18} {'Pearson':>8} {'AUROC_sens':>11} {'P@50':>7}")
    print("-" * 68)
    for _, r in summary_df.iterrows():
        print(f"{r['direction']:<20} {r['model']:<18} "
              f"{r.get('pearson_mean', float('nan')):>8.4f} "
              f"{r.get('auroc_sensitizer_mean', float('nan')):>11.4f} "
              f"{r.get('precision_at_50_mean', float('nan')):>7.4f}")


if __name__ == "__main__":
    main()
