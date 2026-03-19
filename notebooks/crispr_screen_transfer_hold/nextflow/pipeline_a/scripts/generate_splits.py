"""Generate reproducible gene holdout splits for Design A.

Outputs:
  split_manifest.csv — one row per split (no gene lists)
  splits/aim1_random_*.json — one JSON per split with train/test gene lists
"""
import argparse
from pathlib import Path

from crispr_al.splits import generate_splits
from crispr_al.io import load_parquet, save_split_manifest, save_split_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-parquet", required=True, help="screen_scores.parquet path")
    parser.add_argument("--n-repeats",   type=int, default=25, help="Number of random splits")
    parser.add_argument("--train-size",  type=int, default=2000, help="Genes per training set")
    parser.add_argument("--output-dir",  default=".", help="Directory for output files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    splits_dir = out_dir / "splits"

    screen_df = load_parquet(args.screen_parquet).reset_index()
    all_genes = screen_df["gene_symbol"].tolist()

    splits = generate_splits(all_genes, n_repeats=args.n_repeats, train_size=args.train_size)

    save_split_manifest(splits, str(out_dir / "split_manifest.csv"))
    save_split_files(splits, str(splits_dir))

    print(f"Generated {len(splits)} splits  |  train={args.train_size}  |  test≈{len(splits[0]['test_genes']):,}")
    print(f"  Manifest: {out_dir / 'split_manifest.csv'}")
    print(f"  Split files: {splits_dir}/  ({len(splits)} JSONs)")
    print(f"  Example hash: {splits[0]['split_hash']}")


if __name__ == "__main__":
    main()
