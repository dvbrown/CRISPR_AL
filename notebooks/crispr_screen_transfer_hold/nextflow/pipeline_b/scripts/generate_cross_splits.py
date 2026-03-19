"""Generate cross-screen transfer splits for Design B.

Two directions:
  chen→sharon  Seeds 21001–21030 (30 splits)
  sharon→chen  Seeds 22001–22030 (30 splits)

Outputs:
  cross_split_manifest.csv            — one row per split (no gene lists)
  splits/aim1_xfer_*.json             — one JSON per split with train/test gene lists
"""
import argparse
from pathlib import Path

from crispr_al.splits import (
    generate_cross_screen_splits,
    XFER_SEED_START,
    XFER_SEED_START_REVERSE,
)
from crispr_al.io import load_parquet, save_split_manifest, save_split_files

CHEN_SCREEN_ID = "chen2019_1393"
SHARON_SCREEN_ID = "sharon2019_1402"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chen-screen",  required=True,
                        help="chen_scores.parquet path")
    parser.add_argument("--sharon-screen", required=True,
                        help="sharon_scores.parquet path")
    parser.add_argument("--n-repeats",  type=int, default=30,
                        help="Number of splits per direction (default: 30)")
    parser.add_argument("--train-size", type=int, default=2000,
                        help="Genes per training set (default: 2000)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    splits_dir = out_dir / "splits"

    chen_df = load_parquet(args.chen_screen).reset_index()
    sharon_df = load_parquet(args.sharon_screen).reset_index()

    chen_genes = chen_df["gene_symbol"].tolist()
    sharon_genes = sharon_df["gene_symbol"].tolist()

    # Chen → Sharon
    splits_c2s = generate_cross_screen_splits(
        train_genes=chen_genes,
        test_genes=sharon_genes,
        train_screen_id=CHEN_SCREEN_ID,
        test_screen_id=SHARON_SCREEN_ID,
        n_repeats=args.n_repeats,
        train_size=args.train_size,
        seed_start=XFER_SEED_START,
    )

    # Sharon → Chen
    splits_s2c = generate_cross_screen_splits(
        train_genes=sharon_genes,
        test_genes=chen_genes,
        train_screen_id=SHARON_SCREEN_ID,
        test_screen_id=CHEN_SCREEN_ID,
        n_repeats=args.n_repeats,
        train_size=args.train_size,
        seed_start=XFER_SEED_START_REVERSE,
    )

    all_splits = splits_c2s + splits_s2c

    save_split_manifest(all_splits, str(out_dir / "cross_split_manifest.csv"))
    save_split_files(all_splits, str(splits_dir))

    print(f"Generated {len(all_splits)} cross-screen splits  "
          f"({len(splits_c2s)} chen→sharon  +  {len(splits_s2c)} sharon→chen)")
    print(f"  Manifest: {out_dir / 'cross_split_manifest.csv'}")
    print(f"  Split files: {splits_dir}/")
    print(f"  Example hash (c→s): {splits_c2s[0]['split_hash']}")
    print(f"  Example hash (s→c): {splits_s2c[0]['split_hash']}")


if __name__ == "__main__":
    main()
