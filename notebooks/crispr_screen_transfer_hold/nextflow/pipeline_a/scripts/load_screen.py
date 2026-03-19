"""Load and normalise the Chen 2019 venetoclax CRISPR screen.

Outputs screen_scores.parquet indexed by gene_symbol with columns:
  entrez_id, cs, pvalue, score_norm, is_hit_sensitizer, is_hit_resistor
"""
import argparse

from crispr_al.screen import load_screen_scores, zscore_normalize, assign_hit_labels
from crispr_al.io import save_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-path", required=True, help="Path to BioGRID-ORCS screen TSV")
    parser.add_argument("--output", default="screen_scores.parquet", help="Output parquet path")
    args = parser.parse_args()

    df = load_screen_scores(args.screen_path)
    df = zscore_normalize(df)
    df = assign_hit_labels(df)
    df = df.set_index("gene_symbol")
    save_parquet(df, args.output)

    n_sens = df["is_hit_sensitizer"].sum()
    n_res = df["is_hit_resistor"].sum()
    print(f"Saved {len(df):,} genes → {args.output}")
    print(f"  Sensitizers (bottom 5%): {n_sens:,}  |  Resistors (top 5%): {n_res:,}")


if __name__ == "__main__":
    main()
