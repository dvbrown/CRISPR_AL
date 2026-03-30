"""Build the 6 pathway feature matrix for the Olivieri 2020 gene universe.

Reads gene_entrez.parquet (from Step 1) and the pathway annotation files
already in data/bulk/pathway_annotations/, then writes:
  - data/olivieri2020/features_6pathway.parquet  (genes × 6 pathway features)

Note: KEGG counts use KEGG_LEGACY only (KEGG_MEDICUS not available locally),
so n_kegg_pathways values differ slightly from the plan's stated sanity-check
numbers (BRCA1=14, TP53=32 with KEGG_LEGACY; vs ~35/86 with both).
"""
import argparse
import logging
from pathlib import Path

from crispr_al.features import build_olivieri_features
from crispr_al.io import load_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PATHWAY_DIR = Path("data/bulk/pathway_annotations")
REACTOME_PATH = PATHWAY_DIR / "NCBI2Reactome_PE_Pathway.txt.gz"
GOA_PATH = PATHWAY_DIR / "goa_human.gaf.gz"
HALLMARKS_PATH = PATHWAY_DIR / "h.all.v2024.1.Hs.symbols.gmt.gz"
KEGG_PATH = PATHWAY_DIR / "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt.gz"


def main(data_dir: str) -> None:
    data = Path(data_dir)

    gene_entrez = load_parquet(str(data / "gene_entrez.parquet"))
    logger.info("Gene universe: %d genes", len(gene_entrez))

    features = build_olivieri_features(
        gene_entrez_df=gene_entrez,
        reactome_path=str(REACTOME_PATH),
        goa_path=str(GOA_PATH),
        hallmarks_path=str(HALLMARKS_PATH),
        kegg_path=str(KEGG_PATH),
    )
    logger.info("Feature matrix shape: %s", features.shape)

    save_parquet(features, str(data / "features_6pathway.parquet"))
    logger.info("Saved features_6pathway.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data/olivieri2020",
        help="Directory containing olivieri2020 data files (default: data/olivieri2020)",
    )
    args = parser.parse_args()
    main(args.data_dir)
