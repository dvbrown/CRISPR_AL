"""Build the feature matrix for Yumm1.7 (Elling 2024) using human orthologue gene symbols.

Reads data/elling2024/scores_long.parquet for the gene universe (post-orthologue
mapping) and the shared pathway annotation files in data/bulk/pathway_annotations/.

If ≥70% of genes are covered by DepMap/CCLE, adds expression and co-essentiality
features (using SK-MEL melanoma lines as a proxy; falls back to MOLM-13 if not
available). Otherwise builds pathway-only (6 features) matching the Olivieri setup.

Writes:
  - data/elling2024/features_yumm17.parquet   (genes × features)

Usage:
    source scripts/activate_env.sh
    python scripts/elling2024/03_build_features.py
    python scripts/elling2024/03_build_features.py --data-dir data/elling2024
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from crispr_al.features import (
    build_pathway_features,
    build_expression_feature,
    build_coessentiality_features,
    assemble_gene_features,
)
from crispr_al.io import load_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PATHWAY_DIR = Path("data/bulk/pathway_annotations")
REACTOME_PATH = PATHWAY_DIR / "NCBI2Reactome_PE_Pathway.txt.gz"
GOA_PATH = PATHWAY_DIR / "goa_human.gaf.gz"
HALLMARKS_PATH = PATHWAY_DIR / "h.all.v2024.1.Hs.symbols.gmt.gz"
KEGG_PATH = PATHWAY_DIR / "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt.gz"

CCLE_PATH = Path("data/bulk/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz")
DEPMAP_PATH = Path("data/bulk/depmap/CRISPRGeneEffect.csv.gz")

# SK-MEL melanoma lines in DepMap (proxy for Yumm1.7 mouse melanoma)
SKMEL_LINE_ID = "ACH-000215"   # SK-MEL-28; fall back to MOLM-13 if absent
FALLBACK_LINE_ID = "ACH-000362"   # MOLM-13

DEPMAP_COVERAGE_THRESHOLD = 0.70


def _get_gene_entrez_df(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Extract unique gene_symbol list from the harmonised scores table."""
    genes = scores_long["gene_symbol"].dropna().unique()
    # entrez_id not available for mapped human symbols; leave as NA
    return pd.DataFrame({"gene_symbol": genes, "entrez_id": pd.NA})


def main(data_dir: Path) -> None:
    scores_long = load_parquet(str(data_dir / "scores_long.parquet"))
    gene_df = _get_gene_entrez_df(scores_long)
    screen_genes = gene_df["gene_symbol"].tolist()
    log.info("Gene universe: %d genes", len(screen_genes))

    # --- Pathway features (always built) ---
    pathway_df = build_pathway_features(
        reactome_path=str(REACTOME_PATH),
        goa_path=str(GOA_PATH),
        hallmarks_path=str(HALLMARKS_PATH),
        kegg_path=str(KEGG_PATH),
        screen_df=gene_df,
    )
    log.info("Pathway features shape: %s", pathway_df.shape)

    # --- Decide whether to include expression + co-essentiality ---
    include_depmap = False
    if CCLE_PATH.exists() and DEPMAP_PATH.exists():
        try:
            depmap_genes_raw = pd.read_csv(DEPMAP_PATH, index_col=0, nrows=0).columns
            import re
            depmap_genes = {re.sub(r"\s*\(\d+\)\s*$", "", c).strip() for c in depmap_genes_raw}
            coverage = sum(1 for g in screen_genes if g in depmap_genes) / max(len(screen_genes), 1)
            log.info("DepMap coverage: %.1f%%", 100 * coverage)
            if coverage >= DEPMAP_COVERAGE_THRESHOLD:
                include_depmap = True
                log.info("Coverage ≥70%% — including expression and co-essentiality features")
            else:
                log.info("Coverage <70%% — using pathway-only features")
        except Exception as exc:
            log.warning("Could not assess DepMap coverage: %s. Using pathway-only.", exc)
    else:
        log.info("DepMap/CCLE files not found. Using pathway-only features.")

    if include_depmap:
        # Prefer SK-MEL line; fall back to MOLM-13
        try:
            expr = build_expression_feature(str(CCLE_PATH), molm13_id=SKMEL_LINE_ID)
            log.info("Using SK-MEL line %s for expression", SKMEL_LINE_ID)
        except Exception:
            expr = build_expression_feature(str(CCLE_PATH), molm13_id=FALLBACK_LINE_ID)
            log.info("SK-MEL line unavailable; using MOLM-13 as expression proxy")

        coess_df = build_coessentiality_features(
            str(DEPMAP_PATH),
            screen_genes=screen_genes,
            molm13_id=SKMEL_LINE_ID,
        )
        features = assemble_gene_features(
            screen_genes=screen_genes,
            expr_series=expr,
            coess_df=coess_df,
            pathway_df=pathway_df,
        )
    else:
        features = pathway_df.reindex(screen_genes).fillna(0.0)

    log.info("Final feature matrix: %s", features.shape)
    log.info("Feature columns: %s", list(features.columns))

    save_parquet(features, str(data_dir / "features_yumm17.parquet"))
    log.info("Saved features_yumm17.parquet")
    log.info("Next step: python scripts/elling2024/04_agreement_metrics.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", default="data/elling2024",
        help="Directory containing elling2024 data files (default: data/elling2024)",
    )
    args = parser.parse_args()
    main(Path(args.data_dir))
