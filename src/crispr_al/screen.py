"""Screen score loading and normalisation for Chen 2019 and Sharon 2019 venetoclax screens."""
import logging

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)


def _load_biogrid_screen(
    path: str,
    column_mapping: dict,
    output_cols: list,
    score_filter_col: str,
    log_duplicates: bool = False,
) -> pd.DataFrame:
    """Load and clean a BioGRID-ORCS screen TSV file.

    Applies column rename, drops rows with empty gene_symbol or non-finite
    score_filter_col, and deduplicates on gene_symbol (keep first occurrence).
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns=column_mapping)
    df = df[output_cols].copy()
    df = df[df["gene_symbol"].notna() & (df["gene_symbol"] != "")]
    df = df[np.isfinite(df[score_filter_col])]
    n_before = len(df)
    df = df.drop_duplicates(subset="gene_symbol").reset_index(drop=True)
    if log_duplicates:
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            logger.info("Dropped %d duplicate gene_symbol rows (kept first)", n_dupes)
    return df


def load_screen_scores(path: str) -> pd.DataFrame:
    """Load tab-separated Chen 2019 BioGRID-ORCS screen file.

    Returns DataFrame with columns:
      gene_symbol, entrez_id, cs (SCORE.1), pvalue (SCORE.2)
    Drops rows where OFFICIAL_SYMBOL is empty or SCORE.1 is non-finite.
    Keeps first occurrence of duplicate gene_symbol.
    """
    return _load_biogrid_screen(
        path,
        column_mapping={
            "OFFICIAL_SYMBOL": "gene_symbol",
            "IDENTIFIER_ID": "entrez_id",
            "SCORE.1": "cs",
            "SCORE.2": "pvalue",
        },
        output_cols=["gene_symbol", "entrez_id", "cs", "pvalue"],
        score_filter_col="cs",
    )


def load_sharon_screen_scores(path: str) -> pd.DataFrame:
    """Load tab-separated Sharon 2019 BioGRID-ORCS screen file (screen 1402).

    Returns DataFrame with columns:
      gene_symbol, entrez_id, lfc (SCORE.5), neg_fdr (SCORE.2), pos_fdr (SCORE.4)
    Drops rows where OFFICIAL_SYMBOL is empty or SCORE.5 is not a finite float.
    Keeps first occurrence of duplicate gene_symbol and logs the duplicate count.
    """
    return _load_biogrid_screen(
        path,
        column_mapping={
            "OFFICIAL_SYMBOL": "gene_symbol",
            "IDENTIFIER_ID": "entrez_id",
            "SCORE.2": "neg_fdr",
            "SCORE.4": "pos_fdr",
            "SCORE.5": "lfc",
        },
        output_cols=["gene_symbol", "entrez_id", "lfc", "neg_fdr", "pos_fdr"],
        score_filter_col="lfc",
        log_duplicates=True,
    )


def load_olivieri_normz(path: str) -> pd.DataFrame:
    """Load the Olivieri 2020 NormZ matrix (genes × screens).

    Returns wide DataFrame with gene symbols as index and screen labels as columns.
    Scores are DrugZ NormZ (Z-score; negative = sensitising KO).
    """
    return pd.read_parquet(path)


def zscore_normalize(df: pd.DataFrame, score_col: str = "cs") -> pd.DataFrame:
    """Fit z-score on ALL genes and add score_norm column.

    This is screen-level harmonization applied once to the full screen before
    split generation. It is not a leakage risk. The leakage_checks field
    normalization_fit_on_train_only in metrics records refers to the
    StandardScaler fitted on X_train during model training, not this step.

    Args:
        df: Screen scores DataFrame.
        score_col: Column to z-score. Default "cs" (Chen). Pass "lfc" for Sharon.
    """
    df = df.copy()
    df["score_norm"] = zscore(df[score_col], ddof=0)
    return df


def assign_hit_labels_zscore(df: pd.DataFrame, threshold: float = 1.645) -> pd.DataFrame:
    """Assign hit labels using z-score thresholds on score_norm (~5% tails).

    is_hit_sensitizer: score_norm < -threshold  (bottom ~5%)
    is_hit_resistor:   score_norm > +threshold  (top ~5%)

    Used for both Chen and Sharon screens. Requires zscore_normalize to have
    been called first to produce the score_norm column.
    """
    df = df.copy()
    df["is_hit_sensitizer"] = df["score_norm"] < -threshold
    df["is_hit_resistor"] = df["score_norm"] > threshold
    return df


# Keep old name as alias so any existing callsites get the updated behaviour.
def assign_hit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign hit labels using z-score ±1.645 on score_norm.

    Delegates to assign_hit_labels_zscore. The previous paper-threshold
    implementation (cs < -1.0, cs > 3.0) has been superseded; all designs
    now use consistent z-score thresholds for cross-screen comparability.
    """
    return assign_hit_labels_zscore(df)
