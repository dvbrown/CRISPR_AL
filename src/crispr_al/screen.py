"""Screen score loading and normalisation for Chen 2019 venetoclax screen."""
import numpy as np
import pandas as pd
from scipy.stats import zscore


def load_screen_scores(path: str) -> pd.DataFrame:
    """Load tab-separated Chen 2019 BioGRID-ORCS screen file.

    Returns DataFrame with columns:
      gene_symbol, entrez_id, cs (SCORE.1), pvalue (SCORE.2)
    Drops rows where OFFICIAL_SYMBOL is empty or SCORE.1 is non-finite.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={
        "OFFICIAL_SYMBOL": "gene_symbol",
        "IDENTIFIER_ID": "entrez_id",
        "SCORE.1": "cs",
        "SCORE.2": "pvalue",
    })
    df = df[["gene_symbol", "entrez_id", "cs", "pvalue"]].copy()
    df = df[df["gene_symbol"].notna() & (df["gene_symbol"] != "")]
    df = df[np.isfinite(df["cs"])]
    df = df.drop_duplicates(subset="gene_symbol").reset_index(drop=True)
    return df


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Fit z-score on ALL genes and add score_norm column."""
    df = df.copy()
    df["score_norm"] = zscore(df["cs"], ddof=0)
    return df


def assign_hit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign hit labels using paper's own CS thresholds (on raw SCORE.1).

    is_sensitizer: cs < -1.0  (more negative = sensitizer)
    is_resistor:   cs > 3.0   (more positive = resistor)
    """
    df = df.copy()
    df["is_hit_sensitizer"] = df["cs"] < -1.0
    df["is_hit_resistor"] = df["cs"] > 3.0
    return df
