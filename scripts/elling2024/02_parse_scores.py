"""Compute gene-level scores from CRISPR-StAR guide UMI counts (Elling 2024).

CRISPR-StAR delivers paired "active" / "inactive" files per sample. Each file
contains guide-level UMI × read counts. Gene scores are computed as:
  1. Sum reads per guide per context (pooling replicates Blue/Green/Red within
     in_vitro and in_vivo).
  2. Guide LFC = log2((active_reads + 1) / (inactive_reads + 1)).
  3. Gene score = median guide LFC (guides named <MouseSymbol>_<index>).
  4. Z-score normalise within each context.
  5. Map mouse gene symbols → human orthologues via Ensembl BioMart one-to-one
     table (cached at data/bulk/human_mouse_orthologs.tsv).

Writes:
  - data/elling2024/scores_long.parquet
  - data/elling2024/screen_metadata.parquet

Usage:
    source scripts/activate_env.sh
    python scripts/elling2024/02_parse_scores.py
    python scripts/elling2024/02_parse_scores.py --raw-dir data/elling2024/raw --out-dir data/elling2024
"""
import argparse
import gzip
import logging
import re
import tarfile
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from crispr_al.features import map_mouse_to_human_orthologues
from crispr_al.io import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/elling2024/raw")
DEFAULT_OUT_DIR = Path("data/elling2024")
ORTHO_CACHE = Path("data/bulk/human_mouse_orthologs.tsv")
PSEUDOCOUNT = 1.0
MIN_GUIDES = 2  # minimum guides per gene to keep the gene score


# ---------------------------------------------------------------------------
# Orthologue fetching
# ---------------------------------------------------------------------------

def fetch_orthologue_table() -> pd.DataFrame:
    """Load or download human↔mouse one-to-one orthologue table."""
    if ORTHO_CACHE.exists():
        log.info("Loading orthologue cache: %s", ORTHO_CACHE)
        df = pd.read_csv(ORTHO_CACHE, sep="\t")
    else:
        log.info("Downloading orthologue table from Ensembl BioMart")
        query = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<!DOCTYPE Query>"
            "<Query virtualSchemaName='default' formatter='TSV' header='1' "
            "uniqueRows='1' count='' datasetConfigVersion='0.6'>"
            "<Dataset name='hsapiens_gene_ensembl' interface='default'>"
            "<Attribute name='external_gene_name'/>"
            "<Attribute name='mmusculus_homolog_associated_gene_name'/>"
            "<Attribute name='mmusculus_homolog_orthology_type'/>"
            "</Dataset></Query>"
        )
        url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote(query)
        with urllib.request.urlopen(url, timeout=180) as resp:
            content = resp.read().decode("utf-8")
        lines = content.strip().splitlines()
        rows = [line.split("\t") for line in lines[1:] if line]
        df = pd.DataFrame(rows, columns=["human_symbol", "mouse_symbol", "orthology_type"])
        ORTHO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ORTHO_CACHE, sep="\t", index=False)
        log.info("Saved orthologue cache: %s", ORTHO_CACHE)

    # Normalise column names from varying BioMart headers
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if ("gene name" in cl or "gene_name" in cl) and "mouse" not in cl and "mmus" not in cl:
            rename[c] = "human_symbol"
        elif "mouse" in cl or "mmus" in cl:
            if "associated" in cl or "gene" in cl:
                rename[c] = "mouse_symbol"
        elif "orthology" in cl or "type" in cl:
            rename[c] = "orthology_type"
    if rename:
        df = df.rename(columns=rename)
    if "human_symbol" not in df.columns:
        df.columns = ["human_symbol", "mouse_symbol", "orthology_type"]

    one2one = df[
        df["orthology_type"].str.contains("one2one", case=False, na=False)
    ].copy()
    one2one = one2one[
        one2one["human_symbol"].notna() & (one2one["human_symbol"] != "") &
        one2one["mouse_symbol"].notna() & (one2one["mouse_symbol"] != "")
    ].drop_duplicates(subset=["mouse_symbol"])
    log.info("One-to-one orthologues: %d", len(one2one))
    return one2one[["human_symbol", "mouse_symbol"]]


# ---------------------------------------------------------------------------
# Guide count loading
# ---------------------------------------------------------------------------

def _load_guide_reads(gz_file) -> pd.Series:
    """Sum reads per guide from an open gzip file object.

    Returns Series: guide_name → total_reads.
    """
    counts: dict[str, int] = defaultdict(int)
    for line in gz_file:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line or line.startswith("guide"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        guide = parts[0]
        try:
            reads = int(parts[3])
        except ValueError:
            continue
        counts[guide] += reads
    return pd.Series(counts, name="reads")


def _context_of(fname: str) -> str:
    """Return 'in_vitro' or 'in_vivo' from a filename."""
    f = fname.lower()
    if "in_vitro" in f or "invitro" in f:
        return "in_vitro"
    if "in_vivo" in f or "invivo" in f:
        return "in_vivo"
    return "unknown"


def _state_of(fname: str) -> str:
    """Return 'active' or 'inactive' from a filename."""
    f = fname.lower()
    if "_active_" in f:
        return "active"
    if "_inactive_" in f:
        return "inactive"
    return "unknown"


def load_all_counts(raw_dir: Path) -> pd.DataFrame:
    """Load guide-level read counts from all samples.

    Returns DataFrame with columns: guide, context, state, reads.
    Reads are summed across UMIs and pooled across replicate colours.
    """
    gz_files = sorted(raw_dir.glob("*.txt.gz"))
    tar_files = sorted(raw_dir.glob("*.tar"))

    log.info("Found %d .txt.gz files and %d .tar files", len(gz_files), len(tar_files))

    # Build list of (filename, open_fn) tuples to iterate
    file_specs: list[tuple[str, object]] = []

    for gz in gz_files:
        if "noShadows" in gz.name or "active" in gz.name.lower():
            file_specs.append((gz.name, None))   # path-based

    # If no extracted files, read from tar directly
    if not gz_files and tar_files:
        log.info("No extracted files; reading directly from tar")

    records: list[dict] = []

    def _process_file(fname: str, gz_obj) -> None:
        ctx = _context_of(fname)
        state = _state_of(fname)
        if ctx == "unknown" or state == "unknown":
            return
        reads = _load_guide_reads(gz_obj)
        for guide, n in reads.items():
            records.append({"guide": guide, "context": ctx, "state": state, "reads": n})

    if gz_files:
        for gz in gz_files:
            ctx = _context_of(gz.name)
            state = _state_of(gz.name)
            if ctx == "unknown" or state == "unknown":
                continue
            with gzip.open(gz, "rt") as f:
                reads = _load_guide_reads(f)
            for guide, n in reads.items():
                records.append({"guide": guide, "context": ctx, "state": state, "reads": n})
    elif tar_files:
        for tar_path in tar_files:
            with tarfile.open(tar_path) as tf:
                for member in tf.getmembers():
                    ctx = _context_of(member.name)
                    state = _state_of(member.name)
                    if ctx == "unknown" or state == "unknown":
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    with gzip.open(fobj, "rt") as gz:
                        reads = _load_guide_reads(gz)
                    for guide, n in reads.items():
                        records.append({"guide": guide, "context": ctx, "state": state, "reads": n})

    df = pd.DataFrame(records)
    log.info("Loaded %d guide×context×state records", len(df))
    return df


# ---------------------------------------------------------------------------
# Gene score computation
# ---------------------------------------------------------------------------

def compute_gene_scores(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute gene-level LFC scores from guide-level active/inactive counts.

    Steps:
    1. Pool reads: sum across replicates per (guide, context, state).
    2. Pivot to (guide, context) × {active, inactive}.
    3. Guide LFC = log2((active + PSEUDOCOUNT) / (inactive + PSEUDOCOUNT)).
    4. Extract mouse gene symbol from guide name (pattern: Symbol_N).
    5. Gene score = median of guide LFCs (per gene, per context).
    6. Discard genes with fewer than MIN_GUIDES guides.

    Returns DataFrame: mouse_symbol, context, score_raw, n_guides.
    """
    # Sum reads across replicates
    pooled = (
        counts_df
        .groupby(["guide", "context", "state"], as_index=False)["reads"]
        .sum()
    )
    # Pivot active/inactive
    wide = pooled.pivot_table(
        index=["guide", "context"], columns="state", values="reads", fill_value=0
    ).reset_index()
    wide.columns.name = None
    for col in ("active", "inactive"):
        if col not in wide.columns:
            wide[col] = 0

    # Guide LFC
    wide["guide_lfc"] = np.log2(
        (wide["active"] + PSEUDOCOUNT) / (wide["inactive"] + PSEUDOCOUNT)
    )

    # Extract gene symbol: everything before the last underscore+digits
    wide["mouse_symbol"] = wide["guide"].str.replace(r"_\d+$", "", regex=True)

    # Aggregate to gene level
    gene_scores = (
        wide.groupby(["mouse_symbol", "context"])
        .agg(
            score_raw=("guide_lfc", "median"),
            n_guides=("guide_lfc", "count"),
        )
        .reset_index()
    )
    before = len(gene_scores)
    gene_scores = gene_scores[gene_scores["n_guides"] >= MIN_GUIDES].copy()
    log.info("Gene scores: %d rows (%d dropped for <=%d guides)", len(gene_scores), before - len(gene_scores), MIN_GUIDES - 1)
    return gene_scores


# ---------------------------------------------------------------------------
# Orthologue mapping + normalisation
# ---------------------------------------------------------------------------

def apply_orthologue_mapping(gene_scores: pd.DataFrame, ortho: pd.DataFrame) -> pd.DataFrame:
    """Map mouse symbols → human orthologues and report coverage."""
    mouse_genes = gene_scores["mouse_symbol"].unique().tolist()
    mapped = map_mouse_to_human_orthologues(mouse_genes, ortho)
    result = gene_scores.merge(mapped, on="mouse_symbol", how="inner")
    # Deduplicate on human_symbol × context (take first)
    result = result.drop_duplicates(subset=["human_symbol", "context"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ortho = fetch_orthologue_table()

    # Load guide counts
    counts_df = load_all_counts(raw_dir)
    if counts_df.empty:
        raise RuntimeError(f"No count data loaded from {raw_dir}. Run 01_download.py first.")

    # Compute gene scores
    gene_scores = compute_gene_scores(counts_df)

    # Z-score normalise within each context
    gene_scores["score_norm"] = gene_scores.groupby("context")["score_raw"].transform(
        lambda x: zscore(x, ddof=0) if len(x) > 1 else x
    )

    # Map to human orthologues
    gene_scores = apply_orthologue_mapping(gene_scores, ortho)

    # Build scores_long
    scores_long = gene_scores.rename(columns={"human_symbol": "gene_symbol"})
    scores_long["screen_id"] = scores_long["context"].map(
        {"in_vitro": "elling2024_invitro", "in_vivo": "elling2024_invivo"}
    )
    scores_long = scores_long[[
        "gene_symbol", "mouse_symbol", "screen_id", "context", "score_raw", "score_norm"
    ]].reset_index(drop=True)

    # Sanity checks
    n_vitro = scores_long[scores_long["context"] == "in_vitro"]["gene_symbol"].nunique()
    n_vivo = scores_long[scores_long["context"] == "in_vivo"]["gene_symbol"].nunique()
    overlap = len(
        set(scores_long[scores_long["context"] == "in_vitro"]["gene_symbol"])
        & set(scores_long[scores_long["context"] == "in_vivo"]["gene_symbol"])
    )
    log.info("In vitro genes: %d | In vivo genes: %d | Overlap: %d", n_vitro, n_vivo, overlap)
    if overlap < 5_000:
        log.warning("Gene overlap (%d) below 5,000 — check data parsing.", overlap)

    # Screen metadata
    screen_metadata = pd.DataFrame([
        {"screen_id": "elling2024_invitro", "context": "in_vitro", "n_genes": n_vitro},
        {"screen_id": "elling2024_invivo", "context": "in_vivo", "n_genes": n_vivo},
    ])

    save_parquet(scores_long, str(out_dir / "scores_long.parquet"))
    save_parquet(screen_metadata, str(out_dir / "screen_metadata.parquet"))
    log.info("Saved scores_long.parquet (%d rows)", len(scores_long))
    log.info("Next step: python scripts/elling2024/03_build_features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Raw GEO download directory")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for parquet files")
    args = parser.parse_args()
    main(Path(args.raw_dir), Path(args.out_dir))
