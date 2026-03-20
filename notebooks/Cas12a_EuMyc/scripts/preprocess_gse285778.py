"""Preprocess GSE285778 (Menuetto/Scherzo/iMDF) CRISPR screen data.

Steps:
  1. Download raw count files from GEO FTP  (skipped if files exist)
  2. Run MAGeCK test for each comparison     (skipped if outputs exist)
  3. Parse MAGeCK gene_summary files → gene score parquets
  4. Fetch mouse→human ortholog map          (skipped if cached)
  5. Build feature matrix (Chronos, CCLE, Reactome binary, Hallmark binary,
     essential flag, co-essentiality PCA)
  6. Save all processed parquets

Usage:
  source scripts/activate_env.sh
  python notebooks/Cas12a_EuMyc/scripts/preprocess_gse285778.py [--dry-run]

Options:
  --dry-run   Print MAGeCK commands instead of running them
  --skip-features   Skip feature matrix build (useful if only re-running MAGeCK)
"""
import argparse
import gzip
import io
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parents[3]
DATA_ROOT  = ROOT / "data" / "bulk"
RAW_DIR    = DATA_ROOT / "menuetto_scherzo_2025" / "raw"
PROC_DIR   = DATA_ROOT / "menuetto_scherzo_2025" / "processed"

DEPMAP_PATH    = DATA_ROOT / "depmap_crispr_gene_effect" / "CRISPRGeneEffect.csv.gz"
CCLE_PATH      = DATA_ROOT / "ccle_expression" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz"
REACTOME_PATH  = DATA_ROOT / "pathway_annotations" / "NCBI2Reactome_PE_Pathway.txt.gz"
HALLMARKS_PATH = DATA_ROOT / "pathway_annotations" / "h.all.v2024.1.Hs.symbols.gmt.gz"

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285778/suppl/"
GEO_FILES = [
    "GSE285778_EuMycCountMenuetto.txt.gz",
    "GSE285778_EuMycCountScherzo.txt.gz",
    "GSE285778_iMDFCount.txt.gz",
    "GSE285778_InVivoCount.txt.gz",
]

# GSE285778 sample name conventions:
#   Menuetto (Dual):  D{rep}{cond}   rep=1-6  cond: i=Input d=DMSO n=Nutlin s=S63845
#   Scherzo  (Quad):  Q{rep}{cond}   same
#   iMDF:             T{tp}-{rep}    tp=0-3   rep=1-3
MENUETTO_COND_SUFFIX = {"i": "input", "d": "dmso", "n": "nutlin", "s": "s63845"}
SCHERZO_COND_SUFFIX  = MENUETTO_COND_SUFFIX


def _grep_suffix(samples, suffix):
    """Return samples whose name ends with the given single character."""
    return [s for s in samples if s.endswith(suffix)]


def _grep_timepoint(samples, tp):
    """Return iMDF samples matching timepoint prefix, e.g. 'T0'."""
    return [s for s in samples if s.startswith(f"T{tp}-") or s.startswith(f"T{tp}_")]


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_raw_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in GEO_FILES:
        dest = RAW_DIR / fname
        if dest.exists():
            print(f"  exists  {fname}  ({dest.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  downloading {fname} ...")
            urllib.request.urlretrieve(GEO_BASE + fname, dest)
            print(f"  saved   {fname}  ({dest.stat().st_size / 1024:.0f} KB)")


# ── Step 2: Detect sample columns ────────────────────────────────────────────

def read_header(path):
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
    # cols[0]=sgRNA, cols[1]=Gene, cols[2:]=samples
    return cols[2:]


def get_sample_groups():
    """Return dict mapping comparison name → (count_file, treatment_cols, control_cols)."""
    menuetto_samples = read_header(RAW_DIR / "GSE285778_EuMycCountMenuetto.txt.gz")
    scherzo_samples  = read_header(RAW_DIR / "GSE285778_EuMycCountScherzo.txt.gz")
    imdf_samples     = read_header(RAW_DIR / "GSE285778_iMDFCount.txt.gz")

    jobs = {
        "menuetto_nutlin_vs_input": (
            RAW_DIR / "GSE285778_EuMycCountMenuetto.txt.gz",
            _grep_suffix(menuetto_samples, "n"),
            _grep_suffix(menuetto_samples, "i"),
        ),
        "menuetto_s63845_vs_input": (
            RAW_DIR / "GSE285778_EuMycCountMenuetto.txt.gz",
            _grep_suffix(menuetto_samples, "s"),
            _grep_suffix(menuetto_samples, "i"),
        ),
        "menuetto_dmso_vs_input": (
            RAW_DIR / "GSE285778_EuMycCountMenuetto.txt.gz",
            _grep_suffix(menuetto_samples, "d"),
            _grep_suffix(menuetto_samples, "i"),
        ),
        "scherzo_nutlin_vs_input": (
            RAW_DIR / "GSE285778_EuMycCountScherzo.txt.gz",
            _grep_suffix(scherzo_samples, "n"),
            _grep_suffix(scherzo_samples, "i"),
        ),
        "scherzo_s63845_vs_input": (
            RAW_DIR / "GSE285778_EuMycCountScherzo.txt.gz",
            _grep_suffix(scherzo_samples, "s"),
            _grep_suffix(scherzo_samples, "i"),
        ),
        "scherzo_dmso_vs_input": (
            RAW_DIR / "GSE285778_EuMycCountScherzo.txt.gz",
            _grep_suffix(scherzo_samples, "d"),
            _grep_suffix(scherzo_samples, "i"),
        ),
        "imdf_t1_vs_t0": (
            RAW_DIR / "GSE285778_iMDFCount.txt.gz",
            _grep_timepoint(imdf_samples, 1),
            _grep_timepoint(imdf_samples, 0),
        ),
        "imdf_t2_vs_t0": (
            RAW_DIR / "GSE285778_iMDFCount.txt.gz",
            _grep_timepoint(imdf_samples, 2),
            _grep_timepoint(imdf_samples, 0),
        ),
        "imdf_t3_vs_t0": (
            RAW_DIR / "GSE285778_iMDFCount.txt.gz",
            _grep_timepoint(imdf_samples, 3),
            _grep_timepoint(imdf_samples, 0),
        ),
    }
    # Validate
    for name, (_, treat, ctrl) in jobs.items():
        if not treat or not ctrl:
            print(f"  WARNING: {name} — could not detect treatment ({treat}) or control ({ctrl})")
        else:
            print(f"  {name}: {len(treat)} treatment, {len(ctrl)} control")
    return jobs


# ── Step 3: MAGeCK ────────────────────────────────────────────────────────────

def run_mageck_jobs(jobs, dry_run=False):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    mageck_ok = subprocess.run(["which", "mageck"], capture_output=True).returncode == 0
    if not mageck_ok and not dry_run:
        print("  ERROR: mageck not in PATH. Activate the mageck env first:")
        print("  micromamba activate .micromamba/envs/mageck")
        sys.exit(1)

    for name, (count_file, treat, ctrl) in jobs.items():
        out = PROC_DIR / f"{name}.gene_summary.txt"
        if out.exists():
            print(f"  skip (exists) {name}")
            continue
        if not treat or not ctrl:
            print(f"  skip (no samples detected) {name}")
            continue

        cmd = [
            "mageck", "test",
            "--count-table",     str(count_file),
            "--treatment-id",    ",".join(treat),
            "--control-id",      ",".join(ctrl),
            "--output-prefix",   str(PROC_DIR / name),
            "--gene-lfc-method", "median",
            "--adjust-method",   "fdr",
            "--norm-method",     "median",
        ]
        if dry_run:
            print("  (dry run) " + " ".join(cmd))
        else:
            print(f"  running {name} ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[-500:]}")
            else:
                print(f"  done → {out.name}")


# ── Step 4: Parse MAGeCK → parquets ──────────────────────────────────────────

def parse_mageck_outputs():
    def _load(prefix):
        p = PROC_DIR / f"{prefix}.gene_summary.txt"
        if not p.exists():
            return None
        df = pd.read_csv(p, sep="\t", index_col=0)
        df.index.name = "gene_symbol"
        return df

    def _extract(prefix, cond):
        df = _load(prefix)
        if df is None:
            return None
        return pd.DataFrame({
            f"lfc_{cond}":     df["neg|lfc"],
            f"fdr_neg_{cond}": df["neg|fdr"],
            f"fdr_pos_{cond}": df["pos|fdr"],
        })

    # Menuetto
    parts = {c: _extract(f"menuetto_{c}_vs_input", c) for c in ["nutlin", "s63845", "dmso"]}
    parts = {k: v for k, v in parts.items() if v is not None}
    if parts:
        menuetto_scores = pd.concat(parts.values(), axis=1)
        out = PROC_DIR / "menuetto_gene_scores.parquet"
        menuetto_scores.to_parquet(out)
        print(f"  menuetto_gene_scores: {menuetto_scores.shape} → {out.name}")
    else:
        print("  menuetto_gene_scores: MAGeCK outputs missing")

    # Scherzo
    parts = {c: _extract(f"scherzo_{c}_vs_input", c) for c in ["nutlin", "s63845", "dmso"]}
    parts = {k: v for k, v in parts.items() if v is not None}
    if parts:
        scherzo_scores = pd.concat(parts.values(), axis=1)
        out = PROC_DIR / "scherzo_gene_scores.parquet"
        scherzo_scores.to_parquet(out)
        print(f"  scherzo_gene_scores: {scherzo_scores.shape} → {out.name}")
    else:
        print("  scherzo_gene_scores: MAGeCK outputs missing")

    # iMDF
    parts = {tp: _extract(f"imdf_{tp}_vs_t0", tp) for tp in ["t1", "t2", "t3"]}
    parts = {k: v for k, v in parts.items() if v is not None}
    if parts:
        imdf_scores = pd.concat(parts.values(), axis=1)
        out = PROC_DIR / "imdf_gene_scores_by_timepoint.parquet"
        imdf_scores.to_parquet(out)
        print(f"  imdf_gene_scores: {imdf_scores.shape} → {out.name}")
    else:
        print("  imdf_gene_scores: MAGeCK outputs missing")


# ── Step 5: Ortholog mapping ──────────────────────────────────────────────────

def fetch_orthologs():
    out = PROC_DIR / "mouse_human_orthologs.parquet"
    if out.exists():
        df = pd.read_parquet(out)
        print(f"  ortholog map: {len(df)} mappings (cached)")
        return df

    print("  fetching mouse→human orthologs from Ensembl BioMart ...")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        'uniqueRows="1" count="" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        '<Attribute name="external_gene_name"/>'
        '<Attribute name="hsapiens_homolog_associated_gene_name"/>'
        '<Attribute name="hsapiens_homolog_orthology_confidence"/>'
        '</Dataset></Query>'
    )
    url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote(xml)
    resp = urllib.request.urlopen(url, timeout=120)
    tsv = resp.read().decode("utf-8")

    df = pd.read_csv(
        io.StringIO(tsv), sep="\t",
        names=["mouse_symbol", "human_symbol", "orthology_confidence"],
        header=0, dtype=str,
    ).dropna(subset=["mouse_symbol", "human_symbol"])
    df = df[df["orthology_confidence"] == "1"]
    df = df[df["human_symbol"].str.strip() != ""]
    df = df.drop_duplicates(subset="mouse_symbol")
    df.to_parquet(out, index=False)
    print(f"  saved {len(df)} 1:1 orthologs → {out.name}")
    return df


# ── Step 6: Feature matrix ────────────────────────────────────────────────────

def _strip_entrez(col):
    return re.sub(r"\s*\(\d+\)\s*$", "", col).strip()


def _parse_gmt(path):
    gene_sets = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = set(parts[2:])
    return gene_sets


def build_feature_matrix(mouse_genes, ortholog_df):
    mouse_to_human = dict(zip(ortholog_df["mouse_symbol"], ortholog_df["human_symbol"]))
    human_genes = [mouse_to_human.get(g) for g in mouse_genes]
    has_ortholog = [1 if h else 0 for h in human_genes]

    # ── DepMap ──
    print("  loading DepMap ...")
    depmap = pd.read_csv(DEPMAP_PATH, index_col=0, compression="gzip").astype(np.float32)
    depmap.columns = [_strip_entrez(c) for c in depmap.columns]

    # Entrez→symbol map for Reactome
    raw_cols = pd.read_csv(DEPMAP_PATH, index_col=0, nrows=1, compression="gzip").columns
    entrez_to_symbol = {}
    for col in raw_cols:
        m = re.match(r"^(.+)\s+\((\d+)\)$", col.strip())
        if m:
            entrez_to_symbol[m.group(2)] = m.group(1).strip()

    chronos_mean = depmap.mean(axis=0)
    is_essential = (chronos_mean < -0.5).astype(int)

    # Co-essentiality PCA
    print("  computing co-essentiality PCA ...")
    gene_mat = depmap.T.fillna(0.0).values.astype(np.float32)
    gene_mat -= gene_mat.mean(axis=1, keepdims=True)
    pca = PCA(n_components=10, random_state=42)
    pcs = pca.fit_transform(gene_mat)
    coess_df = pd.DataFrame(pcs, index=depmap.columns,
                            columns=[f"coess_pc{i+1}" for i in range(10)])
    print(f"  PCA explained variance (10 PCs): {pca.explained_variance_ratio_.cumsum()[-1]:.1%}")

    # ── CCLE ──
    print("  loading CCLE expression ...")
    ccle = pd.read_csv(CCLE_PATH, index_col=0, compression="gzip")
    ccle.columns = [_strip_entrez(c) for c in ccle.columns]
    molm13 = ccle.loc["ACH-000362"] if "ACH-000362" in ccle.index else ccle.mean(axis=0)

    # ── Reactome binary ──
    print("  parsing Reactome pathways ...")
    gene_to_pathways = {}
    with gzip.open(REACTOME_PATH, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or parts[7].strip() != "Homo sapiens":
                continue
            sym = entrez_to_symbol.get(parts[0].strip())
            if sym:
                gene_to_pathways.setdefault(sym, set()).add(parts[4].strip())

    pathway_sizes = {}
    for paths in gene_to_pathways.values():
        for p in paths:
            pathway_sizes[p] = pathway_sizes.get(p, 0) + 1
    kept_pathways = sorted(p for p, s in pathway_sizes.items() if 10 <= s <= 500)
    kept_set = set(kept_pathways)
    print(f"  Reactome pathways (10–500 genes): {len(kept_pathways)}")

    # ── Hallmarks ──
    print("  parsing Hallmark gene sets ...")
    hallmarks = _parse_gmt(HALLMARKS_PATH)
    hallmark_cols = sorted(hallmarks.keys())

    # ── Assemble ──
    print(f"  assembling feature matrix for {len(mouse_genes)} mouse genes ...")
    rows = []
    for mg, hg in zip(mouse_genes, human_genes):
        row = {
            "has_ortholog":      int(hg is not None),
            "chronos_mean":      float(chronos_mean.get(hg, np.nan)) if hg else np.nan,
            "is_core_essential": int(is_essential.get(hg, 0))        if hg else 0,
            "ccle_log_tpm":      float(molm13.get(hg, np.nan))       if hg else np.nan,
        }
        # Co-essentiality PCA
        if hg is not None and hg in coess_df.index:
            for i, pc in enumerate(coess_df.loc[hg], 1):
                row[f"coess_pc{i}"] = float(pc)
        else:
            for i in range(1, 11):
                row[f"coess_pc{i}"] = 0.0
        # Reactome binary
        hg_paths = gene_to_pathways.get(hg, set()) if hg else set()
        for p in kept_pathways:
            row[p] = 1 if p in hg_paths else 0
        # Hallmark binary
        for h in hallmark_cols:
            row[h] = 1 if (hg in hallmarks[h]) else 0
        rows.append(row)

    features = pd.DataFrame(rows, index=pd.Index(mouse_genes, name="gene_symbol"))

    # Median-impute numeric NaN
    for col in ["chronos_mean", "ccle_log_tpm"]:
        med = features[col].median()
        n_miss = features[col].isna().sum()
        features[col] = features[col].fillna(med)
        if n_miss > 0:
            print(f"  imputed {n_miss} NaN in {col} with median={med:.3f}")

    out = PROC_DIR / "features_mouse_genes.parquet"
    features.to_parquet(out)
    print(f"  saved feature matrix: {features.shape} → {out.name}")
    return features


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print MAGeCK commands without running them")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature matrix build")
    args = parser.parse_args()

    print("\n=== Step 1: Download raw count files ===")
    download_raw_files()

    print("\n=== Step 2: Detect sample groups ===")
    jobs = get_sample_groups()

    print("\n=== Step 3: Run MAGeCK ===")
    run_mageck_jobs(jobs, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n=== Step 4: Parse MAGeCK outputs ===")
        parse_mageck_outputs()

        print("\n=== Step 5: Ortholog mapping ===")
        ortholog_df = fetch_orthologs()

        if not args.skip_features:
            print("\n=== Step 6: Build feature matrix ===")
            # Collect all mouse genes from score files
            mouse_genes = set()
            for f in [
                PROC_DIR / "menuetto_gene_scores.parquet",
                PROC_DIR / "scherzo_gene_scores.parquet",
                PROC_DIR / "imdf_gene_scores_by_timepoint.parquet",
            ]:
                if f.exists():
                    mouse_genes.update(pd.read_parquet(f).index.tolist())
            if mouse_genes:
                build_feature_matrix(sorted(mouse_genes), ortholog_df)
            else:
                print("  no gene score files found — skipping feature matrix")

    print("\nDone.")


if __name__ == "__main__":
    main()
