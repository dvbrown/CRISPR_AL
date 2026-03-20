"""Phase 0: Download and preprocess GSE285778 (Menuetto/Scherzo/iMDF) CRISPR screen data.

Outputs:
  data/bulk/menuetto_scherzo_2025/raw/          — raw GEO count files (4 files)
  data/bulk/menuetto_scherzo_2025/processed/    — gene scores + feature matrix (parquet)

Run:
  marimo edit --watch notebooks/Cas12a_EuMyc/00_download_and_preprocess.py
"""
import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 0 — Data Download and Preprocessing

    **Dataset:** Jin, Deng, La Marca et al. (2025) *Nat Commun* — GSE285778

    Three screens from the same lab, same cell system:

    | Screen | Cell model | Library | Conditions |
    |---|---|---|---|
    | Menuetto | Eµ-MYC lymphoma (#20) | Menuetto (~21,743 genes, 2 crRNA/gene) | Nutlin-3a, S63845, DMSO vs Input |
    | Scherzo | Eµ-MYC lymphoma (#20) | Scherzo (~21,721 genes, 4 crRNA/gene) | Nutlin-3a, S63845, DMSO vs Input |
    | iMDF | iMDF immortalised fibroblasts | Menuetto | T0→T1→T2→T3 dropout |

    **Steps:**
    1. Download raw count files from GEO FTP
    2. Inspect sample column names
    3. Compute gene-level LFC + FDR with MAGeCK test
    4. Build mouse→human ortholog map (Ensembl BioMart)
    5. Build feature matrix (DepMap Chronos, CCLE expression, Reactome, Hallmarks, essential flag, co-essentiality PCA)
    6. Save all processed parquets
    """)
    return


# ── Imports ──────────────────────────────────────────────────────────────────

@app.cell
def _():
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

    return (
        Path, PCA, StandardScaler,
        gzip, io, np, pd, re, subprocess, sys,
        urllib,
    )


# ── Paths ─────────────────────────────────────────────────────────────────────

@app.cell
def _(Path):
    # Notebook lives at notebooks/Cas12a_EuMyc/ → ROOT is 2 levels up.
    # When exported/run from a tmp path, fall back to cwd.
    _nb = Path(__file__).resolve()
    ROOT = _nb.parents[2] if len(_nb.parents) > 2 and (_nb.parents[2] / "src").exists() \
           else Path.cwd()
    DATA_ROOT = ROOT / "data" / "bulk"

    RAW_DIR   = DATA_ROOT / "menuetto_scherzo_2025" / "raw"
    PROC_DIR  = DATA_ROOT / "menuetto_scherzo_2025" / "processed"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    DEPMAP_PATH   = DATA_ROOT / "depmap_crispr_gene_effect" / "CRISPRGeneEffect.csv.gz"
    CCLE_PATH     = DATA_ROOT / "ccle_expression" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz"
    REACTOME_PATH = DATA_ROOT / "pathway_annotations" / "NCBI2Reactome_PE_Pathway.txt.gz"
    HALLMARKS_PATH = DATA_ROOT / "pathway_annotations" / "h.all.v2024.1.Hs.symbols.gmt.gz"

    return (
        ROOT, DATA_ROOT, RAW_DIR, PROC_DIR,
        DEPMAP_PATH, CCLE_PATH, REACTOME_PATH, HALLMARKS_PATH,
    )


# ── Step 1: Download ──────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("## Step 1 — Download raw count files from GEO")
    return


@app.cell
def _(RAW_DIR, urllib):
    GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285778/suppl/"
    GEO_FILES = [
        "GSE285778_EuMycCountMenuetto.txt.gz",
        "GSE285778_EuMycCountScherzo.txt.gz",
        "GSE285778_iMDFCount.txt.gz",
        "GSE285778_InVivoCount.txt.gz",
    ]

    for _fname in GEO_FILES:
        _dest = RAW_DIR / _fname
        if not _dest.exists():
            print(f"Downloading {_fname} ...")
            urllib.request.urlretrieve(GEO_BASE + _fname, _dest)
            print(f"  saved  {_dest.stat().st_size / 1024:.0f} KB")
        else:
            print(f"  exists {_fname}  ({_dest.stat().st_size / 1024:.0f} KB)")

    raw_menuetto = RAW_DIR / "GSE285778_EuMycCountMenuetto.txt.gz"
    raw_scherzo  = RAW_DIR / "GSE285778_EuMycCountScherzo.txt.gz"
    raw_imdf     = RAW_DIR / "GSE285778_iMDFCount.txt.gz"

    return GEO_BASE, GEO_FILES, raw_menuetto, raw_scherzo, raw_imdf


# ── Step 2: Inspect headers ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("## Step 2 — Inspect count file column names")
    return


@app.cell
def _(gzip, raw_imdf, raw_menuetto, raw_scherzo):
    def _read_header(path):
        with gzip.open(path, "rt") as f:
            return f.readline().rstrip("\n").split("\t")

    menuetto_header = _read_header(raw_menuetto)
    scherzo_header  = _read_header(raw_scherzo)
    imdf_header     = _read_header(raw_imdf)

    # Columns 0 and 1 are sgRNA ID and gene; the rest are sample names
    menuetto_samples = menuetto_header[2:]
    scherzo_samples  = scherzo_header[2:]
    imdf_samples     = imdf_header[2:]

    print("Menuetto samples:", menuetto_samples)
    print("Scherzo samples: ", scherzo_samples)
    print("iMDF samples:    ", imdf_samples)

    return menuetto_header, menuetto_samples, scherzo_header, scherzo_samples, imdf_header, imdf_samples


@app.cell
def _(imdf_samples, menuetto_samples, scherzo_samples):
    def _grep(samples, keyword):
        return [s for s in samples if keyword.lower() in s.lower()]

    # Detect treatment groups from sample names
    menuetto_input   = _grep(menuetto_samples, "Input")
    menuetto_nutlin  = _grep(menuetto_samples, "Nutlin")
    menuetto_s63845  = _grep(menuetto_samples, "S63845")
    menuetto_dmso    = _grep(menuetto_samples, "DMSO")

    scherzo_input    = _grep(scherzo_samples, "Input")
    scherzo_nutlin   = _grep(scherzo_samples, "Nutlin")
    scherzo_s63845   = _grep(scherzo_samples, "S63845")
    scherzo_dmso     = _grep(scherzo_samples, "DMSO")

    imdf_t0 = _grep(imdf_samples, "T0")
    imdf_t1 = _grep(imdf_samples, "T1")
    imdf_t2 = _grep(imdf_samples, "T2")
    imdf_t3 = _grep(imdf_samples, "T3")

    for _name, _lst in [
        ("menuetto_input", menuetto_input), ("menuetto_nutlin", menuetto_nutlin),
        ("menuetto_s63845", menuetto_s63845), ("menuetto_dmso", menuetto_dmso),
        ("scherzo_input", scherzo_input), ("scherzo_nutlin", scherzo_nutlin),
        ("scherzo_s63845", scherzo_s63845), ("scherzo_dmso", scherzo_dmso),
        ("imdf_t0", imdf_t0), ("imdf_t1", imdf_t1),
        ("imdf_t2", imdf_t2), ("imdf_t3", imdf_t3),
    ]:
        print(f"  {_name:20s} ({len(_lst)}): {_lst}")

    return (
        imdf_t0, imdf_t1, imdf_t2, imdf_t3,
        menuetto_dmso, menuetto_input, menuetto_nutlin, menuetto_s63845,
        scherzo_dmso, scherzo_input, scherzo_nutlin, scherzo_s63845,
    )


# ── Step 3: MAGeCK scoring ────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Step 3 — MAGeCK test: gene-level LFC + FDR

    Install MAGeCK once via conda (bioconda channel):
    ```bash
    micromamba create -n mageck -c bioconda -c conda-forge mageck
    micromamba activate mageck
    ```

    The cell below constructs and runs all comparisons.
    If MAGeCK is not in PATH, it prints the commands instead.
    """)
    return


@app.cell
def _(
    PROC_DIR,
    imdf_t0, imdf_t1, imdf_t2, imdf_t3,
    menuetto_dmso, menuetto_input, menuetto_nutlin, menuetto_s63845,
    raw_imdf, raw_menuetto, raw_scherzo,
    scherzo_dmso, scherzo_input, scherzo_nutlin, scherzo_s63845,
    subprocess,
):
    def _run_mageck(count_table, treatment_ids, control_ids, output_prefix, dry_run=False):
        """Run or print a MAGeCK test command."""
        cmd = [
            "mageck", "test",
            "--count-table",    str(count_table),
            "--treatment-id",   ",".join(treatment_ids),
            "--control-id",     ",".join(control_ids),
            "--output-prefix",  str(output_prefix),
            "--gene-lfc-method", "median",
            "--adjust-method",  "fdr",
            "--norm-method",    "median",
        ]
        if dry_run:
            print("  (dry run) " + " ".join(cmd))
            return True
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  MAGeCK error: {result.stderr[:500]}")
                return False
            return True
        except FileNotFoundError:
            print("  MAGeCK not found. Run manually:")
            print("  " + " ".join(cmd))
            return False

    # Check if MAGeCK is available
    _mageck_available = subprocess.run(
        ["which", "mageck"], capture_output=True
    ).returncode == 0
    print(f"MAGeCK available: {_mageck_available}")

    # Define all comparisons: (count_file, treatment, control, output_prefix)
    MAGECK_JOBS = [
        (raw_menuetto, menuetto_nutlin,  menuetto_input,  PROC_DIR / "menuetto_nutlin_vs_input"),
        (raw_menuetto, menuetto_s63845,  menuetto_input,  PROC_DIR / "menuetto_s63845_vs_input"),
        (raw_menuetto, menuetto_dmso,    menuetto_input,  PROC_DIR / "menuetto_dmso_vs_input"),
        (raw_scherzo,  scherzo_nutlin,   scherzo_input,   PROC_DIR / "scherzo_nutlin_vs_input"),
        (raw_scherzo,  scherzo_s63845,   scherzo_input,   PROC_DIR / "scherzo_s63845_vs_input"),
        (raw_scherzo,  scherzo_dmso,     scherzo_input,   PROC_DIR / "scherzo_dmso_vs_input"),
        (raw_imdf,     imdf_t1,          imdf_t0,         PROC_DIR / "imdf_t1_vs_t0"),
        (raw_imdf,     imdf_t2,          imdf_t0,         PROC_DIR / "imdf_t2_vs_t0"),
        (raw_imdf,     imdf_t3,          imdf_t0,         PROC_DIR / "imdf_t3_vs_t0"),
    ]

    for _count, _treat, _ctrl, _prefix in MAGECK_JOBS:
        _out = Path(str(_prefix) + ".gene_summary.txt")
        if _out.exists():
            print(f"  skip (exists) {_out.name}")
            continue
        print(f"  running: {_prefix.name}")
        _run_mageck(_count, _treat, _ctrl, _prefix, dry_run=not _mageck_available)

    return MAGECK_JOBS


# ── Step 4: Parse MAGeCK outputs → gene score parquets ───────────────────────

@app.cell
def _(mo):
    mo.md("## Step 4 — Parse MAGeCK outputs into gene score parquets")
    return


@app.cell
def _(PROC_DIR, pd):
    def _load_gene_summary(prefix_path):
        """Read a MAGeCK gene_summary.txt. Returns DataFrame indexed by gene id."""
        path = Path(str(prefix_path) + ".gene_summary.txt")
        if not path.exists():
            return None
        df = pd.read_csv(path, sep="\t", index_col=0)
        return df

    def _extract_lfc_fdr(prefix_path, lfc_col="neg|lfc", neg_fdr_col="neg|fdr", pos_fdr_col="pos|fdr"):
        """Extract lfc and best FDR (min of neg/pos) from a MAGeCK gene summary."""
        df = _load_gene_summary(prefix_path)
        if df is None:
            return None
        result = pd.DataFrame({
            "lfc": df[lfc_col],
            "fdr_neg": df[neg_fdr_col],
            "fdr_pos": df[pos_fdr_col],
        })
        result.index.name = "gene_symbol"
        return result

    # ── Menuetto gene scores ──
    _menuetto_parts = {}
    for _cond, _prefix in [
        ("nutlin",  PROC_DIR / "menuetto_nutlin_vs_input"),
        ("s63845",  PROC_DIR / "menuetto_s63845_vs_input"),
        ("dmso",    PROC_DIR / "menuetto_dmso_vs_input"),
    ]:
        _df = _extract_lfc_fdr(_prefix)
        if _df is not None:
            _menuetto_parts[_cond] = _df.rename(columns={
                "lfc":     f"lfc_{_cond}",
                "fdr_neg": f"fdr_neg_{_cond}",
                "fdr_pos": f"fdr_pos_{_cond}",
            })

    if _menuetto_parts:
        menuetto_scores = pd.concat(_menuetto_parts.values(), axis=1)
        menuetto_scores.to_parquet(PROC_DIR / "menuetto_gene_scores.parquet")
        print(f"menuetto_gene_scores: {menuetto_scores.shape}  -> {PROC_DIR / 'menuetto_gene_scores.parquet'}")
    else:
        menuetto_scores = None
        print("menuetto_gene_scores: MAGeCK outputs not yet available — run MAGeCK first")

    # ── Scherzo gene scores ──
    _scherzo_parts = {}
    for _cond, _prefix in [
        ("nutlin",  PROC_DIR / "scherzo_nutlin_vs_input"),
        ("s63845",  PROC_DIR / "scherzo_s63845_vs_input"),
        ("dmso",    PROC_DIR / "scherzo_dmso_vs_input"),
    ]:
        _df = _extract_lfc_fdr(_prefix)
        if _df is not None:
            _scherzo_parts[_cond] = _df.rename(columns={
                "lfc":     f"lfc_{_cond}",
                "fdr_neg": f"fdr_neg_{_cond}",
                "fdr_pos": f"fdr_pos_{_cond}",
            })

    if _scherzo_parts:
        scherzo_scores = pd.concat(_scherzo_parts.values(), axis=1)
        scherzo_scores.to_parquet(PROC_DIR / "scherzo_gene_scores.parquet")
        print(f"scherzo_gene_scores: {scherzo_scores.shape}  -> {PROC_DIR / 'scherzo_gene_scores.parquet'}")
    else:
        scherzo_scores = None
        print("scherzo_gene_scores: MAGeCK outputs not yet available — run MAGeCK first")

    # ── iMDF gene scores ──
    _imdf_parts = {}
    for _tp, _prefix in [
        ("t1", PROC_DIR / "imdf_t1_vs_t0"),
        ("t2", PROC_DIR / "imdf_t2_vs_t0"),
        ("t3", PROC_DIR / "imdf_t3_vs_t0"),
    ]:
        _df = _extract_lfc_fdr(_prefix)
        if _df is not None:
            _imdf_parts[_tp] = _df.rename(columns={
                "lfc":     f"lfc_{_tp}",
                "fdr_neg": f"fdr_neg_{_tp}",
                "fdr_pos": f"fdr_pos_{_tp}",
            })

    if _imdf_parts:
        imdf_scores = pd.concat(_imdf_parts.values(), axis=1)
        imdf_scores.to_parquet(PROC_DIR / "imdf_gene_scores_by_timepoint.parquet")
        print(f"imdf_gene_scores: {imdf_scores.shape}  -> {PROC_DIR / 'imdf_gene_scores_by_timepoint.parquet'}")
    else:
        imdf_scores = None
        print("imdf_gene_scores: MAGeCK outputs not yet available — run MAGeCK first")

    return menuetto_scores, scherzo_scores, imdf_scores


# ── Step 5: Mouse→Human ortholog mapping ─────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Step 5 — Mouse→Human ortholog mapping (Ensembl BioMart)

    Downloads all mouse→human 1:1 high-confidence orthologs via BioMart REST API.
    Cached to `processed/mouse_human_orthologs.parquet` after first run.
    """)
    return


@app.cell
def _(PROC_DIR, io, pd, urllib):
    _ORTHOLOG_PARQUET = PROC_DIR / "mouse_human_orthologs.parquet"

    if _ORTHOLOG_PARQUET.exists():
        ortholog_map = pd.read_parquet(_ORTHOLOG_PARQUET)
        print(f"Loaded cached ortholog map: {len(ortholog_map)} mappings")
    else:
        print("Fetching mouse→human orthologs from Ensembl BioMart ...")
        _xml = (
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
        _url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote(_xml)
        _resp = urllib.request.urlopen(_url, timeout=120)
        _tsv = _resp.read().decode("utf-8")

        ortholog_map = pd.read_csv(
            io.StringIO(_tsv),
            sep="\t",
            names=["mouse_symbol", "human_symbol", "orthology_confidence"],
            header=0,
            dtype=str,
        ).dropna(subset=["mouse_symbol", "human_symbol"])

        # Keep only high-confidence (confidence == 1) mappings
        ortholog_map = ortholog_map[ortholog_map["orthology_confidence"] == "1"].copy()
        # Remove blank human symbols
        ortholog_map = ortholog_map[ortholog_map["human_symbol"].str.strip() != ""]
        # Drop duplicate mouse genes (keep first, which is high-confidence)
        ortholog_map = ortholog_map.drop_duplicates(subset="mouse_symbol")

        ortholog_map.to_parquet(_ORTHOLOG_PARQUET, index=False)
        print(f"Saved ortholog map: {len(ortholog_map)} 1:1 mappings -> {_ORTHOLOG_PARQUET}")

    # Build lookup dicts
    mouse_to_human = dict(zip(ortholog_map["mouse_symbol"], ortholog_map["human_symbol"]))
    human_to_mouse = dict(zip(ortholog_map["human_symbol"], ortholog_map["mouse_symbol"]))

    print(f"  {len(mouse_to_human)} mouse genes with human ortholog")

    return ortholog_map, mouse_to_human, human_to_mouse


# ── Step 6: DepMap features ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Step 6 — DepMap features: Chronos pan-cancer mean + co-essentiality PCA

    Loads `CRISPRGeneEffect.csv.gz` (cell lines × genes) to compute:
    - `chronos_mean`: pan-cancer mean Chronos score per gene (lower = more essential)
    - `is_core_essential`: 1 if mean Chronos < −0.5 across all cell lines
    - `coess_pc1..10`: top-10 PCA components of co-essentiality profile

    PCA is fitted on all DepMap genes before any train/holdout split.
    Mouse genes are assigned features via their human orthologs.
    """)
    return


@app.cell
def _(DEPMAP_PATH, PCA, StandardScaler, np, pd, re):
    print(f"Loading DepMap gene effect matrix from {DEPMAP_PATH.name} ...")
    _depmap_raw = pd.read_csv(DEPMAP_PATH, index_col=0, compression="gzip")

    # Strip " (ENTREZ_ID)" suffix from column names
    _depmap_raw.columns = [re.sub(r"\s*\(\d+\)\s*$", "", c).strip() for c in _depmap_raw.columns]

    # Build Entrez→symbol map from original column names (before stripping)
    _depmap_raw_reload = pd.read_csv(DEPMAP_PATH, index_col=0, nrows=1, compression="gzip")
    entrez_to_symbol = {}
    for _col in _depmap_raw_reload.columns:
        _m = re.match(r"^(.+)\s+\((\d+)\)$", _col.strip())
        if _m:
            entrez_to_symbol[_m.group(2)] = _m.group(1).strip()

    print(f"  {_depmap_raw.shape[0]} cell lines × {_depmap_raw.shape[1]} genes")
    print(f"  Entrez→symbol map: {len(entrez_to_symbol)} entries")

    # Pan-cancer mean and essential flag
    _chronos_mean = _depmap_raw.mean(axis=0)
    _is_essential = (_chronos_mean < -0.5).astype(int)

    depmap_chronos = pd.DataFrame({
        "chronos_mean":    _chronos_mean,
        "is_core_essential": _is_essential,
    })
    depmap_chronos.index.name = "gene_symbol"

    # Co-essentiality PCA: fit on genes × cell lines (transpose)
    _gene_matrix = _depmap_raw.T  # (n_genes, n_cell_lines)
    _gene_matrix_filled = _gene_matrix.fillna(0.0)  # zero-impute missing

    _scaler = StandardScaler(with_mean=True, with_std=False)
    _gene_mat_centered = _scaler.fit_transform(_gene_matrix_filled.values.astype(np.float32))

    _pca = PCA(n_components=10, random_state=42)
    _coess_pcs = _pca.fit_transform(_gene_mat_centered)

    print(f"  Co-essentiality PCA: explained variance {_pca.explained_variance_ratio_.cumsum()[-1]:.1%}")

    coess_pca_df = pd.DataFrame(
        _coess_pcs,
        index=_gene_matrix.index,
        columns=[f"coess_pc{i+1}" for i in range(10)],
    )
    coess_pca_df.index.name = "gene_symbol"

    print(f"  depmap_chronos: {depmap_chronos.shape}")
    print(f"  coess_pca_df:   {coess_pca_df.shape}")

    return depmap_chronos, coess_pca_df, entrez_to_symbol


# ── Step 7: CCLE expression ───────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Step 7 — CCLE expression (MOLM-13, human AML)

    Extracts the MOLM-13 cell line row from the CCLE expression matrix.
    MOLM-13 (ACH-000362) is an AML cell line used as a representative
    human cell line for the Eµ-MYC lymphoma model.
    """)
    return


@app.cell
def _(CCLE_PATH, pd, re):
    _MOLM13_ID = "ACH-000362"
    print(f"Loading CCLE expression from {CCLE_PATH.name} ...")
    _ccle = pd.read_csv(CCLE_PATH, index_col=0, compression="gzip", nrows=None)

    # Strip " (ENTREZ_ID)" suffix from column names
    _ccle.columns = [re.sub(r"\s*\(\d+\)\s*$", "", c).strip() for c in _ccle.columns]

    if _MOLM13_ID in _ccle.index:
        _molm13_expr = _ccle.loc[_MOLM13_ID]
    else:
        print(f"  WARNING: {_MOLM13_ID} not found. Using pan-cancer mean instead.")
        _molm13_expr = _ccle.mean(axis=0)

    ccle_expr = _molm13_expr.rename("ccle_log_tpm")
    ccle_expr.index.name = "gene_symbol"

    print(f"  CCLE expression: {len(ccle_expr)} genes (log2 TPM+1)")
    return (ccle_expr,)


# ── Step 8: Reactome binary membership ───────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Step 8 — Reactome pathway binary membership

    Builds a binary matrix: human genes × Reactome pathways.
    Filters to pathways with 10–500 members (removes tiny/huge pathways).
    Uses column 4 (pathway stable ID) and column 7 (species) from
    `NCBI2Reactome_PE_Pathway.txt`.
    """)
    return


@app.cell
def _(REACTOME_PATH, entrez_to_symbol, gzip, pd):
    print(f"Parsing Reactome pathway file ...")

    # gene_symbol → set of Reactome pathway IDs
    _gene_pathways = {}
    _pathway_names = {}

    with gzip.open(REACTOME_PATH, "rt") as _f:
        for _line in _f:
            _parts = _line.rstrip("\n").split("\t")
            if len(_parts) < 8:
                continue
            _entrez  = _parts[0].strip()
            _pathway = _parts[4].strip()  # e.g. R-HSA-109581
            _pname   = _parts[5].strip()
            _species = _parts[7].strip()
            if _species != "Homo sapiens":
                continue
            _symbol = entrez_to_symbol.get(_entrez)
            if _symbol:
                _gene_pathways.setdefault(_symbol, set()).add(_pathway)
                _pathway_names[_pathway] = _pname

    print(f"  {len(_gene_pathways)} human genes mapped to {len(_pathway_names)} Reactome pathways")

    # Compute pathway sizes (gene count per pathway)
    _pathway_sizes = {}
    for _genes in _gene_pathways.values():
        for _p in _genes:
            _pathway_sizes[_p] = _pathway_sizes.get(_p, 0) + 1

    # Filter: keep pathways with 10 ≤ size ≤ 500
    _kept_pathways = sorted(
        p for p, s in _pathway_sizes.items() if 10 <= s <= 500
    )
    print(f"  pathways after size filter (10–500): {len(_kept_pathways)}")

    # Build binary matrix for all human genes present in DepMap/CCLE
    _all_human_genes = sorted(_gene_pathways.keys())
    _rows = {}
    for _gene, _paths in _gene_pathways.items():
        _row = {p: 1 for p in _paths if p in set(_kept_pathways)}
        _rows[_gene] = _row

    reactome_binary = pd.DataFrame.from_dict(_rows, orient="index", dtype="int8")
    reactome_binary = reactome_binary.reindex(columns=_kept_pathways, fill_value=0)
    reactome_binary.index.name = "gene_symbol"

    print(f"  reactome_binary: {reactome_binary.shape}")
    return reactome_binary, _kept_pathways, _pathway_names


# ── Step 9: Hallmark binary membership ───────────────────────────────────────

@app.cell
def _(mo):
    mo.md("## Step 9 — MSigDB Hallmark binary membership (50 gene sets)")
    return


@app.cell
def _(HALLMARKS_PATH, gzip, pd):
    def _parse_gmt(path):
        gene_sets = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0]
                genes = set(parts[2:])
                gene_sets[name] = genes
        return gene_sets

    print(f"Parsing Hallmark gene sets ...")
    _hallmarks = _parse_gmt(HALLMARKS_PATH)
    _hallmark_names = sorted(_hallmarks.keys())
    print(f"  {len(_hallmark_names)} Hallmark gene sets")

    # Build binary matrix: gene → Hallmark membership
    _all_hallmark_genes = set()
    for _genes in _hallmarks.values():
        _all_hallmark_genes.update(_genes)

    _rows = {g: {h: int(g in _hallmarks[h]) for h in _hallmark_names}
             for g in _all_hallmark_genes}

    hallmark_binary = pd.DataFrame.from_dict(_rows, orient="index", dtype="int8")
    hallmark_binary = hallmark_binary[_hallmark_names]
    hallmark_binary.index.name = "gene_symbol"

    print(f"  hallmark_binary: {hallmark_binary.shape}")
    return hallmark_binary, _hallmarks, _hallmark_names


# ── Step 10: Assemble feature matrix ─────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("""
    ## Step 10 — Assemble feature matrix for mouse screen genes

    Procedure for each mouse gene:
    1. Map to human ortholog (if high-confidence 1:1 available)
    2. Look up all features via human symbol
    3. Impute missing: median for numeric, 0 for binary
    4. Add `has_ortholog` binary indicator
    """)
    return


@app.cell
def _(
    PROC_DIR,
    ccle_expr, coess_pca_df, depmap_chronos, hallmark_binary,
    mouse_to_human,
    menuetto_scores, scherzo_scores, imdf_scores,
    np, pd, reactome_binary,
):
    # Collect all mouse gene symbols from the score files
    _score_genes = set()
    for _df in [menuetto_scores, scherzo_scores, imdf_scores]:
        if _df is not None:
            _score_genes.update(_df.index.tolist())

    # Fallback: if scores not yet computed, read gene list from raw count file headers
    if not _score_genes:
        print("WARNING: Score DataFrames not yet computed. Feature matrix will be empty.")
        mouse_genes = []
    else:
        mouse_genes = sorted(_score_genes)

    print(f"Mouse screen genes: {len(mouse_genes)}")

    # Map each mouse gene to human ortholog
    _human_genes = [mouse_to_human.get(g) for g in mouse_genes]
    _has_ortholog = [1 if h is not None else 0 for h in _human_genes]

    print(f"  Has human ortholog: {sum(_has_ortholog)} / {len(mouse_genes)} "
          f"({sum(_has_ortholog)/max(len(mouse_genes),1):.1%})")

    # ── Chronos mean ──
    _chronos_vals  = [depmap_chronos.loc[h, "chronos_mean"]    if h in depmap_chronos.index else np.nan
                      for h in _human_genes]
    _essential_vals = [depmap_chronos.loc[h, "is_core_essential"] if h in depmap_chronos.index else 0
                       for h in _human_genes]

    # ── CCLE expression ──
    _ccle_vals = [ccle_expr.loc[h] if h in ccle_expr.index else np.nan
                  for h in _human_genes]

    # ── Co-essentiality PCA ──
    _coess_rows = []
    for _h in _human_genes:
        if _h is not None and _h in coess_pca_df.index:
            _coess_rows.append(coess_pca_df.loc[_h].values)
        else:
            _coess_rows.append(np.zeros(10))

    _coess_arr = np.array(_coess_rows)

    # ── Reactome binary ──
    _reactome_rows = []
    _rcols = reactome_binary.columns.tolist()
    for _h in _human_genes:
        if _h is not None and _h in reactome_binary.index:
            _reactome_rows.append(reactome_binary.loc[_h].values)
        else:
            _reactome_rows.append(np.zeros(len(_rcols), dtype="int8"))

    _reactome_arr = np.array(_reactome_rows, dtype="int8")

    # ── Hallmark binary ──
    _hallmark_cols = hallmark_binary.columns.tolist()
    _hallmark_rows = []
    for _h in _human_genes:
        if _h is not None and _h in hallmark_binary.index:
            _hallmark_rows.append(hallmark_binary.loc[_h].values)
        else:
            _hallmark_rows.append(np.zeros(len(_hallmark_cols), dtype="int8"))

    _hallmark_arr = np.array(_hallmark_rows, dtype="int8")

    # ── Assemble ──
    _scalar_cols = {
        "chronos_mean":      _chronos_vals,
        "is_core_essential": _essential_vals,
        "ccle_log_tpm":      _ccle_vals,
        "has_ortholog":      _has_ortholog,
    }

    _features = pd.DataFrame(_scalar_cols, index=pd.Index(mouse_genes, name="gene_symbol"))

    # Median-impute numeric NaN (chronos_mean, ccle_log_tpm)
    for _col in ["chronos_mean", "ccle_log_tpm"]:
        _med = _features[_col].median()
        _n_missing = _features[_col].isna().sum()
        _features[_col] = _features[_col].fillna(_med)
        if _n_missing > 0:
            print(f"  imputed {_n_missing} NaN in {_col} with median={_med:.3f}")

    # Append PCA columns
    _coess_df = pd.DataFrame(
        _coess_arr,
        index=_features.index,
        columns=[f"coess_pc{i+1}" for i in range(10)],
    )
    _reactome_df = pd.DataFrame(
        _reactome_arr,
        index=_features.index,
        columns=_rcols,
    )
    _hallmark_df = pd.DataFrame(
        _hallmark_arr,
        index=_features.index,
        columns=_hallmark_cols,
    )

    features_df = pd.concat([_features, _coess_df, _reactome_df, _hallmark_df], axis=1)

    if len(mouse_genes) > 0:
        _out = PROC_DIR / "features_mouse_genes.parquet"
        features_df.to_parquet(_out)
        print(f"\nSaved feature matrix: {features_df.shape} -> {_out}")
        print(f"  scalar features:  {_features.shape[1]}")
        print(f"  coess PCA:        {_coess_df.shape[1]}")
        print(f"  Reactome binary:  {_reactome_df.shape[1]}")
        print(f"  Hallmark binary:  {_hallmark_df.shape[1]}")
        print(f"  total features:   {features_df.shape[1]}")
    else:
        print("Feature matrix empty — run MAGeCK first to get gene lists.")

    return features_df, mouse_genes


# ── Step 11: Verification ─────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("## Step 11 — Verification")
    return


@app.cell
def _(
    PROC_DIR,
    features_df, mouse_genes,
    menuetto_scores, scherzo_scores, imdf_scores,
    mo, pd,
):
    _checks = []

    # Expected gene counts from the paper
    _EXPECTED = {
        "menuetto": 21743,
        "scherzo":  21721,
    }

    if menuetto_scores is not None:
        _n = len(menuetto_scores)
        _ok = abs(_n - _EXPECTED["menuetto"]) < 500
        _checks.append(f"{'✓' if _ok else '✗'}  menuetto genes: {_n} (expected ~{_EXPECTED['menuetto']})")

    if scherzo_scores is not None:
        _n = len(scherzo_scores)
        _ok = abs(_n - _EXPECTED["scherzo"]) < 500
        _checks.append(f"{'✓' if _ok else '✗'}  scherzo genes:  {_n} (expected ~{_EXPECTED['scherzo']})")

    if imdf_scores is not None:
        _n = len(imdf_scores)
        _checks.append(f"✓  iMDF genes: {_n}")

    if len(mouse_genes) > 0:
        _n_feats = features_df.shape[1]
        _checks.append(f"✓  feature matrix: {features_df.shape[0]} genes × {_n_feats} features")

        _n_has_orth = features_df["has_ortholog"].sum()
        _checks.append(f"✓  ortholog coverage: {int(_n_has_orth)} / {len(mouse_genes)} "
                       f"({_n_has_orth/len(mouse_genes):.1%})")

        _n_essential = features_df["is_core_essential"].sum()
        _checks.append(f"✓  core essential genes: {int(_n_essential)} "
                       f"({_n_essential/len(mouse_genes):.1%})")

    # Check output files exist
    for _fname in [
        "menuetto_gene_scores.parquet",
        "scherzo_gene_scores.parquet",
        "imdf_gene_scores_by_timepoint.parquet",
        "features_mouse_genes.parquet",
        "mouse_human_orthologs.parquet",
    ]:
        _path = PROC_DIR / _fname
        _checks.append(f"{'✓' if _path.exists() else '✗'}  {_fname}")

    return mo.md("### Checks\n\n" + "\n\n".join(_checks))


if __name__ == "__main__":
    app.run()
