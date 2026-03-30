"""Feature engineering for Design A."""
import re
import gzip
import numpy as np
import pandas as pd


def _strip_entrez(col: str) -> str:
    """Strip ' (ENTREZ_ID)' suffix from column names like 'BCL2 (596)'."""
    return re.sub(r"\s*\(\d+\)\s*$", "", col).strip()


def build_expression_feature(ccle_path: str, molm13_id: str = "ACH-000362") -> pd.Series:
    """Extract MOLM-13 row from CCLE expression matrix.

    Returns Series indexed by HGNC gene symbol with log2(TPM+1) values.
    """
    df = pd.read_csv(ccle_path, index_col=0, compression="gzip")
    row = df.loc[molm13_id]
    row.index = [_strip_entrez(c) for c in row.index]
    row.name = "molm13_log_tpm"
    return row


def build_coessentiality_features(
    depmap_path: str,
    screen_genes: list,
    molm13_id: str = "ACH-000362",
) -> pd.DataFrame:
    """Compute co-essentiality features from DepMap CRISPR gene effect matrix.

    For each screen gene present in DepMap, computes:
      - coessential_mean_r_top50: mean Pearson r with top-50 most correlated genes
      - coessential_molm13_chronos: raw Chronos score for MOLM-13

    Missing genes → 0.0.
    Returns DataFrame indexed by gene_symbol.
    """
    # Load as float32 to save memory
    matrix = pd.read_csv(depmap_path, index_col=0, compression="gzip").astype(np.float32)
    matrix.columns = [_strip_entrez(c) for c in matrix.columns]

    # Extract MOLM-13 Chronos row
    molm13_chronos = matrix.loc[molm13_id] if molm13_id in matrix.index else pd.Series(dtype=np.float32)

    # Identify screen genes present in DepMap
    depmap_genes = set(matrix.columns)
    present_genes = [g for g in screen_genes if g in depmap_genes]

    # Transpose: genes × cell lines for correlation computation
    gene_matrix = matrix[present_genes].T.values  # (n_genes_present, n_cell_lines)

    # Compute row-wise Pearson correlations
    # For each screen gene, correlate with all DepMap genes
    all_genes_matrix = matrix.T.values  # (18435, 1186)

    # Center the matrices
    gene_mat_centered = gene_matrix - gene_matrix.mean(axis=1, keepdims=True)
    all_mat_centered = all_genes_matrix - all_genes_matrix.mean(axis=1, keepdims=True)

    # Norms
    gene_norms = np.linalg.norm(gene_mat_centered, axis=1, keepdims=True)
    all_norms = np.linalg.norm(all_mat_centered, axis=1, keepdims=True)

    # Precompute column→index map for O(1) self-correlation exclusion
    col_index = {g: i for i, g in enumerate(matrix.columns)}

    # Pearson r: (n_screen, n_all) in chunks to bound memory
    chunk_size = 500
    mean_r_top50 = np.zeros(len(present_genes), dtype=np.float32)

    for start in range(0, len(present_genes), chunk_size):
        end = min(start + chunk_size, len(present_genes))
        chunk = gene_mat_centered[start:end]
        chunk_norms = gene_norms[start:end]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (chunk @ all_mat_centered.T) / (chunk_norms * all_norms.T + 1e-10)
        corr = np.clip(corr, -1.0, 1.0)
        for i in range(end - start):
            row_corr = corr[i]
            gene_name = present_genes[start + i]
            gene_idx = col_index.get(gene_name)
            if gene_idx is not None:
                row_corr = row_corr.copy()
                row_corr[gene_idx] = -2.0
            # O(n) partial sort — only need top-50
            mean_r_top50[start + i] = np.partition(row_corr, -50)[-50:].mean()

    result = pd.DataFrame({
        "coessential_mean_r_top50": mean_r_top50,
        "coessential_molm13_chronos": [float(molm13_chronos.get(g, 0.0)) for g in present_genes],
    }, index=pd.Index(present_genes, name="gene_symbol"))

    return result.reindex(screen_genes).fillna(0.0)


def _parse_gmt(path: str) -> dict:
    """Parse GMT file (gzip or plain). Returns {geneset_name: set_of_genes}."""
    gene_sets = {}
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = set(parts[2:])
            gene_sets[name] = genes
    return gene_sets


def build_pathway_features(
    reactome_path: str,
    goa_path: str,
    hallmarks_path: str,
    kegg_path: str,
    screen_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build scalar pathway features for each screen gene.

    screen_df must have columns: gene_symbol, entrez_id

    Returns DataFrame indexed by gene_symbol with columns:
      n_reactome_pathways, n_go_bp_terms, n_go_mf_terms,
      in_hallmark_apoptosis, in_hallmark_oxidative_phosphorylation, n_kegg_pathways
    """
    gene_symbols = screen_df["gene_symbol"].tolist()
    entrez_ids = screen_df.set_index("gene_symbol")["entrez_id"].astype(str)

    # --- Reactome ---
    reactome_counts = {g: 0 for g in gene_symbols}
    entrez_to_gene = {str(v): k for k, v in entrez_ids.items()}
    open_fn = gzip.open if str(reactome_path).endswith(".gz") else open
    with open_fn(reactome_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            entrez = parts[0].strip()
            species = parts[7].strip()
            if species != "Homo sapiens":
                continue
            gene = entrez_to_gene.get(entrez)
            if gene and gene in reactome_counts:
                reactome_counts[gene] += 1

    # --- GOA ---
    go_bp_counts = {g: 0 for g in gene_symbols}
    go_mf_counts = {g: 0 for g in gene_symbols}
    gene_set_symbols = set(gene_symbols)
    open_fn = gzip.open if str(goa_path).endswith(".gz") else open
    with open_fn(goa_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            symbol = parts[2].strip()
            evidence = parts[6].strip()
            aspect = parts[8].strip()
            if evidence == "IEA":
                continue
            if symbol not in gene_set_symbols:
                continue
            if aspect == "P":
                go_bp_counts[symbol] += 1
            elif aspect == "F":
                go_mf_counts[symbol] += 1

    # --- GMT files ---
    hallmarks = _parse_gmt(hallmarks_path)
    kegg = _parse_gmt(kegg_path)

    apoptosis_genes = hallmarks.get("HALLMARK_APOPTOSIS", set())
    oxphos_genes = hallmarks.get("HALLMARK_OXIDATIVE_PHOSPHORYLATION", set())
    kegg_counts = {g: 0 for g in gene_symbols}
    for geneset_genes in kegg.values():
        for g in geneset_genes:
            if g in kegg_counts:
                kegg_counts[g] += 1

    # Assemble
    rows = []
    for gene in gene_symbols:
        rows.append({
            "gene_symbol": gene,
            "n_reactome_pathways": reactome_counts[gene],
            "n_go_bp_terms": go_bp_counts[gene],
            "n_go_mf_terms": go_mf_counts[gene],
            "in_hallmark_apoptosis": int(gene in apoptosis_genes),
            "in_hallmark_oxidative_phosphorylation": int(gene in oxphos_genes),
            "n_kegg_pathways": kegg_counts[gene],
        })

    result = pd.DataFrame(rows).set_index("gene_symbol")
    return result


def map_mouse_to_human_orthologues(
    mouse_genes: list,
    ortho_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map mouse gene symbols to human orthologue symbols.

    ortho_df must have columns: human_symbol, mouse_symbol (one-to-one orthologues).

    Returns DataFrame with columns: mouse_symbol, human_symbol. Genes without
    a one-to-one orthologue are excluded. Logs coverage statistics.
    """
    import logging
    log = logging.getLogger(__name__)

    mouse_series = pd.Series(mouse_genes, name="mouse_symbol").drop_duplicates()
    mapped = mouse_series.to_frame().merge(
        ortho_df[["mouse_symbol", "human_symbol"]].drop_duplicates(subset=["mouse_symbol"]),
        on="mouse_symbol",
        how="left",
    )
    n_total = len(mapped)
    n_mapped = mapped["human_symbol"].notna().sum()
    coverage = n_mapped / n_total if n_total > 0 else 0.0
    log.info("Orthologue mapping: %d/%d genes mapped (%.1f%%)", n_mapped, n_total, 100 * coverage)
    return mapped.dropna(subset=["human_symbol"]).reset_index(drop=True)


def build_olivieri_features(
    gene_entrez_df: pd.DataFrame,
    reactome_path: str,
    goa_path: str,
    hallmarks_path: str,
    kegg_path: str,
) -> pd.DataFrame:
    """Build 6 pathway features for the Olivieri 2020 gene universe.

    gene_entrez_df must have columns: gene_symbol, entrez_id.
    Delegates to build_pathway_features(); skips expression and co-essentiality
    features because RPE1-hTERT is not in DepMap/CCLE.

    Returns DataFrame indexed by gene_symbol with columns:
      n_reactome_pathways, n_go_bp_terms, n_go_mf_terms,
      in_hallmark_apoptosis, in_hallmark_oxidative_phosphorylation, n_kegg_pathways
    """
    return build_pathway_features(
        reactome_path=reactome_path,
        goa_path=goa_path,
        hallmarks_path=hallmarks_path,
        kegg_path=kegg_path,
        screen_df=gene_entrez_df,
    )


def assemble_gene_features(
    screen_genes: list,
    expr_series: pd.Series,
    coess_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all features into a single DataFrame.

    Index: gene_symbol. Columns: 9 features.
    Missing values filled with 0.0.
    """
    expr_df = expr_series.reindex(screen_genes).fillna(0.0).to_frame()
    expr_df.index.name = "gene_symbol"
    expr_df.columns = ["molm13_log_tpm"]

    coess_aligned = coess_df.reindex(screen_genes).fillna(0.0)
    pathway_aligned = pathway_df.reindex(screen_genes).fillna(0.0)

    result = pd.concat([expr_df, coess_aligned, pathway_aligned], axis=1)
    result = result.fillna(0.0)

    expected_cols = [
        "molm13_log_tpm",
        "coessential_mean_r_top50",
        "coessential_molm13_chronos",
        "n_reactome_pathways",
        "n_go_bp_terms",
        "n_go_mf_terms",
        "in_hallmark_apoptosis",
        "in_hallmark_oxidative_phosphorylation",
        "n_kegg_pathways",
    ]
    assert list(result.columns) == expected_cols, f"Expected {expected_cols}, got {list(result.columns)}"
    assert result.shape[0] == len(screen_genes), f"Expected {len(screen_genes)} rows, got {result.shape[0]}"

    return result
