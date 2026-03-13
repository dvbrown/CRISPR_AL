#!/usr/bin/env nextflow
/*
 * Design A Feature Engineering Pipeline
 * Builds gene_features.parquet and screen_scores.parquet for Aim 1 analysis.
 */

nextflow.enable.dsl=2

params.screen_path    = "${projectDir}/../../data/bulk/chen2019_venetoclax/BIOGRID-ORCS-SCREEN_1393-2.0.18.screen.tab.txt"
params.ccle_path      = "${projectDir}/../../data/bulk/ccle_expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz"
params.depmap_path    = "${projectDir}/../../data/bulk/depmap_crispr_gene_effect/CRISPRGeneEffect.csv.gz"
params.reactome_path  = "${projectDir}/../../data/bulk/pathway_annotations/NCBI2Reactome_PE_Pathway.txt.gz"
params.goa_path       = "${projectDir}/../../data/bulk/pathway_annotations/goa_human.gaf.gz"
params.hallmarks_path = "${projectDir}/../../data/bulk/pathway_annotations/h.all.v2024.1.Hs.symbols.gmt.gz"
params.kegg_path      = "${projectDir}/../../data/bulk/pathway_annotations/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt.gz"
params.output_dir     = "${projectDir}/../../notebooks/crispr_screen_transfer/artifacts"
params.molm13_id      = "ACH-000362"

def src = "${projectDir}/../../src"

process LOAD_SCREEN_SCORES {
    input:
    val screen_path

    output:
    path "screen_scores.parquet", emit: screen_parquet

    script:
    """
    python3 - <<'PYEOF'
import sys; sys.path.insert(0, "${src}")
from crispr_al.screen import load_screen_scores, zscore_normalize, assign_hit_labels
from crispr_al.io import save_parquet

df = load_screen_scores("${screen_path}")
df = zscore_normalize(df)
df = assign_hit_labels(df)
save_parquet(df.set_index("gene_symbol"), "screen_scores.parquet")
print(f"Saved screen_scores.parquet: {len(df)} genes", flush=True)
PYEOF
    """
}

process BUILD_EXPRESSION_FEATURES {
    input:
    val ccle_path

    output:
    path "expression_features.parquet", emit: expr_parquet

    script:
    """
    python3 - <<'PYEOF'
import sys; sys.path.insert(0, "${src}")
from crispr_al.features import build_expression_feature
from crispr_al.io import save_parquet

series = build_expression_feature("${ccle_path}", molm13_id="${params.molm13_id}")
df = series.to_frame()
df.index.name = "gene_symbol"
save_parquet(df, "expression_features.parquet")
print(f"Saved expression_features.parquet: {len(df)} genes", flush=True)
PYEOF
    """
}

process BUILD_COESSENTIALITY_FEATURES {
    memory '8 GB'
    cpus 4

    input:
    val  depmap_path
    path screen_parquet

    output:
    path "coessentiality_features.parquet", emit: coess_parquet

    script:
    """
    python3 - <<'PYEOF'
import sys; sys.path.insert(0, "${src}")
from crispr_al.features import build_coessentiality_features
from crispr_al.io import load_parquet, save_parquet

screen_df   = load_parquet("${screen_parquet}").reset_index()
screen_genes = screen_df["gene_symbol"].tolist()
coess_df    = build_coessentiality_features("${depmap_path}", screen_genes, molm13_id="${params.molm13_id}")
save_parquet(coess_df, "coessentiality_features.parquet")
print(f"Saved coessentiality_features.parquet: {len(coess_df)} genes", flush=True)
PYEOF
    """
}

process BUILD_PATHWAY_FEATURES {
    memory '4 GB'
    cpus 2

    input:
    path screen_parquet

    output:
    path "pathway_features.parquet", emit: pathway_parquet

    script:
    """
    python3 - <<'PYEOF'
import sys; sys.path.insert(0, "${src}")
from crispr_al.features import build_pathway_features
from crispr_al.io import load_parquet, save_parquet

screen_df  = load_parquet("${screen_parquet}").reset_index()
pathway_df = build_pathway_features(
    reactome_path="${params.reactome_path}",
    goa_path="${params.goa_path}",
    hallmarks_path="${params.hallmarks_path}",
    kegg_path="${params.kegg_path}",
    screen_df=screen_df,
)
save_parquet(pathway_df, "pathway_features.parquet")
print(f"Saved pathway_features.parquet: {len(pathway_df)} genes", flush=True)
PYEOF
    """
}

process ASSEMBLE_FEATURES {
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path screen_parquet
    path expr_parquet
    path coess_parquet
    path pathway_parquet

    output:
    path "gene_features.parquet",  emit: features_parquet
    path "screen_scores.parquet",  emit: screen_out

    script:
    """
    python3 - <<'PYEOF'
import sys, shutil; sys.path.insert(0, "${src}")
from crispr_al.features import assemble_gene_features
from crispr_al.io import load_parquet, save_parquet

screen_df    = load_parquet("${screen_parquet}").reset_index()
screen_genes = screen_df["gene_symbol"].tolist()
expr_series  = load_parquet("${expr_parquet}").iloc[:, 0]
coess_df     = load_parquet("${coess_parquet}")
pathway_df   = load_parquet("${pathway_parquet}")

features = assemble_gene_features(screen_genes, expr_series, coess_df, pathway_df)
save_parquet(features, "gene_features.parquet")
shutil.copy("${screen_parquet}", "screen_scores.parquet")
print(f"Saved gene_features.parquet: {features.shape}", flush=True)
PYEOF
    """
}

workflow {
    screen_out = LOAD_SCREEN_SCORES(params.screen_path)
    expr_out   = BUILD_EXPRESSION_FEATURES(params.ccle_path)
    coess_out  = BUILD_COESSENTIALITY_FEATURES(params.depmap_path, screen_out.screen_parquet)
    pathway_out = BUILD_PATHWAY_FEATURES(screen_out.screen_parquet)
    ASSEMBLE_FEATURES(
        screen_out.screen_parquet,
        expr_out.expr_parquet,
        coess_out.coess_parquet,
        pathway_out.pathway_parquet,
    )
}
