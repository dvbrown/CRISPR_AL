#!/usr/bin/env nextflow

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    crispr-al/design_a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Design A: Within-Screen Holdout — 5-Loop Autonomous Research Pipeline

    Loop 1  Baseline pipeline  — 25 splits × 2 models + baseline → 51 metric JSONs
    Loop 2  Feature ablation   — leave-one-out Ridge (9 × 25 = 225 parallelised runs)
    Loop 3  Reduced model      — top features from ablation, Ridge + RF × 25 splits
    Loop 4  Transfer calib.    — score distributions, stratified rank corr, RF importances
    Loop 5  Summary report     — Markdown synthesis of all findings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    if (params.help) {
        helpMessage()
        exit 0
    }

    // -------------------------------------------------------------------------
    //  Input file channels
    // -------------------------------------------------------------------------
    chen_screen_ch   = Channel.fromPath(params.chen_screen_path,  checkIfExists: true)
    ccle_ch          = Channel.fromPath(params.ccle_path,          checkIfExists: true)
    depmap_ch        = Channel.fromPath(params.depmap_path,        checkIfExists: true)
    reactome_ch      = Channel.fromPath(params.reactome_path,      checkIfExists: true)
    goa_ch           = Channel.fromPath(params.goa_path,           checkIfExists: true)
    hallmarks_ch     = Channel.fromPath(params.hallmarks_path,     checkIfExists: true)
    kegg_ch          = Channel.fromPath(params.kegg_path,          checkIfExists: true)
    schema_ch        = Channel.fromPath(params.schema_path,        checkIfExists: true)

    // -------------------------------------------------------------------------
    //  Script file channels (passed as process inputs, not hardcoded paths)
    // -------------------------------------------------------------------------
    load_screen_script_ch       = Channel.fromPath("${projectDir}/scripts/load_screen.py")
    build_features_script_ch    = Channel.fromPath("${projectDir}/scripts/build_features.py")
    generate_splits_script_ch   = Channel.fromPath("${projectDir}/scripts/generate_splits.py")
    run_split_script_ch         = Channel.fromPath("${projectDir}/scripts/run_split.py")
    run_baseline_script_ch      = Channel.fromPath("${projectDir}/scripts/run_baseline.py")
    aggregate_results_script_ch = Channel.fromPath("${projectDir}/scripts/aggregate_results.py")
    validate_schema_script_ch   = Channel.fromPath("${projectDir}/scripts/validate_schema.py")
    run_ablation_script_ch      = Channel.fromPath("${projectDir}/scripts/run_ablation_split.py")
    aggregate_ablation_script_ch = Channel.fromPath("${projectDir}/scripts/aggregate_ablation.py")
    calibration_script_ch       = Channel.fromPath("${projectDir}/scripts/transfer_calibration.py")
    summary_script_ch           = Channel.fromPath("${projectDir}/scripts/summary_report.py")

    // =========================================================================
    //  LOOP 1 — BASELINE PIPELINE
    // =========================================================================

    LOAD_SCREEN(
        chen_screen_ch,
        load_screen_script_ch
    )

    BUILD_FEATURES(
        LOAD_SCREEN.out.screen_parquet,
        ccle_ch,
        depmap_ch,
        reactome_ch,
        goa_ch,
        hallmarks_ch,
        kegg_ch,
        build_features_script_ch
    )

    GENERATE_SPLITS(
        LOAD_SCREEN.out.screen_parquet,
        generate_splits_script_ch
    )

    // Cross-product: each split JSON × {ridge, rf}
    split_model_ch = GENERATE_SPLITS.out.split_files
        .flatten()
        .map  { f -> [ f.baseName, f ] }
        .combine(Channel.of("ridge", "rf"))
        .map  { split_id, split_json, model -> [ split_id, model, split_json ] }

    features_subset_ch = params.features_subset_file
        ? Channel.fromPath(params.features_subset_file, checkIfExists: true)
        : Channel.value(file('NO_FILE'))

    RUN_SPLIT(
        split_model_ch,
        LOAD_SCREEN.out.screen_parquet.first(),
        BUILD_FEATURES.out.features_parquet.first(),
        schema_ch.first(),
        run_split_script_ch.first(),
        features_subset_ch.first()
    )

    // Baseline on the first split only
    RUN_BASELINE(
        GENERATE_SPLITS.out.split_files.flatten().first(),
        LOAD_SCREEN.out.screen_parquet.first(),
        BUILD_FEATURES.out.features_parquet.first(),
        schema_ch.first(),
        run_baseline_script_ch.first()
    )

    // Collect all metric JSONs for validation
    all_metric_jsons_ch = RUN_SPLIT.out.metrics_json
        .mix(RUN_BASELINE.out.baseline_json)
        .collect()

    VALIDATE_SCHEMA(
        all_metric_jsons_ch,
        schema_ch.first(),
        validate_schema_script_ch.first()
    )

    // Collect all row CSVs for aggregation
    all_row_csvs_ch = RUN_SPLIT.out.metrics_row
        .mix(RUN_BASELINE.out.baseline_row)
        .collect()

    AGGREGATE_RESULTS(
        all_row_csvs_ch,
        aggregate_results_script_ch.first()
    )

    // =========================================================================
    //  LOOP 2 — FEATURE ABLATION
    // =========================================================================

    ablation_ch = GENERATE_SPLITS.out.split_files
        .flatten()
        .map  { f -> [ f.baseName, f ] }
        .combine(Channel.of(
            "molm13_log_tpm",
            "coessential_mean_r_top50",
            "coessential_molm13_chronos",
            "n_reactome_pathways",
            "n_go_bp_terms",
            "n_go_mf_terms",
            "in_hallmark_apoptosis",
            "in_hallmark_oxidative_phosphorylation",
            "n_kegg_pathways"
        ))
        .map { split_id, split_json, feature -> [ split_id, feature, split_json ] }

    RUN_ABLATION(
        ablation_ch,
        LOAD_SCREEN.out.screen_parquet.first(),
        BUILD_FEATURES.out.features_parquet.first(),
        run_ablation_script_ch.first()
    )

    AGGREGATE_ABLATION(
        RUN_ABLATION.out.ablation_row.collect(),
        AGGREGATE_RESULTS.out.summary_csv,
        aggregate_ablation_script_ch.first()
    )

    // =========================================================================
    //  LOOP 3 — REDUCED FEATURE MODEL
    // =========================================================================

    reduced_split_ch = GENERATE_SPLITS.out.split_files
        .flatten()
        .map  { f -> [ f.baseName, f ] }
        .combine(Channel.of("ridge", "rf"))
        .map  { split_id, split_json, model -> [ split_id, model, split_json ] }
        .combine(AGGREGATE_ABLATION.out.top_features)

    RUN_REDUCED_SPLIT(
        reduced_split_ch,
        LOAD_SCREEN.out.screen_parquet.first(),
        BUILD_FEATURES.out.features_parquet.first(),
        schema_ch.first(),
        run_split_script_ch.first()
    )

    AGGREGATE_REDUCED(
        RUN_REDUCED_SPLIT.out.metrics_row.collect(),
        aggregate_results_script_ch.first()
    )

    // =========================================================================
    //  LOOP 4 — TRANSFER CALIBRATION
    // =========================================================================

    TRANSFER_CALIBRATION(
        GENERATE_SPLITS.out.split_files.flatten().collect(),
        LOAD_SCREEN.out.screen_parquet.first(),
        BUILD_FEATURES.out.features_parquet.first(),
        calibration_script_ch.first()
    )

    // =========================================================================
    //  LOOP 5 — SUMMARY REPORT
    // =========================================================================

    // Collect all results into one channel; SUMMARY_REPORT reads them by name
    summary_inputs_ch = AGGREGATE_RESULTS.out.summary_csv
        .mix(AGGREGATE_ABLATION.out.ablation_csv)
        .mix(AGGREGATE_ABLATION.out.ablation_summary_csv)
        .mix(AGGREGATE_REDUCED.out.summary_csv)
        .mix(TRANSFER_CALIBRATION.out.calibration_csv)
        .collect()

    SUMMARY_REPORT(
        summary_inputs_ch,
        summary_script_ch.first()
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PROCESS DEFINITIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// ─── Loop 1: data preparation ─────────────────────────────────────────────────

process LOAD_SCREEN {
    // Load Chen 2019 venetoclax screen; z-score normalise; assign hit labels

    publishDir params.artifacts_dir, mode: 'copy'

    input:
    path screen_file
    path script_file

    output:
    path "screen_scores.parquet", emit: screen_parquet

    script:
    """
    python ${script_file} \\
        --screen-path ${screen_file} \\
        --output screen_scores.parquet
    """

    stub:
    """
    python -c "import pandas as pd; pd.DataFrame({'gene_symbol': ['A','B'], 'cs': [1.0, -1.0], 'score_norm': [0.5, -0.5], 'is_hit_sensitizer': [False, True], 'is_hit_resistor': [False, False]}).set_index('gene_symbol').to_parquet('screen_scores.parquet')"
    """
}


process BUILD_FEATURES {
    // Build 9-feature gene matrix (CCLE expression, DepMap co-essentiality, pathways)

    publishDir params.artifacts_dir, mode: 'copy'

    input:
    path screen_parquet
    path ccle_file
    path depmap_file
    path reactome_file
    path goa_file
    path hallmarks_file
    path kegg_file
    path script_file

    output:
    path "gene_features.parquet", emit: features_parquet

    script:
    def quick_flag = params.quick_features ? "--quick" : ""
    """
    python ${script_file} \\
        --screen-parquet ${screen_parquet} \\
        --ccle-path      ${ccle_file} \\
        --depmap-path    ${depmap_file} \\
        --reactome-path  ${reactome_file} \\
        --goa-path       ${goa_file} \\
        --hallmarks-path ${hallmarks_file} \\
        --kegg-path      ${kegg_file} \\
        --output gene_features.parquet \\
        ${quick_flag}
    """

    stub:
    """
    python -c "
import pandas as pd, numpy as np
genes = ['A', 'B']
cols = ['molm13_log_tpm','coessential_mean_r_top50','coessential_molm13_chronos',
        'n_reactome_pathways','n_go_bp_terms','n_go_mf_terms',
        'in_hallmark_apoptosis','in_hallmark_oxidative_phosphorylation','n_kegg_pathways']
pd.DataFrame(np.zeros((2,9)), index=pd.Index(genes, name='gene_symbol'), columns=cols).to_parquet('gene_features.parquet')
"
    """
}


process GENERATE_SPLITS {
    // Generate n_repeats random gene holdout splits

    publishDir "${params.artifacts_dir}/splits", mode: 'copy', saveAs: { fn ->
        fn.endsWith(".csv") ? "../${fn}" : fn
    }

    input:
    path screen_parquet
    path script_file

    output:
    path "split_manifest.csv",     emit: manifest
    path "splits/aim1_random_*.json", emit: split_files

    script:
    """
    python ${script_file} \\
        --screen-parquet ${screen_parquet} \\
        --n-repeats  ${params.n_repeats} \\
        --train-size ${params.train_size} \\
        --output-dir .
    """

    stub:
    """
    mkdir -p splits
    echo "split_id,seed" > split_manifest.csv
    for i in 001 002; do
        echo '{"split_id":"aim1_random_chen2019_1393_r'"\${i}"'","generator_id":"aim1_random_gene_holdout","family":"random_gene_holdout","aim":"aim1_venetoclax","metrics_profile":"aim1_transfer","seed":11001,"repeat_index":1,"train_screen_id":"chen2019_1393","test_screen_id":"chen2019_1393","split_hash":"abcd1234abcd1234","train_genes":["A"],"test_genes":["B"]}' \\
            > splits/aim1_random_chen2019_1393_r\${i}.json
    done
    """
}


process RUN_SPLIT {
    // Train Ridge or RF on one split; compute metrics; output JSON + row CSV + predictions CSV

    tag "${split_id}:${model}"

    publishDir "${params.artifacts_dir}/metrics", mode: 'copy', saveAs: { fn ->
        fn.endsWith('.json') ? fn : null
    }

    input:
    tuple val(split_id), val(model), path(split_json)
    path screen_parquet
    path features_parquet
    path schema_json
    path script_file
    path features_subset_file

    output:
    path "${split_id}_${model}.json",      emit: metrics_json
    path "${split_id}_${model}_row.csv",   emit: metrics_row
    path "${split_id}_${model}_preds.csv", emit: predictions

    script:
    def subset_flag = (features_subset_file.name != 'NO_FILE') ? "--features-subset-file ${features_subset_file}" : ""
    """
    python ${script_file} \\
        --split-json      ${split_json} \\
        --screen-parquet  ${screen_parquet} \\
        --features-parquet ${features_parquet} \\
        --schema-json     ${schema_json} \\
        --model           ${model} \\
        --split-id        ${split_id} \\
        --n-estimators    ${params.n_estimators} \\
        ${subset_flag}
    """

    stub:
    """
    echo '{"schema_version":"1.0.0","run_id":"stub","timestamp_utc":"2026-01-01T00:00:00Z","code_commit":"abc1234","split":{"split_id":"${split_id}","generator_id":"aim1_random_gene_holdout","family":"random_gene_holdout","aim":"aim1_venetoclax","metrics_profile":"aim1_transfer","seed":11001,"split_hash":"abcd1234abcd1234"},"data_counts":{"train_row_count":2000,"test_row_count":100,"n_unique_train_genes":2000,"n_unique_test_genes":100},"leakage_checks":{"disjoint_gene_label_rows":true,"normalization_fit_on_train_only":true,"split_hash_logged":true},"metrics":{"regression":{"pearson":0.1},"ranking":{"k_metrics":[{"k":50,"n":50,"precision_at_k":0.1,"recall_at_k":0.1}]},"classification":{"labels":[{"label":"sensitizer","auroc":0.6,"auprc":0.1}]}}}' > ${split_id}_${model}.json
    echo "model,split_id,pearson" > ${split_id}_${model}_row.csv
    echo "${model},${split_id},0.1" >> ${split_id}_${model}_row.csv
    echo "gene_symbol,y_test,y_pred" > ${split_id}_${model}_preds.csv
    """
}


process RUN_BASELINE {
    // Zero-predictor baseline on the first split

    publishDir "${params.artifacts_dir}/metrics", mode: 'copy'

    input:
    path split_json
    path screen_parquet
    path features_parquet
    path schema_json
    path script_file

    output:
    path "baseline_zero.json", emit: baseline_json
    path "baseline_row.csv",   emit: baseline_row

    script:
    """
    python ${script_file} \\
        --split-json      ${split_json} \\
        --screen-parquet  ${screen_parquet} \\
        --features-parquet ${features_parquet} \\
        --schema-json     ${schema_json} \\
        --output baseline_zero.json

    # Create a row CSV for aggregation
    python -c "
import json, pandas as pd
with open('baseline_zero.json') as f:
    rec = json.load(f)
m = rec['metrics']
r = m['regression']
c = m['classification']['labels']
k = m['ranking']['k_metrics']
km = {x['k']: x for x in k}
row = {'model': 'baseline_zero',
       'split_id': rec['split']['split_id'],
       'pearson': r['pearson'], 'spearman': r.get('spearman', 0),
       'auroc_sensitizer': c[0]['auroc'] if c else None,
       'precision_at_50': km.get(50, {}).get('precision_at_k', None)}
pd.DataFrame([row]).to_csv('baseline_row.csv', index=False)
"
    """

    stub:
    """
    echo '{"schema_version":"1.0.0","run_id":"aim1_baseline_zero","timestamp_utc":"2026-01-01T00:00:00Z","code_commit":"abc1234","split":{"split_id":"stub_baseline","generator_id":"aim1_random_gene_holdout","family":"random_gene_holdout","aim":"aim1_venetoclax","metrics_profile":"aim1_transfer","seed":11001,"split_hash":"abcd1234abcd1234"},"data_counts":{"train_row_count":2000,"test_row_count":100,"n_unique_train_genes":2000,"n_unique_test_genes":100},"leakage_checks":{"disjoint_gene_label_rows":true,"normalization_fit_on_train_only":true,"split_hash_logged":true},"metrics":{"regression":{"pearson":0.0},"ranking":{"k_metrics":[{"k":50,"n":50,"precision_at_k":0.05,"recall_at_k":0.05}]},"classification":{"labels":[{"label":"sensitizer","auroc":0.5,"auprc":0.05}]}}}' > baseline_zero.json
    echo "model,split_id,pearson" > baseline_row.csv
    echo "baseline_zero,stub_baseline,0.0" >> baseline_row.csv
    """
}


process VALIDATE_SCHEMA {
    // Validate all 51 metric JSONs against metrics.schema.json

    publishDir params.results_dir, mode: 'copy'

    input:
    path metric_jsons
    path schema_json
    path script_file

    output:
    path "validation_report.txt", emit: report

    script:
    """
    cp ${schema_json} metrics_schema.json
    python ${script_file} \\
        --schema-json metrics_schema.json \\
        --output validation_report.txt
    """

    stub:
    """
    echo "PASS  stub_baseline.json" > validation_report.txt
    echo "Schema validation: 1 PASS / 0 FAIL / 1 total" >> validation_report.txt
    """
}


process AGGREGATE_RESULTS {
    // Aggregate per-split row CSVs → per-model CSVs + BCa CI summary

    publishDir params.results_dir, mode: 'copy'

    input:
    path row_csvs
    path script_file

    output:
    path "design_a_results_ridge.csv", emit: ridge_csv
    path "design_a_results_rf.csv",    emit: rf_csv
    path "design_a_results_all.csv",   emit: all_csv
    path "design_a_summary.csv",       emit: summary_csv

    script:
    """
    python ${script_file} --tag design_a
    """

    stub:
    """
    echo "model,pearson_mean" > design_a_results_ridge.csv
    echo "ridge,0.1" >> design_a_results_ridge.csv
    cp design_a_results_ridge.csv design_a_results_rf.csv
    cp design_a_results_ridge.csv design_a_results_all.csv
    echo "model,pearson_mean,pearson_ci_lo,pearson_ci_hi" > design_a_summary.csv
    echo "ridge,0.1,0.08,0.12" >> design_a_summary.csv
    echo "rf,0.12,0.1,0.14" >> design_a_summary.csv
    """
}


// ─── Loop 2: feature ablation ─────────────────────────────────────────────────

process RUN_ABLATION {
    // Ridge with one feature dropped (leave-one-out ablation, one split)

    tag "${split_id}:drop=${dropped_feature}"

    input:
    tuple val(split_id), val(dropped_feature), path(split_json)
    path screen_parquet
    path features_parquet
    path script_file

    output:
    path "${split_id}_${dropped_feature}_ablation_row.json", emit: ablation_row

    script:
    """
    python ${script_file} \\
        --split-json      ${split_json} \\
        --screen-parquet  ${screen_parquet} \\
        --features-parquet ${features_parquet} \\
        --drop-feature    ${dropped_feature} \\
        --split-id        ${split_id}
    """

    stub:
    """
    echo '{"split_id":"${split_id}","dropped_feature":"${dropped_feature}","n_features_used":8,"pearson":0.1,"auroc_sensitizer":0.6,"auprc_sensitizer":0.05,"precision_at_50":0.1,"recall_at_50":0.05,"precision_at_100":0.09,"precision_at_200":0.08,"precision_at_500":0.07}' > ${split_id}_${dropped_feature}_ablation_row.json
    """
}


process AGGREGATE_ABLATION {
    // Collect ablation rows → CSV + identify top features to keep

    publishDir params.results_dir, mode: 'copy'

    input:
    path ablation_rows
    path baseline_summary_csv
    path script_file

    output:
    path "feature_ablation_ridge.csv", emit: ablation_csv
    path "ablation_summary.csv",       emit: ablation_summary_csv
    path "top_features.txt",           emit: top_features

    script:
    """
    python ${script_file} \\
        --baseline-summary-csv ${baseline_summary_csv} \\
        --top-k                ${params.top_k_features} \\
        --drop-threshold       ${params.ablation_drop_threshold}
    """

    stub:
    """
    echo "dropped_feature,mean_precision_at_50" > feature_ablation_ridge.csv
    echo "molm13_log_tpm,0.08" >> feature_ablation_ridge.csv
    cp feature_ablation_ridge.csv ablation_summary.csv
    printf "molm13_log_tpm\\ncoessential_mean_r_top50\\ncoessential_molm13_chronos\\n" > top_features.txt
    """
}


// ─── Loop 3: reduced feature model ────────────────────────────────────────────

process RUN_REDUCED_SPLIT {
    // Train Ridge or RF using only the top features identified by ablation

    tag "${split_id}:${model}:reduced"

    publishDir "${params.artifacts_dir}/metrics_reduced", mode: 'copy', saveAs: { fn ->
        fn.endsWith('.json') ? fn : null
    }

    input:
    tuple val(split_id), val(model), path(split_json), path(top_features_file)
    path screen_parquet
    path features_parquet
    path schema_json
    path script_file

    output:
    path "${split_id}_${model}_reduced.json",    emit: metrics_json
    path "${split_id}_${model}_reduced_row.csv", emit: metrics_row

    script:
    """
    python ${script_file} \\
        --split-json           ${split_json} \\
        --screen-parquet       ${screen_parquet} \\
        --features-parquet     ${features_parquet} \\
        --schema-json          ${schema_json} \\
        --model                ${model} \\
        --split-id             ${split_id}_reduced \\
        --n-estimators         ${params.n_estimators} \\
        --features-subset-file ${top_features_file}

    # Rename outputs to _reduced suffix
    mv ${split_id}_reduced_${model}.json     ${split_id}_${model}_reduced.json
    mv ${split_id}_reduced_${model}_row.csv  ${split_id}_${model}_reduced_row.csv
    """

    stub:
    """
    echo '{"schema_version":"1.0.0","run_id":"stub_reduced"}' > ${split_id}_${model}_reduced.json
    echo "model,split_id,pearson" > ${split_id}_${model}_reduced_row.csv
    echo "${model},${split_id}_reduced,0.09" >> ${split_id}_${model}_reduced_row.csv
    """
}


process AGGREGATE_REDUCED {
    // Aggregate reduced-model row CSVs → per-model CSVs + BCa CI summary

    publishDir params.results_dir, mode: 'copy'

    input:
    path row_csvs
    path script_file

    output:
    path "design_a_reduced_results_ridge.csv", emit: ridge_csv
    path "design_a_reduced_results_rf.csv",    emit: rf_csv
    path "design_a_reduced_results_all.csv",   emit: all_csv
    path "design_a_reduced_summary.csv",       emit: summary_csv

    script:
    """
    python ${script_file} --tag design_a_reduced
    """

    stub:
    """
    echo "model,pearson_mean" > design_a_reduced_results_ridge.csv
    echo "ridge,0.09" >> design_a_reduced_results_ridge.csv
    cp design_a_reduced_results_ridge.csv design_a_reduced_results_rf.csv
    cp design_a_reduced_results_ridge.csv design_a_reduced_results_all.csv
    echo "model,pearson_mean" > design_a_reduced_summary.csv
    echo "ridge,0.09" >> design_a_reduced_summary.csv
    """
}


// ─── Loop 4: transfer calibration ─────────────────────────────────────────────

process TRANSFER_CALIBRATION {
    // Full calibration: score distributions, stratified rank corr, RF importances

    publishDir params.results_dir, mode: 'copy'

    input:
    path split_jsons
    path screen_parquet
    path features_parquet
    path script_file

    output:
    path "transfer_calibration_design_a.csv", emit: calibration_csv
    path "figure_score_dist.png",             emit: fig_score_dist
    path "figure_stratified_spearman.png",    emit: fig_stratified
    path "figure_feature_importance.png",     emit: fig_importance

    script:
    """
    python ${script_file} \\
        --split-jsons     ${split_jsons.join(' ')} \\
        --screen-parquet  ${screen_parquet} \\
        --features-parquet ${features_parquet} \\
        --n-estimators    ${params.n_estimators} \\
        --output-csv      transfer_calibration_design_a.csv
    """

    stub:
    """
    echo "split_id,stratum,n,spearman_r" > transfer_calibration_design_a.csv
    echo "stub,hit_sensitizer,50,0.15" >> transfer_calibration_design_a.csv
    touch figure_score_dist.png figure_stratified_spearman.png figure_feature_importance.png
    """
}


// ─── Loop 5: summary report ───────────────────────────────────────────────────

process SUMMARY_REPORT {
    // Synthesise all findings into design_a_report.md

    publishDir params.results_dir, mode: 'copy'

    input:
    path results_files
    path script_file

    output:
    path "design_a_report.md", emit: report

    script:
    """
    python ${script_file} \\
        --results-dir . \\
        --output design_a_report.md
    """

    stub:
    """
    echo "# Design A Report (stub)" > design_a_report.md
    """
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

def helpMessage() {
    log.info"""
    Usage:
      nextflow run main.nf [options]

    Options:
      --help                    Show this message and exit

    Data paths (all have defaults in nextflow.config):
      --chen_screen_path        Chen 2019 BioGRID-ORCS screen TSV
      --ccle_path               CCLE expression CSV.gz
      --depmap_path             DepMap CRISPRGeneEffect CSV.gz
      --reactome_path           NCBI2Reactome_PE_Pathway file
      --goa_path                goa_human.gaf.gz
      --hallmarks_path          Hallmarks GMT.gz
      --kegg_path               KEGG GMT.gz
      --schema_path             metrics.schema.json

    Output directories:
      --artifacts_dir           Where to publish parquets, splits, metric JSONs
      --results_dir             Where to publish aggregate CSVs and figures

    Run parameters:
      --n_repeats               Number of random splits (default: 25)
      --train_size              Training set size per split (default: 2000)
      --n_estimators            RF trees (default: 200)
      --top_k_features          Features to keep in reduced model (default: 5)
      --ablation_drop_threshold Minimum relative P@50 drop to flag as important (default: 0.05)
      --quick_features          Zero-impute DepMap features (fast test mode)

    Profiles:
      -profile standard         Local execution
      -profile slurm            SLURM cluster execution
      -profile test             Small run with n_repeats=2 and --quick
    """.stripIndent()
}
