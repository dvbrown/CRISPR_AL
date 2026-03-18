#!/usr/bin/env nextflow

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    crispr-al/design_b
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Design B: Cross-Screen Transfer — 3-Loop Autonomous Research Pipeline

    Loop 1  Core pipeline   — 60 splits × 2 models + 2 baselines → 122 metric JSONs
    Loop 2  Transfer calib. — stratified Spearman ρ, feature importances, 3 figures
    Loop 3  Summary report  — Markdown synthesis of all findings
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
    sharon_screen_raw_ch = Channel.fromPath(params.sharon_screen_path, checkIfExists: true)
    schema_ch            = Channel.fromPath(params.schema_path,        checkIfExists: true)

    // Pre-built Chen artifacts (reuse from Design A run)
    chen_parquet_ch   = Channel.fromPath("${params.artifacts_dir}/screen_scores.parquet",
                                          checkIfExists: true)
    features_parquet_ch = Channel.fromPath("${params.artifacts_dir}/gene_features.parquet",
                                            checkIfExists: true)

    // -------------------------------------------------------------------------
    //  Script file channels
    // -------------------------------------------------------------------------
    load_sharon_script_ch        = Channel.fromPath("${projectDir}/scripts/load_sharon_screen.py")
    generate_cross_splits_ch     = Channel.fromPath("${projectDir}/scripts/generate_cross_splits.py")
    run_cross_split_ch           = Channel.fromPath("${projectDir}/scripts/run_cross_split.py")
    run_cross_baseline_ch        = Channel.fromPath("${projectDir}/scripts/run_cross_baseline.py")
    validate_schema_script_ch    = Channel.fromPath("${projectDir}/scripts/validate_schema.py")
    aggregate_cross_results_ch   = Channel.fromPath("${projectDir}/scripts/aggregate_cross_results.py")
    calibration_script_ch        = Channel.fromPath("${projectDir}/scripts/transfer_calibration.py")
    summary_script_ch            = Channel.fromPath("${projectDir}/scripts/summary_report_b.py")

    // Optimal feature set written by Phase A
    design_b_features_ch = params.design_b_features_file
        ? Channel.fromPath(params.design_b_features_file, checkIfExists: true)
        : Channel.value(file('NO_FILE'))

    // =========================================================================
    //  LOOP 1 — CORE PIPELINE
    // =========================================================================

    LOAD_SHARON_SCREEN(
        sharon_screen_raw_ch,
        load_sharon_script_ch
    )

    GENERATE_CROSS_SPLITS(
        chen_parquet_ch.first(),
        LOAD_SHARON_SCREEN.out.sharon_parquet.first(),
        generate_cross_splits_ch
    )

    // Cross-product: each split JSON × {ridge, rf}
    cross_split_model_ch = GENERATE_CROSS_SPLITS.out.split_files
        .flatten()
        .map  { f -> [ f.baseName, f ] }
        .combine(Channel.of("ridge", "rf"))
        .map  { split_id, split_json, model -> [ split_id, model, split_json ] }

    RUN_CROSS_SPLIT(
        cross_split_model_ch,
        chen_parquet_ch.first(),
        LOAD_SHARON_SCREEN.out.sharon_parquet.first(),
        features_parquet_ch.first(),
        schema_ch.first(),
        run_cross_split_ch.first(),
        design_b_features_ch.first()
    )

    RUN_CROSS_BASELINE(
        chen_parquet_ch.first(),
        LOAD_SHARON_SCREEN.out.sharon_parquet.first(),
        features_parquet_ch.first(),
        schema_ch.first(),
        run_cross_baseline_ch.first(),
        design_b_features_ch.first()
    )

    // Collect all metric JSONs for validation
    all_metric_jsons_ch = RUN_CROSS_SPLIT.out.metrics_json
        .mix(RUN_CROSS_BASELINE.out.baseline_json_c2s)
        .mix(RUN_CROSS_BASELINE.out.baseline_json_s2c)
        .collect()

    VALIDATE_SCHEMA_B(
        all_metric_jsons_ch,
        schema_ch.first(),
        validate_schema_script_ch.first()
    )

    // Collect all row CSVs for aggregation
    all_row_csvs_ch = RUN_CROSS_SPLIT.out.metrics_row
        .mix(RUN_CROSS_BASELINE.out.baseline_row_c2s)
        .mix(RUN_CROSS_BASELINE.out.baseline_row_s2c)
        .collect()

    AGGREGATE_CROSS_RESULTS(
        all_row_csvs_ch,
        aggregate_cross_results_ch.first()
    )

    // =========================================================================
    //  LOOP 2 — TRANSFER CALIBRATION (per direction)
    // =========================================================================

    TRANSFER_CALIBRATION_C2S(
        GENERATE_CROSS_SPLITS.out.split_files
            .flatten()
            .filter { f -> f.baseName.contains("chen2019_1393_to_sharon") }
            .collect(),
        chen_parquet_ch.first(),
        LOAD_SHARON_SCREEN.out.sharon_parquet.first(),
        features_parquet_ch.first(),
        calibration_script_ch.first(),
        design_b_features_ch.first()
    )

    TRANSFER_CALIBRATION_S2C(
        GENERATE_CROSS_SPLITS.out.split_files
            .flatten()
            .filter { f -> f.baseName.contains("sharon2019_1402_to_chen") }
            .collect(),
        LOAD_SHARON_SCREEN.out.sharon_parquet.first(),
        chen_parquet_ch.first(),
        features_parquet_ch.first(),
        calibration_script_ch.first(),
        design_b_features_ch.first()
    )

    // =========================================================================
    //  LOOP 3 — SUMMARY REPORT
    // =========================================================================

    summary_inputs_ch = AGGREGATE_CROSS_RESULTS.out.summary_csv
        .mix(AGGREGATE_CROSS_RESULTS.out.all_csv)
        .mix(TRANSFER_CALIBRATION_C2S.out.calibration_csv)
        .mix(TRANSFER_CALIBRATION_S2C.out.calibration_csv)
        .collect()

    SUMMARY_REPORT_B(
        summary_inputs_ch,
        summary_script_ch.first()
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PROCESS DEFINITIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// ─── Loop 1: data preparation ──────────────────────────────────────────────────

process LOAD_SHARON_SCREEN {
    // Load Sharon 2019 venetoclax screen; z-score normalise; assign hit labels

    publishDir params.artifacts_dir, mode: 'copy'

    input:
    path sharon_screen_file
    path script_file

    output:
    path "sharon_scores.parquet", emit: sharon_parquet

    script:
    """
    python ${script_file} \\
        --sharon-screen-path ${sharon_screen_file} \\
        --output sharon_scores.parquet
    """

    stub:
    """
    python -c "
import pandas as pd
pd.DataFrame({
    'entrez_id': [1, 2], 'lfc': [0.5, -0.5],
    'neg_fdr': [0.1, 0.05], 'pos_fdr': [0.05, 0.1],
    'score_norm': [0.5, -0.5],
    'is_hit_sensitizer': [False, True], 'is_hit_resistor': [False, False]
}, index=pd.Index(['A', 'B'], name='gene_symbol')).to_parquet('sharon_scores.parquet')
"
    """
}


process GENERATE_CROSS_SPLITS {
    // Generate 30 chen→sharon + 30 sharon→chen cross-screen splits

    publishDir "${params.artifacts_dir}/splits_b", mode: 'copy', saveAs: { fn ->
        fn.endsWith(".csv") ? "../${fn}" : fn
    }

    input:
    path chen_parquet
    path sharon_parquet
    path script_file

    output:
    path "cross_split_manifest.csv",     emit: manifest
    path "splits/aim1_xfer_*.json",      emit: split_files

    script:
    """
    python ${script_file} \\
        --chen-screen   ${chen_parquet} \\
        --sharon-screen ${sharon_parquet} \\
        --n-repeats     ${params.n_repeats_b} \\
        --train-size    ${params.train_size} \\
        --output-dir    .
    """

    stub:
    """
    mkdir -p splits
    echo "split_id,seed" > cross_split_manifest.csv
    for pair in "chen2019_1393_to_sharon2019_1402_r001 chen2019_1393_to_sharon2019_1402_r002 sharon2019_1402_to_chen2019_1393_r001 sharon2019_1402_to_chen2019_1393_r002"; do
        for sid in \$pair; do
            echo '{"split_id":"aim1_xfer_'"${sid}"'","generator_id":"aim1_cross_screen_transfer","family":"context_zeroshot","aim":"aim1_venetoclax","metrics_profile":"aim1_transfer","seed":21001,"repeat_index":1,"train_screen_id":"chen2019_1393","test_screen_id":"sharon2019_1402","split_hash":"abcd1234abcd1234","train_genes":["A"],"test_genes":["B"]}' \\
                > splits/aim1_xfer_\${sid}.json
        done
    done
    """
}


process RUN_CROSS_SPLIT {
    // Train Ridge or RF on one cross-screen split; compute metrics

    tag "${split_id}:${model}"

    publishDir "${params.artifacts_dir}/metrics_b", mode: 'copy', saveAs: { fn ->
        fn.endsWith('.json') ? fn : null
    }

    input:
    tuple val(split_id), val(model), path(split_json)
    path chen_parquet
    path sharon_parquet
    path features_parquet
    path schema_json
    path script_file
    path features_subset_file

    output:
    path "${split_id}_${model}.json",      emit: metrics_json
    path "${split_id}_${model}_row.csv",   emit: metrics_row
    path "${split_id}_${model}_preds.csv", emit: predictions

    script:
    // Route train/test parquets based on split direction
    def is_c2s = split_id.contains("chen2019_1393_to_sharon")
    def train_parquet = is_c2s ? chen_parquet : sharon_parquet
    def test_parquet  = is_c2s ? sharon_parquet : chen_parquet
    def subset_flag = (features_subset_file.name != 'NO_FILE') ? "--features-subset-file ${features_subset_file}" : ""
    """
    python ${script_file} \\
        --split-json             ${split_json} \\
        --train-screen-parquet   ${train_parquet} \\
        --test-screen-parquet    ${test_parquet} \\
        --features-parquet       ${features_parquet} \\
        --schema-json            ${schema_json} \\
        --model                  ${model} \\
        --split-id               ${split_id} \\
        --n-estimators           ${params.n_estimators} \\
        ${subset_flag}
    """

    stub:
    """
    echo '{"schema_version":"1.0.0","run_id":"stub","timestamp_utc":"2026-01-01T00:00:00Z","code_commit":"abc1234","split":{"split_id":"${split_id}","generator_id":"aim1_cross_screen_transfer","family":"context_zeroshot","aim":"aim1_venetoclax","metrics_profile":"aim1_transfer","seed":21001,"split_hash":"abcd1234abcd1234"},"data_counts":{"train_row_count":2000,"test_row_count":100,"n_unique_train_genes":2000,"n_unique_test_genes":100,"n_overlap_genes_train_test":0},"leakage_checks":{"disjoint_gene_label_rows":true,"normalization_fit_on_train_only":true,"split_hash_logged":true},"metrics":{"regression":{"pearson":0.1},"ranking":{"k_metrics":[{"k":50,"n":50,"precision_at_k":0.1,"recall_at_k":0.1}]},"classification":{"labels":[{"label":"sensitizer","auroc":0.6,"auprc":0.1}]}}}' > ${split_id}_${model}.json
    echo "model,direction,split_id,pearson" > ${split_id}_${model}_row.csv
    echo "${model},chen_to_sharon,${split_id},0.1" >> ${split_id}_${model}_row.csv
    echo "gene_symbol,y_test,y_pred" > ${split_id}_${model}_preds.csv
    """
}


process RUN_CROSS_BASELINE {
    // Overlap-only baseline: train on shared genes, test on each full screen

    publishDir "${params.artifacts_dir}/metrics_b", mode: 'copy'

    input:
    path chen_parquet
    path sharon_parquet
    path features_parquet
    path schema_json
    path script_file
    path features_subset_file

    output:
    path "baseline_chen_to_sharon.json",     emit: baseline_json_c2s
    path "baseline_sharon_to_chen.json",     emit: baseline_json_s2c
    path "baseline_chen_to_sharon_row.csv",  emit: baseline_row_c2s
    path "baseline_sharon_to_chen_row.csv",  emit: baseline_row_s2c

    script:
    def subset_flag = (features_subset_file.name != 'NO_FILE') ? "--features-subset-file ${features_subset_file}" : ""
    """
    python ${script_file} \\
        --chen-screen-parquet    ${chen_parquet} \\
        --sharon-screen-parquet  ${sharon_parquet} \\
        --features-parquet       ${features_parquet} \\
        --schema-json            ${schema_json} \\
        ${subset_flag}
    """

    stub:
    """
    echo '{"schema_version":"1.0.0","run_id":"aim1_overlap_baseline_chen2019_1393_to_sharon2019_1402","timestamp_utc":"2026-01-01T00:00:00Z","code_commit":"abc1234","split":{"split_id":"aim1_overlap_baseline_chen2019_1393_to_sharon2019_1402","generator_id":"aim1_overlap_baseline","family":"overlap_baseline","aim":"aim1_venetoclax","metrics_profile":"aim1_transfer","seed":0,"split_hash":"abcd1234abcd1234"},"data_counts":{"train_row_count":100,"test_row_count":200,"n_unique_train_genes":100,"n_unique_test_genes":200,"n_overlap_genes_train_test":100},"leakage_checks":{"disjoint_gene_label_rows":false,"normalization_fit_on_train_only":true,"split_hash_logged":true},"metrics":{"regression":{"pearson":0.05},"ranking":{"k_metrics":[{"k":50,"n":50,"precision_at_k":0.07,"recall_at_k":0.07}]},"classification":{"labels":[{"label":"sensitizer","auroc":0.52,"auprc":0.07}]}}}' > baseline_chen_to_sharon.json
    cp baseline_chen_to_sharon.json baseline_sharon_to_chen.json
    echo "model,direction,split_id,pearson" > baseline_chen_to_sharon_row.csv
    echo "overlap_baseline,chen_to_sharon,aim1_overlap_baseline_chen2019_1393_to_sharon2019_1402,0.05" >> baseline_chen_to_sharon_row.csv
    cp baseline_chen_to_sharon_row.csv baseline_sharon_to_chen_row.csv
    """
}


process VALIDATE_SCHEMA_B {
    // Validate all Design B metric JSONs against metrics.schema.json

    publishDir params.results_dir, mode: 'copy'

    input:
    path metric_jsons
    path schema_json
    path script_file

    output:
    path "validation_report_b.txt", emit: report

    script:
    """
    cp ${schema_json} metrics_schema.json
    python ${script_file} \\
        --schema-json metrics_schema.json \\
        --output validation_report_b.txt
    """

    stub:
    """
    echo "PASS  baseline_chen_to_sharon.json" > validation_report_b.txt
    echo "Schema validation: 1 PASS / 0 FAIL / 1 total" >> validation_report_b.txt
    """
}


process AGGREGATE_CROSS_RESULTS {
    // Aggregate per-split row CSVs → per-direction CSVs + BCa CI summary

    publishDir params.results_dir, mode: 'copy'

    input:
    path row_csvs
    path script_file

    output:
    path "design_b_results_chen_to_sharon.csv", emit: c2s_csv
    path "design_b_results_sharon_to_chen.csv", emit: s2c_csv
    path "design_b_results_all.csv",            emit: all_csv
    path "design_b_summary.csv",                emit: summary_csv

    script:
    """
    python ${script_file} --tag design_b
    """

    stub:
    """
    echo "direction,model,pearson_mean" > design_b_results_chen_to_sharon.csv
    echo "chen_to_sharon,ridge,0.04" >> design_b_results_chen_to_sharon.csv
    cp design_b_results_chen_to_sharon.csv design_b_results_sharon_to_chen.csv
    cp design_b_results_chen_to_sharon.csv design_b_results_all.csv
    echo "direction,model,pearson_mean,pearson_ci_lo,pearson_ci_hi" > design_b_summary.csv
    echo "chen_to_sharon,ridge,0.04,0.02,0.06" >> design_b_summary.csv
    echo "sharon_to_chen,ridge,0.18,0.15,0.21" >> design_b_summary.csv
    """
}


// ─── Loop 2: transfer calibration ─────────────────────────────────────────────

process TRANSFER_CALIBRATION_C2S {
    // Calibration for chen→sharon direction

    publishDir params.results_dir, mode: 'copy'

    input:
    path split_jsons
    path train_screen_parquet
    path test_screen_parquet
    path features_parquet
    path script_file
    path features_subset_file

    output:
    path "transfer_calibration_design_b_c2s.csv", emit: calibration_csv
    path "figure_score_dist_c2s.png",             emit: fig_score_dist
    path "figure_stratified_spearman_c2s.png",    emit: fig_stratified
    path "figure_feature_importance_c2s.png",     emit: fig_importance

    script:
    def subset_flag = (features_subset_file.name != 'NO_FILE') ? "--features-subset-file ${features_subset_file}" : ""
    """
    python ${script_file} \\
        --split-jsons            ${split_jsons.join(' ')} \\
        --train-screen-parquet   ${train_screen_parquet} \\
        --screen-parquet         ${test_screen_parquet} \\
        --features-parquet       ${features_parquet} \\
        --n-estimators           ${params.n_estimators} \\
        --output-csv             transfer_calibration_design_b_c2s.csv \\
        --figure-suffix          _c2s \\
        ${subset_flag}
    """

    stub:
    """
    echo "split_id,stratum,n,spearman_r" > transfer_calibration_design_b_c2s.csv
    echo "stub,hit_sensitizer,50,0.05" >> transfer_calibration_design_b_c2s.csv
    touch figure_score_dist_c2s.png figure_stratified_spearman_c2s.png figure_feature_importance_c2s.png
    """
}


process TRANSFER_CALIBRATION_S2C {
    // Calibration for sharon→chen direction

    publishDir params.results_dir, mode: 'copy'

    input:
    path split_jsons
    path train_screen_parquet
    path test_screen_parquet
    path features_parquet
    path script_file
    path features_subset_file

    output:
    path "transfer_calibration_design_b_s2c.csv", emit: calibration_csv
    path "figure_score_dist_s2c.png",             emit: fig_score_dist
    path "figure_stratified_spearman_s2c.png",    emit: fig_stratified
    path "figure_feature_importance_s2c.png",     emit: fig_importance

    script:
    def subset_flag = (features_subset_file.name != 'NO_FILE') ? "--features-subset-file ${features_subset_file}" : ""
    """
    python ${script_file} \\
        --split-jsons            ${split_jsons.join(' ')} \\
        --train-screen-parquet   ${train_screen_parquet} \\
        --screen-parquet         ${test_screen_parquet} \\
        --features-parquet       ${features_parquet} \\
        --n-estimators           ${params.n_estimators} \\
        --output-csv             transfer_calibration_design_b_s2c.csv \\
        --figure-suffix          _s2c \\
        ${subset_flag}
    """

    stub:
    """
    echo "split_id,stratum,n,spearman_r" > transfer_calibration_design_b_s2c.csv
    echo "stub,hit_sensitizer,50,0.15" >> transfer_calibration_design_b_s2c.csv
    touch figure_score_dist_s2c.png figure_stratified_spearman_s2c.png figure_feature_importance_s2c.png
    """
}


// ─── Loop 3: summary report ────────────────────────────────────────────────────

process SUMMARY_REPORT_B {
    // Synthesise all findings into design_b_report.md

    publishDir params.results_dir, mode: 'copy'

    input:
    path results_files
    path script_file

    output:
    path "design_b_report.md", emit: report

    script:
    """
    python ${script_file} \\
        --results-dir . \\
        --output design_b_report.md
    """

    stub:
    """
    echo "# Design B Report (stub)" > design_b_report.md
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
      nextflow run main_b.nf [options]

    Options:
      --help                    Show this message and exit

    Data paths (all have defaults in nextflow.config):
      --sharon_screen_path      Sharon 2019 BioGRID-ORCS screen TSV
      --chen_screen_path        Chen 2019 BioGRID-ORCS screen TSV (for channel creation)
      --schema_path             metrics.schema.json

    Pre-built artifacts (from Design A run):
      --artifacts_dir           Directory containing screen_scores.parquet and gene_features.parquet

    Output directories:
      --artifacts_dir           Where to publish sharon parquet, splits, metric JSONs
      --results_dir             Where to publish aggregate CSVs and figures

    Run parameters:
      --n_repeats_b             Number of cross-screen splits per direction (default: 30)
      --train_size              Training set size per split (default: 2000)
      --n_estimators            RF trees (default: 200)
      --design_b_features_file  Feature subset file from Phase A optimisation

    Profiles:
      -profile standard         Local execution
      -profile slurm            SLURM cluster execution
      -profile test             Small run with n_repeats_b=2 and fast settings
    """.stripIndent()
}
