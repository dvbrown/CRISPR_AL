import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", title="Design A: Within-Screen Holdout (Aim 1)")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md("""
    # Design A: Within-Screen Holdout (Aim 1)

    ## Problem Framing

    Can we predict **venetoclax sensitivity** of a gene knockout from gene features alone?

    We train a regressor on 2,000 randomly sampled genes from the Chen 2019 genome-wide
    venetoclax CRISPR screen, then predict sensitivity scores for the remaining ~17,000
    holdout genes. We repeat this 25 times with different random splits and report metrics
    with 95% confidence intervals.

    **Models:** Ridge regression and Random Forest regressor
    **Features per gene:** 9 (expression, co-essentiality, pathway annotations)
    **Target:** z-scored CRISPR Score (CS) from the Chen 2019 screen
    """)
    return


@app.cell
def __():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    import numpy as np
    import pandas as pd
    from pathlib import Path
    return Path, np, os, pd, sys


@app.cell
def __():
    from crispr_al.metrics import bootstrap_ci_bca
    return (bootstrap_ci_bca,)


@app.cell
def __(Path):
    ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
    RESULTS_DIR = Path(__file__).parent / "results"
    SCHEMA_PATH = Path(__file__).parent / "metrics.schema.json"
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bulk"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "splits").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR, DATA_DIR, RESULTS_DIR, SCHEMA_PATH


@app.cell
def __(mo):
    mo.md("""
    ## Section 1: Data Structures

    ### Screen Scores

    The Chen 2019 venetoclax CRISPR screen assigns a **CRISPR Score (CS)** to each gene:
    - **Negative CS** = gene knockout sensitizes cells to venetoclax
    - **Positive CS** = gene knockout confers resistance to venetoclax

    We z-score normalize the CS across all ~19,109 genes to produce `score_norm`.

    **Hit labels** use the paper's own thresholds:
    - `is_hit_sensitizer`: CS < -1.0
    - `is_hit_resistor`: CS > 3.0
    """)
    return


@app.cell
def __(ARTIFACTS_DIR, mo):
    from crispr_al.io import load_parquet

    screen_parquet_path = ARTIFACTS_DIR / "screen_scores.parquet"
    features_parquet_path = ARTIFACTS_DIR / "gene_features.parquet"

    if screen_parquet_path.exists():
        screen_df = load_parquet(str(screen_parquet_path)).reset_index()
        mo.md(f"✓ Loaded screen_scores.parquet: **{len(screen_df):,} genes**")
    else:
        screen_df = None
        mo.md("⚠ screen_scores.parquet not found. Run the Nextflow pipeline first:\n```\nnextflow run pipelines/design_a/main.nf\n```")
    return features_parquet_path, load_parquet, screen_df, screen_parquet_path


@app.cell
def __(mo, screen_df):
    if screen_df is not None:
        n_sensitizers = screen_df["is_hit_sensitizer"].sum()
        n_resistors = screen_df["is_hit_resistor"].sum()
        mo.md(f"""
        **Screen statistics:**
        - Total genes: {len(screen_df):,}
        - Sensitizers (CS < -1.0): {n_sensitizers:,} ({100*n_sensitizers/len(screen_df):.1f}%)
        - Resistors (CS > 3.0): {n_resistors:,} ({100*n_resistors/len(screen_df):.1f}%)
        """)
    return n_resistors, n_sensitizers


@app.cell
def __(features_parquet_path, load_parquet, mo):
    if features_parquet_path.exists():
        features_df = load_parquet(str(features_parquet_path))
        mo.md(f"✓ Loaded gene_features.parquet: **{features_df.shape[0]:,} genes × {features_df.shape[1]} features**")
    else:
        features_df = None
        mo.md("⚠ gene_features.parquet not found. Run the Nextflow pipeline first.")
    return (features_df,)


@app.cell
def __(mo):
    mo.md("""
    ## Section 2: Feature Descriptions

    Each gene is represented by **9 biological features**:

    | Feature | Description |
    |---------|-------------|
    | `molm13_log_tpm` | MOLM-13 expression level log₂(TPM+1) from CCLE |
    | `coessential_mean_r_top50` | Mean Pearson r with top-50 co-essential genes across DepMap |
    | `coessential_molm13_chronos` | Raw Chronos score in MOLM-13 (baseline essentiality) |
    | `n_reactome_pathways` | Number of Reactome pathway memberships |
    | `n_go_bp_terms` | Number of non-IEA GO Biological Process annotations |
    | `n_go_mf_terms` | Number of non-IEA GO Molecular Function annotations |
    | `in_hallmark_apoptosis` | Member of HALLMARK_APOPTOSIS gene set (0/1) |
    | `in_hallmark_oxidative_phosphorylation` | Member of HALLMARK_OXIDATIVE_PHOSPHORYLATION (0/1) |
    | `n_kegg_pathways` | Number of KEGG pathway memberships |

    **Biological rationale:** Genes with high expression in MOLM-13, strong co-essentiality
    with other venetoclax-relevant genes, or membership in apoptosis pathways are expected
    to be enriched for venetoclax sensitivity hits.
    """)
    return


@app.cell
def __(features_df, mo):
    if features_df is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        corr_matrix = features_df.corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr_matrix.columns, fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        mo.pyplot(fig)
    return corr_matrix, fig


@app.cell
def __(mo):
    mo.md("""
    ## Section 3: Split Generation

    We generate **25 independent random splits** of the ~19,109 genes:
    - **Train set:** 2,000 randomly sampled genes
    - **Test set:** remaining ~17,109 genes

    Each split gets a unique SHA-256 hash for reproducibility and auditability.
    Seeds are deterministically incremented from SEED_START=11001.
    """)
    return


@app.cell
def __(ARTIFACTS_DIR, mo, screen_df):
    import json

    if screen_df is not None:
        from crispr_al.splits import generate_splits, SCREEN_ID

        all_genes = screen_df["gene_symbol"].tolist()
        splits = generate_splits(all_genes, n_repeats=25, train_size=2000)

        from crispr_al.io import save_split_manifest, save_split_files
        import pandas as pd

        manifest_path = ARTIFACTS_DIR / "split_manifest.csv"
        save_split_manifest(splits, str(manifest_path))
        manifest_df = pd.read_csv(manifest_path)

        splits_dir = ARTIFACTS_DIR / "splits"
        save_split_files(splits, str(splits_dir))

        mo.md(f"""
        ✓ Generated **{len(splits)} splits**
        - Manifest saved: `{manifest_path.name}` ({len(manifest_df)} rows)
        - Split files: `artifacts/splits/` ({len(splits)} JSON files)
        - Example split hash: `{splits[0]['split_hash']}`
        - Train genes per split: {len(splits[0]['train_genes'])}
        - Test genes per split: {len(splits[0]['test_genes'])}
        """)
    else:
        splits = []
        mo.md("⚠ Cannot generate splits without screen data.")
    return SCREEN_ID, all_genes, json, manifest_df, manifest_path, manifest_records, splits, splits_dir


@app.cell
def __(mo):
    mo.md("""
    ## Section 4: Model Training

    ### Ridge Regression

    Ridge regression adds an L2 penalty to ordinary least squares:

    $$\\hat{\\beta} = \\arg\\min_\\beta \\|y - X\\beta\\|^2 + \\alpha \\|\\beta\\|^2$$

    **Why Ridge over LASSO?** With only p=9 features and n=2,000 training samples, LASSO's
    variable selection is unnecessary. Ridge keeps all features and avoids zeroing real
    signal when all 9 features are expected to contribute.

    We use **RidgeCV** with 5-fold cross-validation to select α from {0.1, 1.0, 10.0, 100.0}.

    ### Random Forest

    An ensemble of 200 decision trees. Each tree is trained on a bootstrap sample and uses
    a random subset of features at each split (max_features="sqrt" = 3 features).

    **Advantage over Ridge:** Captures non-linear feature interactions. Feature importances
    reveal which biological properties are most predictive.
    """)
    return


@app.cell
def __(ARTIFACTS_DIR, features_df, mo, np, screen_df, splits):
    if screen_df is not None and features_df is not None and len(splits) > 0:
        from crispr_al.models import train_ridge, train_rf, scale_features, predict

        # Demo on split r001
        s0 = splits[0]
        train_genes = s0["train_genes"]
        test_genes = s0["test_genes"]

        score_idx = screen_df.set_index("gene_symbol")["score_norm"]

        X_train = features_df.loc[train_genes].values
        y_train = score_idx.loc[train_genes].values
        X_test = features_df.loc[test_genes].values
        y_test = score_idx.loc[test_genes].values

        X_train_s, X_test_s = scale_features(X_train, X_test)

        ridge_model = train_ridge(X_train_s, y_train)
        mo.md(f"""
        ### Ridge (split r001 demo)
        - Selected α: **{ridge_model.alpha_:.2f}**
        - Coefficients: {np.round(ridge_model.coef_, 3).tolist()}

        **Feature names:** molm13_log_tpm, coessential_mean_r_top50, coessential_molm13_chronos,
        n_reactome_pathways, n_go_bp_terms, n_go_mf_terms, in_hallmark_apoptosis,
        in_hallmark_oxidative_phosphorylation, n_kegg_pathways
        """)
    return (
        X_test,
        X_test_s,
        X_train,
        X_train_s,
        predict,
        ridge_model,
        s0,
        scale_features,
        score_idx,
        test_genes,
        train_genes,
        train_rf,
        train_ridge,
        y_test,
        y_train,
    )


@app.cell
def __(mo):
    mo.md("""
    ## Section 5: Metrics Deep Dive

    ### Regression metrics
    - **Pearson r**: Linear correlation between predicted and actual scores
    - **Spearman ρ**: Rank correlation (robust to outliers)
    - **R²**: Fraction of variance explained

    ### Precision@K (Ranking)
    For sensitizers, rank all test genes by predicted score (ascending). Among the top K:

    $$\\text{Precision@K} = \\frac{|\\{\\text{top-K predicted}\\} \\cap \\{\\text{true sensitizers}\\}|}{K}$$

    ### AUROC (Classification)
    Area under ROC curve for separating sensitizers (CS < -1.0) from non-sensitizers.
    0.5 = random, 1.0 = perfect, 0.0 = perfectly inverted.
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ## Section 6: Run All 25 Splits

    Running Ridge and Random Forest on all 25 splits. Each split trains on 2,000 genes
    and predicts on ~17,109 holdout genes.
    """)
    return


@app.cell
def __(
    ARTIFACTS_DIR,
    SCHEMA_PATH,
    features_df,
    mo,
    np,
    pd,
    scale_features,
    screen_df,
    splits,
    train_rf,
    train_ridge,
):
    import datetime

    if screen_df is not None and features_df is not None and len(splits) > 0:
        from crispr_al.metrics import (
            compute_regression_metrics,
            compute_ranking_metrics,
            compute_classification_metrics,
            build_metrics_record,
            validate_metrics_record,
            flatten_metrics_row,
        )
        from crispr_al.models import predict as model_predict
        from crispr_al.io import save_metrics_json

        from crispr_al.io import get_code_commit
        code_commit = get_code_commit(cwd=str(ARTIFACTS_DIR.parent.parent.parent))

        score_idx2 = screen_df.set_index("gene_symbol")["score_norm"]
        hit_sens_idx = screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
        hit_res_idx = screen_df.set_index("gene_symbol")["is_hit_resistor"]

        metrics_dir = ARTIFACTS_DIR / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        progress = mo.status.progress_bar(range(len(splits)), title="Running splits")

        for i, split in enumerate(splits):
            train_genes_i = split["train_genes"]
            test_genes_i = split["test_genes"]

            X_train_i = features_df.loc[train_genes_i].values.astype(np.float64)
            y_train_i = score_idx2.loc[train_genes_i].values
            X_test_i = features_df.loc[test_genes_i].values.astype(np.float64)
            y_test_i = score_idx2.loc[test_genes_i].values
            hit_sens_i = hit_sens_idx.loc[test_genes_i].values
            hit_res_i = hit_res_idx.loc[test_genes_i].values

            assert len(set(train_genes_i) & set(test_genes_i)) == 0

            X_train_scaled, X_test_scaled = scale_features(X_train_i, X_test_i)

            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            data_counts = {
                "train_row_count": len(train_genes_i),
                "test_row_count": len(test_genes_i),
                "n_unique_train_genes": len(set(train_genes_i)),
                "n_unique_test_genes": len(set(test_genes_i)),
                "n_overlap_genes_train_test": 0,
            }
            leakage_checks = {
                "disjoint_gene_label_rows": True,
                # Refers to StandardScaler fitted on X_train only.
                # Screen-level z-scoring is pre-registered harmonization done
                # before split generation and is not a leakage risk.
                "normalization_fit_on_train_only": True,
                "split_hash_logged": True,
            }

            for model_name, model_obj in [
                ("ridge", train_ridge(X_train_scaled, y_train_i)),
                ("rf", train_rf(X_train_scaled, y_train_i, seed=split["seed"], n_estimators=200)),
            ]:
                y_pred_i = model_predict(model_obj, X_test_scaled)

                reg = compute_regression_metrics(y_test_i, y_pred_i)
                rank = compute_ranking_metrics(y_pred_i, hit_sens_i, hit_res_i)
                clf = compute_classification_metrics(y_pred_i, hit_sens_i, hit_res_i)

                record = build_metrics_record(
                    split=split,
                    data_counts=data_counts,
                    leakage_checks=leakage_checks,
                    regression=reg,
                    ranking=rank,
                    classification=clf,
                    run_id=f"{split['split_id']}_{model_name}",
                    timestamp_utc=timestamp,
                    code_commit=code_commit,
                )

                json_path = metrics_dir / f"{split['split_id']}_{model_name}.json"
                save_metrics_json(record, str(json_path))
                validate_metrics_record(record, str(SCHEMA_PATH))

                row = {"model": model_name, **flatten_metrics_row(split, reg, rank, clf)}
                all_results.append(row)

            next(progress)

        results_df = pd.DataFrame(all_results)
        mo.md(f"✓ Completed {len(splits)} splits × 2 models = **{len(splits)*2} metric JSON files**")
    else:
        results_df = None
        mo.md("⚠ Skipped: missing screen data or features.")
    return (
        X_test_scaled,
        X_train_scaled,
        all_results,
        clf,
        code_commit,
        compute_classification_metrics,
        compute_ranking_metrics,
        compute_regression_metrics,
        data_counts,
        datetime,
        hit_res_i,
        hit_res_idx,
        hit_sens_i,
        hit_sens_idx,
        i,
        json_path,
        leakage_checks,
        metrics_dir,
        model_name,
        model_obj,
        model_predict,
        progress,
        rank,
        record,
        reg,
        results_df,
        row,
        save_metrics_json,
        score_idx2,
        split,
        timestamp,
        train_genes_i,
        test_genes_i,
        validate_metrics_record,
        build_metrics_record,
    )


@app.cell
def __(mo):
    mo.md("""
    ## Section 7: Baseline Comparison

    ### Zero predictor
    Predicts 0.0 for all genes (the global mean of z-scored data). This guarantees:
    - Pearson r = 0.0
    - AUROC = 0.5
    - Precision@K = hit_prevalence (base rate)

    ### Chronos-only baseline
    Uses the raw MOLM-13 Chronos score (no training, no screen data). Tests whether
    venetoclax screen signal adds value beyond prior MOLM-13 essentiality.
    """)
    return


@app.cell
def __(
    ARTIFACTS_DIR,
    SCHEMA_PATH,
    code_commit,
    compute_classification_metrics,
    compute_ranking_metrics,
    compute_regression_metrics,
    datetime,
    features_df,
    mo,
    np,
    pd,
    save_metrics_json,
    screen_df,
    splits,
    validate_metrics_record,
    build_metrics_record,
):
    if screen_df is not None and features_df is not None and len(splits) > 0:
        score_idx3 = screen_df.set_index("gene_symbol")["score_norm"]
        hit_sens_idx3 = screen_df.set_index("gene_symbol")["is_hit_sensitizer"]
        hit_res_idx3 = screen_df.set_index("gene_symbol")["is_hit_resistor"]

        # Use the first split as representative for baselines
        test_genes_b = splits[0]["test_genes"]
        y_test_b = score_idx3.loc[test_genes_b].values
        hit_sens_b = hit_sens_idx3.loc[test_genes_b].values
        hit_res_b = hit_res_idx3.loc[test_genes_b].values

        baseline_results = []

        # Zero predictor
        y_zero = np.zeros(len(y_test_b))
        reg_zero = compute_regression_metrics(y_test_b, y_zero)
        rank_zero = compute_ranking_metrics(y_zero, hit_sens_b, hit_res_b)
        clf_zero = compute_classification_metrics(y_zero, hit_sens_b, hit_res_b)
        baseline_results.append({
            "baseline": "zero_predictor",
            "pearson": reg_zero["pearson"],
            "auroc_sensitizer": clf_zero["labels"][0]["auroc"],
            "precision_at_50": rank_zero["k_metrics"][0]["precision_at_k"],
        })

        # Chronos-only baseline
        chronos_series = features_df["coessential_molm13_chronos"]
        y_chronos = chronos_series.loc[test_genes_b].values
        # More negative Chronos = more essential = potentially more sensitizer
        reg_chronos = compute_regression_metrics(y_test_b, y_chronos)
        rank_chronos = compute_ranking_metrics(y_chronos, hit_sens_b, hit_res_b)
        clf_chronos = compute_classification_metrics(y_chronos, hit_sens_b, hit_res_b)
        baseline_results.append({
            "baseline": "chronos_only",
            "pearson": reg_chronos["pearson"],
            "auroc_sensitizer": clf_chronos["labels"][0]["auroc"],
            "precision_at_50": rank_chronos["k_metrics"][0]["precision_at_k"],
        })

        baseline_df = pd.DataFrame(baseline_results)

        # Save baseline metrics
        baseline_record = {
            "schema_version": "1.0.0",
            "run_id": "aim1_baselines",
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "code_commit": code_commit,
            "split": {
                "split_id": splits[0]["split_id"] + "_baseline",
                "generator_id": "aim1_random_gene_holdout",
                "family": "random_gene_holdout",
                "aim": "aim1_venetoclax",
                "metrics_profile": "aim1_transfer",
                "seed": splits[0]["seed"],
                "repeat_index": 1,
                "train_screen_id": "chen2019_1393",
                "test_screen_id": "chen2019_1393",
                "split_hash": splits[0]["split_hash"],
            },
            "data_counts": {
                "train_row_count": 2000,
                "test_row_count": len(test_genes_b),
                "n_unique_train_genes": 2000,
                "n_unique_test_genes": len(test_genes_b),
                "n_overlap_genes_train_test": 0,
            },
            "leakage_checks": {
                "disjoint_gene_label_rows": True,
                "normalization_fit_on_train_only": True,
                "split_hash_logged": True,
            },
            "metrics": {
                "regression": reg_zero,
                "ranking": {"k_metrics": rank_zero["k_metrics"]},
                "classification": clf_zero,
            },
            "notes": "Zero predictor baseline",
        }
        save_metrics_json(baseline_record, str(ARTIFACTS_DIR / "metrics" / "aim1_baseline_zero.json"))
        validate_metrics_record(baseline_record, str(SCHEMA_PATH))

        mo.md(f"""
        **Baseline Results (on split r001 test set):**

        {baseline_df.to_markdown(index=False)}

        Zero predictor Pearson r ≈ {reg_zero['pearson']:.4f} (should be ~0)
        Chronos-only Pearson r ≈ {reg_chronos['pearson']:.4f}
        """)
    else:
        baseline_df = None
        mo.md("⚠ Skipped baselines.")
    return (
        baseline_df,
        baseline_record,
        baseline_results,
        chronos_series,
        clf_chronos,
        clf_zero,
        hit_res_b,
        hit_res_idx3,
        hit_sens_b,
        hit_sens_idx3,
        rank_chronos,
        rank_zero,
        reg_chronos,
        reg_zero,
        score_idx3,
        test_genes_b,
        y_chronos,
        y_test_b,
        y_zero,
    )


@app.cell
def __(mo):
    mo.md("""
    ## Section 8: Aggregation & Confidence Intervals

    Bootstrap 1,000× across 25 per-split estimates to compute 95% split-stability
    intervals. These quantify variability across different 2,000-gene training
    samples drawn from the same screen, not population-level sampling uncertainty.
    """)
    return


@app.cell
def __(ARTIFACTS_DIR, RESULTS_DIR, bootstrap_ci_bca, mo, pd, results_df):
    if results_df is not None:
        summary_rows = []
        metric_cols = [
            "pearson", "spearman", "r2", "rmse", "mae",
            "auroc_sensitizer", "auroc_resistor",
            "auprc_sensitizer", "auprc_resistor",
            "precision_at_50", "precision_at_100", "precision_at_200", "precision_at_500",
            "recall_at_50", "recall_at_100", "recall_at_200", "recall_at_500",
            "precision_at_50_resistor", "precision_at_100_resistor",
            "precision_at_200_resistor", "precision_at_500_resistor",
            "recall_at_50_resistor", "recall_at_100_resistor",
            "recall_at_200_resistor", "recall_at_500_resistor",
        ]

        for model_name in ["ridge", "rf"]:
            model_df = results_df[results_df["model"] == model_name]
            row = {"model": model_name}
            for col in metric_cols:
                if col in model_df.columns:
                    mean, lo, hi = bootstrap_ci_bca(model_df[col].values)
                    row[f"{col}_mean"] = round(mean, 4)
                    row[f"{col}_ci_lo"] = round(lo, 4)
                    row[f"{col}_ci_hi"] = round(hi, 4)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # Save per-model CSVs
        for model_name in ["ridge", "rf"]:
            model_df = results_df[results_df["model"] == model_name].copy()
            model_df.to_csv(RESULTS_DIR / f"design_a_results_{model_name}.csv", index=False)

        results_df.to_csv(RESULTS_DIR / "design_a_results_all.csv", index=False)
        summary_df.to_csv(RESULTS_DIR / "design_a_summary.csv", index=False)

        display_cols = ["model", "pearson_mean", "pearson_ci_lo", "pearson_ci_hi",
                        "auroc_sensitizer_mean", "auroc_sensitizer_ci_lo", "auroc_sensitizer_ci_hi"]
        available_display = [c for c in display_cols if c in summary_df.columns]
        mo.md(f"""
        **Aggregated results across 25 splits:**

        {summary_df[available_display].to_markdown(index=False)}

        Full results saved to `results/design_a_results_*.csv`
        """)
    else:
        summary_df = None
        mo.md("⚠ No results to aggregate.")
    return (
        available_display,
        col,
        display_cols,
        metric_cols,
        model_df,
        model_name,
        row,
        summary_df,
        summary_rows,
    )


@app.cell
def __(mo):
    mo.md("""
    ## Section 9: Validation

    Validate all metric JSON files against the schema.
    """)
    return


@app.cell
def __(ARTIFACTS_DIR, SCHEMA_PATH, mo, validate_metrics_record):
    import glob

    metrics_files = list((ARTIFACTS_DIR / "metrics").glob("*.json"))
    n_pass = 0
    n_fail = 0
    fail_list = []

    for mf in metrics_files:
        try:
            import json as json_mod
            with open(mf) as f:
                rec = json_mod.load(f)
            validate_metrics_record(rec, str(SCHEMA_PATH))
            n_pass += 1
        except Exception as e:
            n_fail += 1
            fail_list.append(f"{mf.name}: {e}")

    status = "✓ PASS" if n_fail == 0 else "✗ FAIL"
    mo.md(f"""
    **Schema validation results:**
    - {status}: {n_pass} passed, {n_fail} failed
    - Total files: {len(metrics_files)}
    {"- Failures: " + str(fail_list[:3]) if fail_list else ""}
    """)
    return f, glob, json_mod, mf, n_fail, n_pass, metrics_files, rec, status, fail_list


if __name__ == "__main__":
    app.run()
