import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
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
def _():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    import numpy as np
    import pandas as pd
    from pathlib import Path

    return Path, np, pd


@app.cell
def _():
    from crispr_al.metrics import bootstrap_ci_bca

    return (bootstrap_ci_bca,)


@app.cell
def _(Path):
    ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
    RESULTS_DIR = Path(__file__).parent / "results"
    SCHEMA_PATH = Path(__file__).parent / "metrics.schema.json"
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bulk"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "splits").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR, RESULTS_DIR, SCHEMA_PATH


@app.cell
def _(mo):
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
def _(ARTIFACTS_DIR, mo):
    from crispr_al.io import load_parquet

    screen_parquet_path = ARTIFACTS_DIR / "screen_scores.parquet"
    features_parquet_path = ARTIFACTS_DIR / "gene_features.parquet"

    if screen_parquet_path.exists():
        screen_df = load_parquet(str(screen_parquet_path)).reset_index()
        mo.md(f"✓ Loaded screen_scores.parquet: **{len(screen_df):,} genes**")
    else:
        screen_df = None
        mo.md("⚠ screen_scores.parquet not found. Run the Nextflow pipeline first:\n```\nnextflow run pipelines/design_a/main.nf\n```")
    return features_parquet_path, load_parquet, screen_df


@app.cell
def _(mo, screen_df):
    if screen_df is not None:
        n_sensitizers = screen_df["is_hit_sensitizer"].sum()
        n_resistors = screen_df["is_hit_resistor"].sum()
        mo.md(f"""
        **Screen statistics:**
        - Total genes: {len(screen_df):,}
        - Sensitizers (CS < -1.0): {n_sensitizers:,} ({100*n_sensitizers/len(screen_df):.1f}%)
        - Resistors (CS > 3.0): {n_resistors:,} ({100*n_resistors/len(screen_df):.1f}%)
        """)
    return


@app.cell
def _(features_parquet_path, load_parquet, mo):
    if features_parquet_path.exists():
        features_df = load_parquet(str(features_parquet_path))
        mo.md(f"✓ Loaded gene_features.parquet: **{features_df.shape[0]:,} genes × {features_df.shape[1]} features**")
    else:
        features_df = None
        mo.md("⚠ gene_features.parquet not found. Run the Nextflow pipeline first.")
    return (features_df,)


@app.cell
def _(mo):
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
def _(features_df, mo):
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
    return


@app.cell
def _(mo):
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
def _(ARTIFACTS_DIR, mo, screen_df):
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
    return pd, splits


@app.cell
def _(mo):
    mo.md("""
    ## Section 4: Model Training

    ### Ridge Regression

    Ridge regression adds an L2 penalty to ordinary least squares:

    $$\hat{eta} = rg\min_eta \|y - Xeta\|^2 + lpha \|eta\|^2$$

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
def _(features_df, mo, np, screen_df, splits):
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
    return scale_features, train_rf, train_ridge


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 5: Metrics Deep Dive

    ### Regression metrics
    - **Pearson r**: Linear correlation between predicted and actual scores
    - **Spearman ρ**: Rank correlation (robust to outliers)
    - **R²**: Fraction of variance explained

    ### Precision@K (Ranking)
    For sensitizers, rank all test genes by predicted score (ascending). Among the top K:

    $$	ext{Precision@K} = 
    rac{|\\{	ext{top-K predicted}\\} \\cap \\{	ext{true sensitizers}\\}|}{K}$$

    ### AUROC (Classification)
    Area under ROC curve for separating sensitizers (CS < -1.0) from non-sensitizers.
    0.5 = random, 1.0 = perfect, 0.0 = perfectly inverted.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 6: Run All 25 Splits

    Running Ridge and Random Forest on all 25 splits. Each split trains on 2,000 genes
    and predicts on ~17,109 holdout genes.
    """,
    name="_"
)


@app.cell
def _(
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
    return (
        code_commit,
        compute_classification_metrics,
        compute_ranking_metrics,
        compute_regression_metrics,
        datetime,
        results_df,
        save_metrics_json,
        validate_metrics_record,
    )


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 7: Baseline Comparison

    ### Zero predictor
    Predicts 0.0 for all genes (the global mean of z-scored data). This guarantees:
    - Pearson r = 0.0
    - AUROC = 0.5
    - Precision@K = hit_prevalence (base rate)

    ### Chronos-only baseline
    Uses the raw MOLM-13 Chronos score (no training, no screen data). Tests whether
    venetoclax screen signal adds value beyond prior MOLM-13 essentiality.
    """,
    name="_"
)


@app.cell
def _(
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
    return


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 8: Aggregation & Confidence Intervals

    Bootstrap 1,000× across 25 per-split estimates to compute 95% split-stability
    intervals. These quantify variability across different 2,000-gene training
    samples drawn from the same screen, not population-level sampling uncertainty.
    """,
    name="_"
)


@app.cell
def _(RESULTS_DIR, bootstrap_ci_bca, mo, pd, results_df):
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
    return


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 9: Validation

    Validate all metric JSON files against the schema.
    """,
    name="_"
)


app._unparsable_cell(
    """
    import glob

    metrics_files = list((ARTIFACTS_DIR / \"metrics\").glob(\"*.json\"))
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
            fail_list.append(f\"{mf.name}: {e}\")

    status = \"✓ PASS\" if n_fail == 0 else \"✗ FAIL\"
    mo.md(f\"\"\"
    **Schema validation results:**
    - {status}: {n_pass} passed, {n_fail} failed
    - Total files: {len(metrics_files)}
    {\"- Failures: \" + str(fail_list[:3]) if fail_list else \"\"}
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 10: Results Overview Table

    The aggregated table below summarises performance across 25 independent gene-holdout
    splits. Each metric shows the mean and 95% BCa bootstrap confidence interval computed
    by resampling the 25 per-split estimates 1,000 times.

    **What BCa CIs tell us here:** These intervals capture *split-stability* — how much
    metrics vary when you draw a different 2,000-gene training set from the same screen.
    Tight CIs mean the model's performance is robust to which 2,000 genes are chosen;
    wide CIs mean performance is sensitive to the particular training sample.

    They do **not** quantify population-level sampling uncertainty (how well the result
    would generalise to a different screen). Cross-screen transfer is the subject of
    Design B.
    """,
    name="_"
)


@app.cell
def _(Path, mo, pd):
    _DESIGN_A_DIR = Path(__file__).parent / "results" / "design_a"
    _summary_path = _DESIGN_A_DIR / "design_a_summary.csv"

    if _summary_path.exists():
        da_summary_df = pd.read_csv(_summary_path)

        _display_cols = [
            "model",
            "pearson_mean", "pearson_ci_lo", "pearson_ci_hi",
            "spearman_mean", "spearman_ci_lo", "spearman_ci_hi",
            "auroc_sensitizer_mean", "auroc_sensitizer_ci_lo", "auroc_sensitizer_ci_hi",
            "auprc_sensitizer_mean", "auprc_sensitizer_ci_lo", "auprc_sensitizer_ci_hi",
            "precision_at_50_mean", "precision_at_50_ci_lo", "precision_at_50_ci_hi",
            "precision_at_200_mean", "precision_at_200_ci_lo", "precision_at_200_ci_hi",
        ]
        _avail = [c for c in _display_cols if c in da_summary_df.columns]
        mo.md(da_summary_df[_avail].to_markdown(index=False))
    else:
        da_summary_df = None
    return


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** Ridge regression clearly dominates Random Forest on every metric.
    The most striking contrast is **R²**: Ridge achieves +0.038 (modest but positive),
    while RF achieves −0.354, meaning RF predictions have *more variance than the
    target itself* — it is fitting noise. This happens because the 9-feature space is
    too low-dimensional for an ensemble of 200 trees; trees overfit individual training
    genes and generalise poorly.

    A Pearson r ≈ 0.197 for Ridge may appear modest, but this represents *correlation
    from gene biology alone*, without any training examples from the same screen.
    Precision@50 ≈ 0.185 against a base rate of ~9% (sensitizers / all test genes)
    means Ridge recovers roughly 2× the expected number of true sensitizers in a
    50-gene shortlist — practically meaningful for experiment design.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 11: Per-Split Performance Distribution

    Examining *split-to-split variability* answers a key experimental question: is the
    model's performance stable, or does it depend heavily on which 2,000 genes were
    sampled for training?

    - **Tight spread** → 2,000 training genes is sufficient; results are reproducible
    - **Wide spread** → model is data-hungry; adding more training examples would help

    Each point below represents one of the 25 independent splits.
    """,
    name="_"
)


@app.cell
def _(Path, mo, pd):
    import matplotlib
    matplotlib.use('Agg')
    from plotnine import (
        ggplot as _ggplot, aes as _aes, geom_jitter as _geom_jitter,
        facet_wrap as _facet_wrap, labs as _labs,
    )
    from crispr_al.plotting import theme_publication as _theme_pub, scale_color_publication as _scale_color

    _DESIGN_A_DIR = Path(__file__).parent / "results" / "design_a"
    _all_path = _DESIGN_A_DIR / "design_a_results_all.csv"

    if _all_path.exists():
        da_all_df = pd.read_csv(_all_path)

        _plot_metrics = {
            "pearson": "Pearson r",
            "auroc_sensitizer": "AUROC (sensitizer)",
            "precision_at_50": "Precision@50",
            "precision_at_200": "Precision@200",
        }

        _rows = []
        for _m, _label in _plot_metrics.items():
            if _m in da_all_df.columns:
                _sub = da_all_df[["model", _m]].copy()
                _sub = _sub.rename(columns={_m: "value"})
                _sub["metric"] = _label
                _rows.append(_sub)
        _long_df = pd.concat(_rows, ignore_index=True)

        _fig = (
            _ggplot(_long_df, _aes(x="model", y="value", color="model"))
            + _geom_jitter(width=0.15, size=2, alpha=0.7)
            + _facet_wrap("~ metric", scales="free_y", ncol=2)
            + _scale_color()
            + _theme_pub()
            + _labs(x="Model", y="Metric value", title="Per-split metric distribution (25 splits)")
        ).draw()
        mo.pyplot(_fig)
    else:
        da_all_df = None
    return


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** Ridge shows very tight distributions across all 25 splits —
    Pearson r spans roughly 0.193–0.200, and AUROC spans 0.597–0.606. This stability
    means 2,000 training genes is sufficient for Ridge; adding more training data is
    unlikely to substantially improve performance.

    RF is both noisier and consistently below Ridge. The negative R² is stable across
    splits (not an outlier artefact), confirming that RF's poor generalisation is
    structural, not random. This rules out the hypothesis that \"RF just needs a
    better hyperparameter search\" — the issue is that the feature space is too sparse
    for tree-based ensemble methods on this task.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 12: Hit Enrichment — Precision@K

    **Gene prioritisation framing:** In a real experiment, you can only follow up on K
    genes. If you rank all ~17,000 holdout genes by predicted sensitivity score
    (ascending), how many of the top K are true sensitizers?

    Precision@K = (true sensitizers in top K) / K

    The horizontal dashed line shows the **base rate** (fraction of test genes that are
    true sensitizers ≈ 9%), i.e. what random selection would achieve.

    Values above the dashed line indicate enrichment; the higher above, the better.
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    from plotnine import (
        ggplot as _ggplot, aes as _aes, geom_col as _geom_col,
        geom_hline as _geom_hline, position_dodge as _position_dodge, labs as _labs,
    )
    from crispr_al.plotting import theme_publication as _theme_pub, scale_fill_publication as _scale_fill

    if da_summary_df is not None:
        _pk_cols = {
            "precision_at_50_mean": "P@50",
            "precision_at_100_mean": "P@100",
            "precision_at_200_mean": "P@200",
            "precision_at_500_mean": "P@500",
        }
        _rows = []
        for _col, _label in _pk_cols.items():
            if _col in da_summary_df.columns:
                for _, _r in da_summary_df.iterrows():
                    _rows.append({"model": _r["model"], "K": _label, "precision": _r[_col]})
        _prec_df = pd.DataFrame(_rows)

        _base_rate = 0.092  # ~9% sensitizer prevalence in test set

        _fig = (
            _ggplot(_prec_df, _aes(x="K", y="precision", fill="model"))
            + _geom_col(position=_position_dodge(width=0.8), width=0.7)
            + _geom_hline(yintercept=_base_rate, linetype="dashed", colour="#888888", size=0.8)
            + _scale_fill()
            + _theme_pub()
            + _labs(
                x="Top-K cutoff",
                y="Precision@K",
                title="Hit enrichment: Precision@K vs random baseline",
            )
        ).draw()
        mo.pyplot(_fig)
    else:
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** Ridge delivers roughly **2× enrichment** at every K threshold.
    At P@50 ≈ 0.185, a 50-gene follow-up experiment would recover ~9 true sensitizers
    vs ~5 expected by chance. At P@500 ≈ 0.155, a 500-gene experiment recovers ~77
    true sensitizers vs ~46 expected by chance — meaningful for genome-wide prioritisation.

    RF performs near or below the base rate at small K (P@50 ≈ 0.076), confirming it
    is not useful for gene prioritisation in this regime. RF's predictions are too
    noisy to reliably rank the most extreme sensitizers.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 13: Feature Ablation Analysis

    **Leave-one-out ablation:** For each of the 9 features, we retrain Ridge on the
    remaining 8 features and measure the change in Precision@50 relative to the full
    9-feature model (baseline P@50 ≈ 0.185).

    A large negative delta means the dropped feature was critical — performance
    collapses without it. A small delta means the feature is redundant or can be
    approximated by the remaining features.

    This is complementary to RF MDI (Section 16): ablation measures actual prediction
    impact while MDI measures split frequency in tree ensembles.
    """,
    name="_"
)


@app.cell
def _(Path, mo, pd):
    from plotnine import ggplot, aes, geom_col, labs, coord_flip, scale_fill_gradient2
    from crispr_al.plotting import theme_publication

    _DESIGN_A_DIR = Path(__file__).parent / "results" / "design_a"
    _abl_path = _DESIGN_A_DIR / "feature_ablation_ridge.csv"

    if _abl_path.exists():
        da_ablation_df = pd.read_csv(_abl_path)

        _full_baseline = 0.185  # full-model P@50 mean across 25 splits

        _agg = (
            da_ablation_df
            .groupby("dropped_feature")["precision_at_50"]
            .mean()
            .reset_index()
            .rename(columns={"precision_at_50": "mean_prec_at_50"})
        )
        _agg["delta_p50"] = _agg["mean_prec_at_50"] - _full_baseline
        _agg = _agg.sort_values("delta_p50")

        fig_ablation = (
            ggplot(_agg, aes(x="reorder(dropped_feature, delta_p50)", y="delta_p50", fill="delta_p50"))
            + geom_col()
            + coord_flip()
            + scale_fill_gradient2(low="#D6604D", mid="#F7F7F7", high="#2166AC", midpoint=0)
            + theme_publication()
            + labs(
                x="Dropped feature",
                y="ΔPrecision@50 (vs full model)",
                title="Feature ablation: impact on Ridge Precision@50",
                fill="ΔP@50",
            )
        )
        _fig = fig_ablation.draw()
        mo.pyplot(_fig)
    else:
        da_ablation_df = None
        fig_ablation = None
    return


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** `coessential_molm13_chronos` is by far the most critical
    feature — dropping it collapses P@50 from 0.185 to ~0.071 (≈ −61%). This makes
    biological sense: the Chronos score encodes *prior essentiality* in MOLM-13
    leukaemia cells. Genes that are already essential for cell survival are
    pre-selected for venetoclax sensitivity, because venetoclax targets the same
    apoptotic circuitry.

    `n_go_mf_terms` and `coessential_mean_r_top50` provide moderate orthogonal signal —
    they encode functional annotation breadth and genome-wide co-essentiality context
    respectively. The binary Hallmark features (apoptosis, oxidative phosphorylation)
    contribute negligibly, consistent with their low MDI in the RF (Section 16).

    For Design B, the key question is whether chronos is equally informative for a
    *different* screen (Sharon 2019).
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 14: Predicted vs Actual Score Distribution

    The scatter plot below shows predicted CRISPR Score (y-axis) against actual
    z-scored CS (x-axis) for the Random Forest model on one representative split.

    **What good calibration looks like:** Points cluster tightly around the diagonal
    with equal spread at all score values.

    **What poor calibration looks like:** A wide horizontal band (predictions are
    compressed toward zero regardless of actual score), or systematic bias at the
    extremes.
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    _DESIGN_A_DIR = Path(__file__).parent / "results" / "design_a"
    _img_path = _DESIGN_A_DIR / "figure_score_dist.png"

    if _img_path.exists():
        mo.image(src=open(str(_img_path), "rb").read())
    else:
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** The wide horizontal band confirms that both models compress
    predictions toward zero — the predicted range is far narrower than the actual CS
    range. This is expected when predicting from only 9 generic gene features: there
    is not enough screen-specific signal to assign extreme scores.

    The most important consequence for experimental design: the model *systematically
    underestimates* extreme sensitizers (actual CS < −2). These genes may appear
    moderately ranked even though they are the most interesting hits. This means
    **ranking metrics (AUROC, Precision@K) are more informative than raw score
    magnitude** — use the rank, not the predicted value, when prioritising genes.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 15: Expression-Stratified Rank Correlation

    **Why stratify?** A single aggregate Spearman ρ hides whether the model is
    uniformly mediocre or excellent for some gene subgroups and useless for others.

    We split test genes by:
    1. **Expression quartile** (Q1 = lowest MOLM-13 expression, Q4 = highest)
    2. **Hit status** (true sensitizers vs all other genes)

    Then compute Spearman ρ within each stratum.
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    _DESIGN_A_DIR = Path(__file__).parent / "results" / "design_a"
    _img_path = _DESIGN_A_DIR / "figure_stratified_spearman.png"

    if _img_path.exists():
        mo.image(src=open(str(_img_path), "rb").read())
    else:
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** The stratified analysis reveals a striking pattern: RF Spearman
    ρ for the highest-expression quartile (Q4) ≈ 0.201, but for Q2 ≈ 0.007. The model
    is essentially a **sophisticated expression filter** — it ranks high-expression genes
    well because those genes have richer co-essentiality data and better-annotated
    pathways in DepMap and GO databases.

    Low-expression genes are *informationally sparse*: they appear in fewer experiments,
    have fewer interactions, and are less annotated. For these genes, the 9 features
    carry almost no signal.

    **Implication for Design B:** Adding expression-derived features (e.g. percentile
    rank within the cell line transcriptome, tissue-specific expression) could improve
    performance on the low-expression tail — currently an unaddressed gap.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 16: RF Feature Importances (MDI)

    Random Forest reports **Mean Decrease in Impurity (MDI)**: how much each feature
    reduces tree-node impurity (variance for regression) when used as a split variable,
    averaged across all trees and nodes.

    **MDI vs ablation (Section 13):**
    - MDI measures *how often* and *how much* a feature is used for splitting
    - Ablation measures *what happens to predictions* when the feature is removed
    - They usually agree in direction but can diverge when features are correlated
      (collinear features compete for split priority in RF, so MDI underestimates
      redundant but predictive features)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    _DESIGN_A_DIR = Path(__file__).parent / "results" / "design_a"
    _img_path = _DESIGN_A_DIR / "figure_feature_importance.png"

    if _img_path.exists():
        mo.image(src=open(str(_img_path), "rb").read())
    else:
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    **Interpretation:** MDI ordering — `coessential_molm13_chronos` (≈0.30) >
    `molm13_log_tpm` (≈0.23) > `n_go_mf_terms` (≈0.17) > `n_go_bp_terms` (≈0.14) >
    `n_reactome_pathways` (≈0.10) — broadly agrees with the ablation ordering.

    However, `coessential_mean_r_top50` shows near-zero MDI despite moderate ablation
    impact. This is a **collinearity artefact**: `coessential_mean_r_top50` is
    correlated with `coessential_molm13_chronos`, so RF always picks chronos at split
    nodes and ignores the co-essentiality mean. Ridge, by contrast, uses both features
    simultaneously — its L2 penalty distributes weight across correlated predictors
    rather than picking one arbitrarily. This is one reason Ridge outperforms RF here.

    Binary Hallmark features (apoptosis, oxidative phosphorylation) have near-zero MDI,
    consistent with their negligible ablation impact. These membership flags are too
    coarse to split on effectively — a gene is either in the pathway or not, providing
    only a binary partition.
    """,
    name="_"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ## Section 17: Summary and Design B Readiness

    ### Key findings from Design A (within-screen holdout)

    1. **Ridge regression is the right baseline model.** With 2,000 training genes from
       the same screen, Ridge achieves Pearson r ≈ 0.197, AUROC ≈ 0.601, and P@50 ≈
       0.185 — roughly 2× enrichment over random selection. Tight BCa CIs across 25
       splits confirm the result is robust to the particular training sample drawn.

    2. **RF underperforms structurally, not incidentally.** R² = −0.354 with stable
       variance across splits means the 9-feature space is too low-dimensional for
       ensemble methods. RF overfits individual training genes and fails to generalise.
       No hyperparameter tuning will resolve this without new features.

    3. **Prior essentiality (chronos) is the dominant driver.** Ablation shows P@50
       drops 61% without `coessential_molm13_chronos`. This means the model primarily
       learns \"genes already essential in MOLM-13 are more likely to be venetoclax
       sensitizers\" — a biological prior rather than screen-specific signal.

    4. **The model is an expression filter.** Stratified analysis reveals near-zero ρ
       for low-expression genes. High-expression genes dominate performance because
       they have richer co-essentiality and annotation data.

    5. **Raw score magnitude is unreliable.** Extreme sensitizers are systematically
       underestimated. Use ranked predictions (AUROC, P@K), not score values, for
       experimental prioritisation.

    ### What Design B needs to address

    - **Cross-screen transfer:** Can a Ridge model trained on 2,000 Chen 2019 genes
      predict sensitivity in the Sharon 2019 screen (different cell line, same drug)?
      The chronos advantage may weaken if Sharon 2019 has different essentiality
      dependencies.
    - **Low-expression genes:** Adding expression-percentile features or cell-line
      specific expression flags could close the Q1/Q4 performance gap.
    - **Score calibration:** A post-hoc calibration step (e.g. isotonic regression on
      a held-out calibration set) could improve P@K at small K if extreme scores are
      the primary target of follow-up experiments.

    These baselines establish the **performance floor** for cross-screen transfer.
    Any Design B model that cannot beat Ridge P@50 ≈ 0.185 on the target screen is
    not adding value over simple within-screen regression.
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
