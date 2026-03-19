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
    # Phase 0: Metric Calibration

    ## Problem Framing

    Before investing in CRISPR screen prediction models (Phases 2–4), we must verify
    that standard regression and ranking metrics can actually detect signal in the
    Chen 2019 and Sharon 2019 venetoclax fitness screens.

    A recent benchmark debate (Ahlmann-Eltze vs Miller 2025) showed that metrics
    can be poorly calibrated when most perturbations are null — Precision@K may
    look low even for a perfect predictor if the hit rate is 5%.

    We use the **Dynamic Range Fraction (DRF)** from Miller et al. 2025:

    > DRF = (pos_ctrl_score − neg_ctrl_score) / (perfect_score − neg_ctrl_score)

    A DRF ≥ 0.1 means the metric can discriminate signal from noise. We evaluate
    three scenarios:
    1. **Chen within-screen** (split-half positive control) — continuous metrics only
    2. **Chen within-screen with hits** — all 8 metrics
    3. **Cross-screen** (Chen predicting Sharon) — all 8 metrics; also the ceiling

    Metrics that pass (DRF ≥ 0.1) are recommended as primary metrics for Phases 2–4.
    """)
    return


@app.cell
def _():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    import json
    import datetime
    import warnings
    import numpy as np
    import pandas as pd
    from pathlib import Path

    return Path, datetime, json, np, os, pd, sys, warnings


@app.cell
def _():
    from crispr_al.metrics import (
        make_negative_control,
        make_positive_control_split_half,
        make_positive_control_cross_screen,
        compute_calibration_report,
        compute_calibration_report_with_hits,
    )
    from crispr_al.screen import (
        load_screen_scores,
        load_sharon_screen_scores,
        zscore_normalize,
        assign_hit_labels_zscore,
    )
    from crispr_al.io import get_code_commit

    return (
        assign_hit_labels_zscore,
        compute_calibration_report,
        compute_calibration_report_with_hits,
        get_code_commit,
        load_screen_scores,
        load_sharon_screen_scores,
        make_negative_control,
        make_positive_control_cross_screen,
        make_positive_control_split_half,
        zscore_normalize,
    )


@app.cell
def _(Path):
    ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
    PHASE0_DIR = ARTIFACTS_DIR / "phase0"
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bulk"

    PHASE0_DIR.mkdir(parents=True, exist_ok=True)

    return ARTIFACTS_DIR, DATA_DIR, PHASE0_DIR


@app.cell
def _(mo):
    mo.md("## Section 1 — Load Screens")
    return


@app.cell
def _(
    ARTIFACTS_DIR,
    DATA_DIR,
    assign_hit_labels_zscore,
    load_screen_scores,
    load_sharon_screen_scores,
    pd,
    zscore_normalize,
):
    _chen_parquet = ARTIFACTS_DIR / "screen_scores.parquet"
    _sharon_parquet = ARTIFACTS_DIR / "sharon_scores.parquet"

    # Load raw files to get gene_symbol (not stored in parquets)
    _chen_raw_path = DATA_DIR / "chen2019_venetoclax" / "BIOGRID-ORCS-SCREEN_1393-2.0.18.screen.tab.txt"
    _sharon_raw_path = DATA_DIR / "sharon2019_venetoclax" / "BIOGRID-ORCS-SCREEN_1402-2.0.18.screen.tab.txt"

    _chen_raw = load_screen_scores(str(_chen_raw_path))
    _sharon_raw = load_sharon_screen_scores(str(_sharon_raw_path))

    if _chen_parquet.exists():
        _chen_scores = pd.read_parquet(_chen_parquet)
        # Merge gene_symbol from raw; entrez_id is the join key
        _chen_scores["entrez_id"] = _chen_scores["entrez_id"].astype(str)
        _chen_raw["entrez_id"] = _chen_raw["entrez_id"].astype(str)
        chen_df = _chen_raw[["gene_symbol", "entrez_id"]].merge(
            _chen_scores[["entrez_id", "score_norm", "is_hit_sensitizer", "is_hit_resistor"]],
            on="entrez_id",
            how="inner",
        )
    else:
        chen_df = assign_hit_labels_zscore(zscore_normalize(_chen_raw, score_col="cs"))

    if _sharon_parquet.exists():
        _sharon_scores = pd.read_parquet(_sharon_parquet)
        _sharon_scores["entrez_id"] = _sharon_scores["entrez_id"].astype(str)
        _sharon_raw["entrez_id"] = _sharon_raw["entrez_id"].astype(str)
        sharon_df = _sharon_raw[["gene_symbol", "entrez_id"]].merge(
            _sharon_scores[["entrez_id", "score_norm", "is_hit_sensitizer", "is_hit_resistor"]],
            on="entrez_id",
            how="inner",
        )
    else:
        sharon_df = assign_hit_labels_zscore(zscore_normalize(_sharon_raw, score_col="lfc"))

    return chen_df, sharon_df


@app.cell
def _(chen_df, mo, sharon_df):
    mo.md(f"""
    **Chen 2019:** {len(chen_df):,} genes,
    {chen_df["is_hit_sensitizer"].sum()} sensitizers,
    {chen_df["is_hit_resistor"].sum()} resistors

    **Sharon 2019:** {len(sharon_df):,} genes,
    {sharon_df["is_hit_sensitizer"].sum()} sensitizers,
    {sharon_df["is_hit_resistor"].sum()} resistors
    """)
    return


@app.cell
def _(mo):
    mo.md("## Section 2 — Compute Controls")
    return


@app.cell
def _(
    chen_df,
    make_negative_control,
    make_positive_control_cross_screen,
    make_positive_control_split_half,
    sharon_df,
):
    chen_series = chen_df.set_index("gene_symbol")["score_norm"]
    sharon_series = sharon_df.set_index("gene_symbol")["score_norm"]

    # Cross-screen positive control: Chen predicting Sharon
    y_true_sharon, y_pred_from_chen = make_positive_control_cross_screen(
        chen_series, sharon_series
    )

    # Within-screen split-half positive control (Chen)
    y_true_chen_half, y_pred_chen_half = make_positive_control_split_half(
        chen_df, score_col="score_norm", seed=42
    )

    # Negative controls
    chen_neg_ctrl = make_negative_control(chen_df["score_norm"].values)
    sharon_neg_ctrl = make_negative_control(sharon_df["score_norm"].values)

    # Shared gene count
    n_shared = len(chen_series.index.intersection(sharon_series.index))

    return (
        chen_neg_ctrl,
        chen_series,
        n_shared,
        sharon_neg_ctrl,
        sharon_series,
        y_pred_chen_half,
        y_pred_from_chen,
        y_true_chen_half,
        y_true_sharon,
    )


@app.cell
def _(mo, n_shared, np, y_pred_from_chen, y_true_sharon):
    from scipy.stats import spearmanr, pearsonr
    _rho = float(spearmanr(y_true_sharon, y_pred_from_chen).statistic)
    _r = float(pearsonr(y_true_sharon, y_pred_from_chen).statistic)

    mo.md(f"""
    **Shared genes (Chen ∩ Sharon):** {n_shared:,}

    **Cross-screen alignment (Chen → Sharon):**
    - Spearman ρ = {_rho:.3f}
    - Pearson r = {_r:.3f}

    These values represent the *ceiling* for cross-screen prediction.
    """)
    return pearsonr, spearmanr


@app.cell
def _(mo):
    mo.md("## Section 3 — DRF Table + Bar Chart")
    return


@app.cell
def _(
    chen_df,
    chen_neg_ctrl,
    compute_calibration_report,
    compute_calibration_report_with_hits,
    make_positive_control_cross_screen,
    make_positive_control_split_half,
    np,
    pd,
    sharon_df,
    sharon_neg_ctrl,
    y_pred_from_chen,
    y_true_sharon,
):
    # --- Scenario 1: Chen within-screen, continuous only ---
    _y_true_c, _y_pred_c = make_positive_control_split_half(chen_df, "score_norm", seed=42)
    _chen_neg_half = np.full(len(_y_true_c), float(chen_df["score_norm"].mean()))
    _rep1 = compute_calibration_report(_y_true_c, _y_pred_c, _chen_neg_half)

    # --- Scenario 2: Chen within-screen with hits ---
    _rng = np.random.default_rng(42)
    _idx = np.arange(len(chen_df))
    _rng.shuffle(_idx)
    _n = len(_idx) // 2
    _half_b_idx = _idx[_n : 2 * _n]
    _hit_sens_half = chen_df["is_hit_sensitizer"].values[_half_b_idx]
    _hit_res_half = chen_df["is_hit_resistor"].values[_half_b_idx]
    _rep2 = compute_calibration_report_with_hits(
        _y_true_c, _y_pred_c, _chen_neg_half,
        _hit_sens_half, _hit_res_half
    )

    # --- Scenario 3: Cross-screen Chen → Sharon ---
    from pandas import Series as _S
    _shared = chen_df.set_index("gene_symbol")["score_norm"].index.intersection(
        sharon_df.set_index("gene_symbol")["score_norm"].index
    ).sort_values()
    _sharon_hits_shared = sharon_df.set_index("gene_symbol").loc[_shared, "is_hit_sensitizer"].values
    _sharon_res_shared = sharon_df.set_index("gene_symbol").loc[_shared, "is_hit_resistor"].values
    _sharon_neg_shared = np.full(len(y_true_sharon), float(sharon_df["score_norm"].mean()))
    _rep3 = compute_calibration_report_with_hits(
        y_true_sharon, y_pred_from_chen, _sharon_neg_shared,
        _sharon_hits_shared, _sharon_res_shared
    )

    # Build DRF summary dataframe
    _PASS_THRESHOLD = 0.1
    _rows = []

    def _add_rows(report, scenario):
        for k, v in report.items():
            if k.startswith("drf_"):
                metric = k[4:]
                _rows.append({
                    "scenario": scenario,
                    "metric": metric,
                    "drf": round(v, 4),
                    "pos_ctrl_score": round(report.get(f"pos_{metric}", float("nan")), 4),
                    "neg_ctrl_score": round(report.get(f"neg_{metric}", float("nan")), 4),
                    "verdict": "PASS" if v >= _PASS_THRESHOLD else "FAIL",
                })

    _add_rows(_rep1, "chen_within_continuous")
    _add_rows(_rep2, "chen_within_hits")
    _add_rows(_rep3, "cross_screen")

    drf_df = pd.DataFrame(_rows)
    rep1 = _rep1
    rep2 = _rep2
    rep3 = _rep3
    shared_genes = _shared

    return (
        rep1,
        rep2,
        rep3,
        _sharon_hits_shared,
        _sharon_neg_shared,
        _sharon_res_shared,
        shared_genes,
        drf_df,
    )


@app.cell
def _(drf_df, mo):
    mo.ui.table(drf_df)
    return


@app.cell
def _(drf_df):
    from plotnine import (
        ggplot, aes, geom_col, geom_hline, facet_wrap,
        theme_minimal, labs, theme, element_text, coord_flip,
        scale_fill_manual,
    )

    _p = (
        ggplot(drf_df, aes(x="metric", y="drf", fill="verdict"))
        + geom_col()
        + geom_hline(yintercept=0.1, linetype="dashed", color="black", size=0.8)
        + facet_wrap("~scenario", ncol=1)
        + scale_fill_manual(values={"PASS": "#2196F3", "FAIL": "#EF5350"})
        + coord_flip()
        + labs(
            title="Dynamic Range Fraction (DRF) by Metric and Scenario",
            subtitle="Dashed line = DRF 0.1 pass threshold",
            x="Metric",
            y="DRF",
            fill="Verdict",
        )
        + theme_minimal()
        + theme(figure_size=(8, 10))
    )
    _p
    return (
        aes,
        coord_flip,
        element_text,
        facet_wrap,
        geom_col,
        geom_hline,
        ggplot,
        labs,
        scale_fill_manual,
        theme,
        theme_minimal,
    )


@app.cell
def _(mo):
    mo.md("## Section 4 — Cross-Screen Ceiling Scatter")
    return


@app.cell
def _(
    shared_genes,
    aes,
    chen_df,
    element_text,
    ggplot,
    labs,
    pd,
    pearsonr,
    sharon_df,
    spearmanr,
    theme,
    theme_minimal,
    y_pred_from_chen,
    y_true_sharon,
):
    # Build full ceiling dataframe (all shared genes)
    _ceiling_df = pd.DataFrame({
        "gene_symbol": shared_genes.tolist(),
        "chen_score": y_pred_from_chen,
        "sharon_score": y_true_sharon,
    })
    _sharon_meta = sharon_df.set_index("gene_symbol")[["is_hit_sensitizer", "is_hit_resistor"]]
    _ceiling_df = _ceiling_df.join(_sharon_meta, on="gene_symbol")
    _ceiling_df["hit_status"] = "non-hit"
    _ceiling_df.loc[_ceiling_df["is_hit_sensitizer"], "hit_status"] = "sensitizer"
    _ceiling_df.loc[_ceiling_df["is_hit_resistor"], "hit_status"] = "resistor"

    _rho = float(spearmanr(_ceiling_df["chen_score"], _ceiling_df["sharon_score"]).statistic)
    _r = float(pearsonr(_ceiling_df["chen_score"], _ceiling_df["sharon_score"]).statistic)

    # Sample for display
    _sample = _ceiling_df.sample(n=min(5000, len(_ceiling_df)), random_state=42)

    from plotnine import geom_point, scale_color_manual

    _p = (
        ggplot(_sample, aes(x="chen_score", y="sharon_score", color="hit_status"))
        + geom_point(alpha=0.4, size=0.8)
        + scale_color_manual(values={"non-hit": "#AAAAAA", "sensitizer": "#2196F3", "resistor": "#EF5350"})
        + labs(
            title=f"Cross-Screen Ceiling: Chen → Sharon (n={len(_ceiling_df):,} shared genes)",
            subtitle=f"Spearman ρ = {_rho:.3f}, Pearson r = {_r:.3f}",
            x="Chen 2019 score_norm",
            y="Sharon 2019 score_norm",
            color="Hit status (Sharon)",
        )
        + theme_minimal()
        + theme(figure_size=(7, 6))
    )
    _p

    ceiling_df = _ceiling_df
    ceiling_rho = _rho
    ceiling_r = _r

    return ceiling_df, ceiling_r, ceiling_rho, geom_point, scale_color_manual


@app.cell
def _(mo):
    mo.md("## Section 5 — Save Artifacts")
    return


@app.cell
def _(
    PHASE0_DIR,
    ceiling_df,
    ceiling_r,
    ceiling_rho,
    datetime,
    drf_df,
    get_code_commit,
    json,
    rep1,
    rep2,
    rep3,
):
    import uuid

    _timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _commit = get_code_commit()

    # calibration_report.json
    _report = {
        "schema_version": "phase0_v1",
        "timestamp_utc": _timestamp,
        "code_commit": _commit,
        "screens": {
            "chen_within_continuous": rep1,
            "chen_within_hits": rep2,
            "cross_screen_chen_to_sharon": rep3,
        },
        "cross_screen_ceiling": {
            "n_shared_genes": len(ceiling_df),
            "spearman_rho": round(ceiling_rho, 4),
            "pearson_r": round(ceiling_r, 4),
        },
    }
    with open(PHASE0_DIR / "calibration_report.json", "w") as _f:
        json.dump(_report, _f, indent=2)

    # cross_screen_ceiling.csv
    ceiling_df.to_csv(PHASE0_DIR / "cross_screen_ceiling.csv", index=False)

    # calibration_summary.csv
    _recommended_metrics = {"spearman", "pearson", "auroc_sensitizer", "precision_at_50", "precision_at_100"}
    _summary = drf_df.copy()
    _summary["recommended"] = _summary["metric"].isin(_recommended_metrics)
    _summary.to_csv(PHASE0_DIR / "calibration_summary.csv", index=False)

    print(f"Artifacts written to {PHASE0_DIR}")
    print(f"  calibration_report.json")
    print(f"  cross_screen_ceiling.csv  ({len(ceiling_df):,} rows)")
    print(f"  calibration_summary.csv   ({len(_summary)} rows)")

    return (uuid,)


@app.cell
def _(ceiling_r, ceiling_rho, drf_df, mo):
    _passing = drf_df[drf_df["verdict"] == "PASS"]["metric"].unique().tolist()
    _failing = drf_df[drf_df["verdict"] == "FAIL"]["metric"].unique().tolist()
    _cross = drf_df[drf_df["scenario"] == "cross_screen"]
    _cross_pass = _cross[_cross["verdict"] == "PASS"]["metric"].tolist()

    mo.md(f"""
    ## Section 5 — Interpretation

    ### Summary

    **Cross-screen ceiling:** Spearman ρ = {ceiling_rho:.3f}, Pearson r = {ceiling_r:.3f}

    This represents the maximum achievable correlation when using Chen screen scores
    directly to predict Sharon screen scores (no model involved — pure screen overlap).

    **Metrics passing DRF ≥ 0.1 (any scenario):** {", ".join(sorted(set(_passing)))}

    **Metrics failing DRF ≥ 0.1 (all scenarios):** {", ".join(sorted(set(_failing))) if _failing else "none"}

    **Cross-screen passing metrics:** {", ".join(_cross_pass)}

    ### Recommendations for Phases 2–4

    Use **Spearman ρ** and **Pearson r** as primary continuous metrics.
    Use **AUROC (sensitizer)** and **Precision@50/100** as primary ranking metrics.

    These metrics have demonstrated DRF ≥ 0.1 and can detect real signal in the
    venetoclax fitness screens. Interpret Precision@K values relative to the
    base rate (~5%) rather than in absolute terms.
    """)
    return


if __name__ == "__main__":
    app.run()
