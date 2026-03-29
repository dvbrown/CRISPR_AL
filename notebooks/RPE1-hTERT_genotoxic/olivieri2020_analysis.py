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
    # Olivieri 2020 — Genotoxic CRISPR Screen Benchmark

    **Reference**: Olivieri M et al., *Cell* 2020, PMID 32649862

    This notebook summarises the three benchmark aims evaluated on the Olivieri 2020
    genotoxic screens from RPE1-hTERT p53−/− Cas9 cells.

    | Aim | Question | Design |
    |-----|----------|--------|
    | 1 | Can pathway features predict within-screen NormZ? | 80/20 random holdout, 25 repeats × 30 screens |
    | 2 | Does the signal transfer across library versions? | Cross-library (TKOv2 ↔ TKOv3) for Cisplatin and Camptothecin |
    | 3 | Can we learn drug–gene associations across drugs? | Leave-one-drug-out within TKOv2 and TKOv3 |

    **Features**: 6 pathway-only (no expression / co-essentiality — RPE1-hTERT not in DepMap/CCLE)
    **Models**: Ridge regression (RidgeCV) and Random Forest
    **Score**: DrugZ NormZ; hit threshold NormZ < −3.0
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
    from plotnine import (
        aes, element_text, facet_wrap, geom_hline, geom_jitter, geom_point,
        geom_violin, ggplot, labs, position_dodge, position_jitter, theme,
    )
    from crispr_al.plotting import theme_publication, scale_color_publication, scale_fill_publication

    return Path, aes, element_text, facet_wrap, geom_hline, geom_jitter, geom_point, geom_violin, ggplot, labs, np, pd, position_dodge, position_jitter, theme, theme_publication, scale_color_publication, scale_fill_publication


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).parent
    RESULTS_DIR = NOTEBOOK_DIR / "results"
    FIGURES_DIR = NOTEBOOK_DIR / "figures"
    DATA_DIR = NOTEBOOK_DIR.parent.parent / "data" / "olivieri2020"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, FIGURES_DIR, NOTEBOOK_DIR, RESULTS_DIR


@app.cell
def _(mo):
    mo.md("## Screen Overview")
    return


@app.cell
def _(DATA_DIR, mo, pd):
    meta = pd.read_parquet(DATA_DIR / "screen_metadata.parquet")
    n_tkov2 = (meta["library"] == "TKOv2").sum()
    n_tkov3 = (meta["library"] == "TKOv3").sum()
    mo.md(f"""
    **{len(meta)} screens** across {meta['drug'].nunique()} genotoxic agents
    (screen 1328 ICRF-187 excluded — QC fail)

    - TKOv2: {n_tkov2} screens
    - TKOv3: {n_tkov3} screens
    """)
    return (meta,)


@app.cell
def _(meta, mo):
    mo.ui.table(meta.sort_values("screen_id"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Aim 1 — Within-Screen Holdout

    **Question**: Can 6 pathway features predict which genes sensitise cells to a genotoxic
    agent within the same screen?

    **Design**: Random 80/20 gene split, 25 repeats per screen (1500 rows total).

    **Expected result**: Near-chance performance (Pearson ≈ 0.03–0.06, AUROC ≈ 0.59).
    Pathway features alone carry little signal when predicting within a single drug screen —
    the main value is as a baseline for Aims 2 and 3.

    #### Findings 1: Within-Screen Holdout

    ---

    > **Findings:** Both models perform near-chance within a single screen, confirming that
    > pathway features do not trivially reconstruct screen-specific gene rankings.
    > - **Ridge > RF on AUROC**: Unexpectedly, Ridge (median ≈ 0.64–0.67) outperforms RF
    >   (median ≈ 0.58) within-screen. RF likely overfits to the limited 80% training split,
    >   picking up screen-specific noise rather than transferable pathway signal.
    > - **Pearson near zero for both models**: Both RF and Ridge cluster around 0.02–0.03,
    >   confirming that continuous NormZ rank prediction is essentially uninformative within
    >   a single screen from pathway features alone.
    > - **High variance across screens**: The wide violin shapes (especially for AUROC) reflect
    >   genuine heterogeneity across 30 genotoxic screens — some screens are more pathway-structured
    >   than others.
    > - **Conclusion**: Aim 1 establishes the within-screen baseline; any improvement in Aims 2–3
    >   reflects genuine cross-condition generalisation rather than memorisation.
    """)
    return


@app.cell
def _(RESULTS_DIR, mo, pd):
    aim1 = pd.read_parquet(RESULTS_DIR / "aim1_within_screen_results.parquet")
    mo.md(f"Loaded Aim 1 results: **{len(aim1):,} rows** ({aim1['screen'].nunique()} screens × {aim1['repeat'].nunique()} repeats × {aim1['model'].nunique()} models)")
    return (aim1,)


@app.cell
def _(aim1, pd, mo):
    aim1_summary = (
        aim1.groupby("model")[["pearson", "auroc", "precision_at_50"]]
        .agg(["median", "mean"])
        .round(3)
    )
    mo.ui.table(aim1_summary.reset_index())
    return (aim1_summary,)


@app.cell
def _(aim1, aes, element_text, facet_wrap, geom_jitter, geom_violin, ggplot, labs, position_jitter, theme, theme_publication, scale_fill_publication, FIGURES_DIR):
    long1 = aim1.melt(
        id_vars=["screen", "model"],
        value_vars=["pearson", "auroc"],
        var_name="metric",
        value_name="value",
    )
    fig1 = (
        ggplot(long1, aes("model", "value", fill="model"))
        + geom_violin(alpha=0.7)
        + geom_jitter(position=position_jitter(width=0.1), size=0.5, alpha=0.3)
        + facet_wrap("~metric", scales="free_y")
        + scale_fill_publication()
        + labs(
            title="Aim 1: Within-screen holdout (25 repeats × 30 screens)",
            x="Model", y="Metric value",
        )
        + theme_publication()
        + theme(legend_position="none", axis_text_x=element_text(size=9))
    )
    fig1.save(str(FIGURES_DIR / "aim1_within_screen_nb.png"), dpi=300)
    fig1
    return (fig1, long1)


@app.cell
def _(mo):
    mo.md(r"""
    ## Aim 2 — Cross-Library Transfer

    **Question**: Does the gene-level signal transfer when the same drug is run in a
    different library (TKOv2 vs TKOv3)?

    **Design**: Train on one library version, predict on the other (6 directed pairs).

    **Expected result**: RF substantially better than Ridge (AUROC ≈ 0.74–0.76 for Cisplatin,
    ≈ 0.74 for Camptothecin). This suggests the pathway features capture a real signal that
    is preserved across library versions.

    #### Findings 2: Cross-Library Transfer

    ---

    > **Findings:** RF consistently outperforms Ridge for both drugs, with AUROC values
    > well above the within-screen baseline (~0.82 vs ~0.67), confirming that pathway
    > membership captures a real, library-independent signal.
    > - **Observed AUROC exceeds expectation**: RF achieves ~0.81 for Camptothecin and
    >   ~0.82–0.84 for Cisplatin — higher than the expected 0.74–0.76. Ridge sits at ~0.67
    >   for both drugs, a modest lift from its Aim 1 baseline.
    > - **Tight RF distributions**: Only 3 data points per panel (6 directed pairs, 2 drugs),
    >   but RF shows low variance — the cross-library signal is consistent regardless of
    >   transfer direction (TKOv2→TKOv3 or TKOv3→TKOv2).
    > - **Drug similarity**: Cisplatin and Camptothecin both induce DNA damage, which may
    >   explain the strong transferability; pathway features for DDR and replication stress
    >   genes are likely the main drivers.
    > - **Conclusion**: The jump from Aim 1 (RF AUROC ≈ 0.58) to Aim 2 (RF ≈ 0.82) is the
    >   core result — pathway features encode transferable signal that is masked by
    >   within-screen noise.
    """)
    return


@app.cell
def _(RESULTS_DIR, mo, pd):
    aim2 = pd.read_parquet(RESULTS_DIR / "aim2_cross_library_results.parquet")
    mo.md(f"Loaded Aim 2 results: **{len(aim2)} rows** ({aim2['drug'].nunique()} drugs × {aim2['model'].nunique()} models, {len(aim2)//aim2['model'].nunique()} directions each)")
    return (aim2,)


@app.cell
def _(aim2, pd, mo):
    aim2_summary = (
        aim2.groupby(["drug", "model"])[["auroc", "pearson"]]
        .median()
        .round(3)
        .reset_index()
    )
    mo.ui.table(aim2_summary)
    return (aim2_summary,)


@app.cell
def _(aim2, aes, element_text, facet_wrap, geom_jitter, geom_violin, ggplot, labs, position_jitter, theme, theme_publication, scale_fill_publication, FIGURES_DIR):
    fig2 = (
        ggplot(aim2, aes("model", "auroc", fill="model"))
        + geom_violin(alpha=0.7)
        + geom_jitter(position=position_jitter(width=0.1), size=2)
        + facet_wrap("~drug")
        + scale_fill_publication()
        + labs(
            title="Aim 2: Cross-library transfer",
            x="Model", y="AUROC",
        )
        + theme_publication()
        + theme(legend_position="none", axis_text_x=element_text(size=9))
    )
    fig2.save(str(FIGURES_DIR / "aim2_cross_library_nb.png"), dpi=300)
    fig2
    return (fig2,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Aim 3 — Leave-One-Drug-Out (LODO)

    **Question**: Can we generalise across drugs within the same library? Train on all
    other screens in a library, predict on the held-out drug screen.

    **Design**: Per-screen Z-normalisation before stacking; predictions back-transformed
    to test screen NormZ scale. 30 held-out screens × 2 models = 60 rows.

    **Expected result**: RF strong (TKOv2 Pearson ≈ 0.34, AUROC ≈ 0.84; TKOv3 Pearson ≈ 0.26,
    AUROC ≈ 0.83), Ridge near-chance. The pathway features generalise across genotoxic agents
    within a library, with RF capturing non-linear interactions.

    #### Findings 3: Leave-One-Drug-Out (LODO)

    ---

    > **Findings:** RF robustly generalises across genotoxic agents (median AUROC ≈ 0.82 in
    > both TKOv2 and TKOv3), while Ridge shows only modest lift above Aim 1 and high variance,
    > confirming that non-linear pathway interactions drive cross-drug generalisation.
    > - **RF consistent across libraries**: TKOv2 and TKOv3 RF distributions are nearly
    >   identical (both ≈ 0.82), suggesting the signal is not library-specific.
    > - **Ridge variance is high**: Ridge distributions are wide with some screens scoring as
    >   low as 0.55–0.58, indicating that linear pathway features are insufficient for some
    >   held-out drug-screen combinations.
    > - **AUROC matches or exceeds Aim 2**: The LODO AUROC for RF is comparable to cross-library
    >   transfer, which is surprising — generalising across chemically diverse drugs (Cisplatin,
    >   Camptothecin, ICRF-193, etc.) appears as feasible as generalising across library versions
    >   of the same drug.
    > - **Practical implication**: A single RF model trained on all available drug screens can
    >   rank candidate sensitiser genes for a novel genotoxic agent with ~0.82 AUROC — a useful
    >   prior for target selection even without expression or co-essentiality features.
    """)
    return


@app.cell
def _(RESULTS_DIR, mo, pd):
    aim3 = pd.read_parquet(RESULTS_DIR / "aim3_lodo_results.parquet")
    mo.md(f"Loaded Aim 3 results: **{len(aim3)} rows** ({aim3['screen'].nunique()} screens × {aim3['model'].nunique()} models)")
    return (aim3,)


@app.cell
def _(aim3, pd, mo):
    aim3_summary = (
        aim3.groupby(["library", "model"])[["pearson", "auroc"]]
        .median()
        .round(3)
        .reset_index()
    )
    mo.ui.table(aim3_summary)
    return (aim3_summary,)


@app.cell
def _(aim3, aes, element_text, facet_wrap, geom_jitter, geom_violin, ggplot, labs, position_jitter, theme, theme_publication, scale_fill_publication, FIGURES_DIR):
    fig3 = (
        ggplot(aim3, aes("model", "auroc", fill="model"))
        + geom_violin(alpha=0.7)
        + geom_jitter(position=position_jitter(width=0.1), size=2)
        + facet_wrap("~library")
        + scale_fill_publication()
        + labs(
            title="Aim 3: Leave-one-drug-out",
            x="Model", y="AUROC",
        )
        + theme_publication()
        + theme(legend_position="none", axis_text_x=element_text(size=9))
    )
    fig3.save(str(FIGURES_DIR / "aim3_lodo_nb.png"), dpi=300)
    fig3
    return (fig3,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary — Median AUROC Across All Aims

    The three aims capture different transfer settings:
    - **Aim 1** (within-screen): near-chance baseline — features don't trivially memorise screen scores
    - **Aim 2** (cross-library): moderate signal transfer for RF — pathway membership generalises across library versions
    - **Aim 3** (LODO): strong RF performance — pathway features robustly generalise across genotoxic agents

    #### Findings 4: Summary — Median AUROC Across All Aims

    ---

    > **Findings:** The summary plot reveals a striking model × transfer-setting interaction:
    > **RF** jumps from near-chance in Aim 1 (AUROC ≈ 0.58) to strong performance in Aims 2
    > and 3 (≈ 0.82), while **Ridge** shows only modest improvement (0.64 → 0.68).
    >
    > | Setting | RF AUROC | Ridge AUROC | Interpretation |
    > |---------|----------|-------------|----------------|
    > | Aim 1 — within-screen | ≈ 0.58 | ≈ 0.64 | Both near-chance; Ridge slightly better |
    > | Aim 2 — cross-library | ≈ 0.82 | ≈ 0.68 | RF captures transferable pathway signal |
    > | Aim 3 — LODO | ≈ 0.82 | ≈ 0.68 | RF generalises across drugs; Ridge plateau |
    >
    > - **RF inversion**: RF is *worse* than Ridge within-screen but *substantially better* across
    >   settings, suggesting RF overfits to within-screen noise while encoding richer cross-condition
    >   pathway interactions in its learned structure.
    > - **Ridge plateau**: Ridge's AUROC barely moves from Aim 1 to Aims 2–3, suggesting that
    >   linear pathway combinations provide a weak but stable signal regardless of transfer setting.
    > - **Aims 2 and 3 are equivalent**: RF AUROC is identical for cross-library and cross-drug
    >   transfer — the pathway features are as informative for drug diversity as for library version
    >   diversity, implying the signal is driven by shared DDR/replication biology, not drug-specific effects.
    > - **Practical guidance**: Use **RF** for any transfer prediction task (cross-library or cross-drug).
    >   Use **Ridge** only as a sanity-check baseline; its near-flat profile confirms it is not
    >   capturing the non-linear pathway interactions that drive genotoxic sensitivity.
    """)
    return


@app.cell
def _(aim1, aim2, aim3, aes, element_text, geom_point, ggplot, labs, pd, position_dodge, theme, theme_publication, scale_color_publication, FIGURES_DIR):
    rows = []
    for _aim_label, _df in [("Aim 1\n(within-screen)", aim1),
                             ("Aim 2\n(cross-library)", aim2),
                             ("Aim 3\n(LODO)", aim3)]:
        for _model in ("Ridge", "RF"):
            _sub = _df[_df["model"] == _model]
            rows.append({"aim": _aim_label, "model": _model, "auroc": _sub["auroc"].median()})
    summary = pd.DataFrame(rows)

    fig_summary = (
        ggplot(summary, aes("aim", "auroc", color="model"))
        + geom_point(position=position_dodge(width=0.4), size=4)
        + scale_color_publication()
        + labs(title="Summary: Median AUROC across all aims", x="Aim", y="Median AUROC")
        + theme_publication()
        + theme(axis_text_x=element_text(size=9))
    )
    fig_summary.save(str(FIGURES_DIR / "summary_all_aims_nb.png"), dpi=300)
    fig_summary
    return (fig_summary, rows, summary)


if __name__ == "__main__":
    app.run()
