"""Generate publication figures for the Elling 2024 CRISPR-StAR analysis.

Reads results from notebooks/crispr_star/results/ and saves 300 dpi PNG figures to
notebooks/crispr_star/figures/crispr_star_analysis/:
  - agreement_scatter.png     : in-vitro vs in-vivo NormZ, coloured by concordance label
  - cat_curve.png             : CAT(N) curve for sensitisers and resistors
  - discordance_scatter.png   : volcano-style Δ (in-vivo − in-vitro) vs in-vitro score
  - predictive_model_auroc.png: ROC curves for RF, LogReg, and naive baseline

Usage:
    source scripts/activate_env.sh
    python scripts/elling2024/06_plot_results.py
    python scripts/elling2024/06_plot_results.py --results-dir notebooks/crispr_star/results
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_abline, geom_hline, geom_vline,
    labs, scale_color_manual, scale_x_continuous, scale_y_continuous,
    theme, element_text, facet_wrap,
)

from crispr_al.plotting import theme_publication, PUBLICATION_COLORS
from crispr_al.io import load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("notebooks/crispr_star/results")
DEFAULT_DATA_DIR = Path("data/elling2024")
FIGURES_DIR = Path("notebooks/crispr_star/figures/crispr_star_analysis")
NORMZ_THRESHOLD = 2.0

CONCORDANCE_COLORS = {
    "concordant": PUBLICATION_COLORS[0],         # blue
    "in_vivo_specific": PUBLICATION_COLORS[1],   # red
    "in_vitro_specific": PUBLICATION_COLORS[3],  # purple
}


def plot_agreement_scatter(discordant_df: pd.DataFrame, scores_long: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: in-vitro NormZ vs in-vivo NormZ, coloured by concordance label."""
    wide = scores_long.pivot_table(
        index="gene_symbol", columns="context", values="score_norm", aggfunc="mean"
    )
    wide.columns.name = None
    wide = wide.dropna(subset=["in_vitro", "in_vivo"])

    plot_df = wide.join(discordant_df[["concordance_label"]], how="inner").reset_index()

    p = (
        ggplot(plot_df, aes(x="in_vitro", y="in_vivo", color="concordance_label"))
        + geom_point(alpha=0.4, size=0.8)
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="grey", size=0.5)
        + geom_hline(yintercept=0, linetype="dotted", color="grey", size=0.3)
        + geom_vline(xintercept=0, linetype="dotted", color="grey", size=0.3)
        + scale_color_manual(
            values=CONCORDANCE_COLORS,
            name="Concordance",
        )
        + labs(
            x="In vitro NormZ",
            y="In vivo NormZ",
            title="In vitro vs in vivo screen agreement (Elling 2024)",
        )
        + theme_publication()
    )
    dest = out_dir / "agreement_scatter.png"
    p.save(str(dest), dpi=300, width=6, height=5)
    log.info("Saved %s", dest)


def plot_cat_curve(agreement_df: pd.DataFrame, out_dir: Path) -> None:
    """CAT(N) curves for sensitisers and resistors."""
    # agreement_df stores cat curve in a nested dict column; flatten it
    cat_cols_sens = [c for c in agreement_df.columns if c.startswith("cat_") and c.endswith("_sens")]
    cat_cols_res = [c for c in agreement_df.columns if c.startswith("cat_") and c.endswith("_res")]

    rows = []
    for col in cat_cols_sens:
        n = int(col.replace("cat_", "").replace("_sens", ""))
        rows.append({"n": n, "cat": float(agreement_df[col].iloc[0]), "direction": "Sensitiser"})
    for col in cat_cols_res:
        n = int(col.replace("cat_", "").replace("_res", ""))
        rows.append({"n": n, "cat": float(agreement_df[col].iloc[0]), "direction": "Resistor"})

    if not rows:
        log.warning("No CAT columns found in agreement_df; skipping cat_curve plot.")
        return

    cat_df = pd.DataFrame(rows)

    p = (
        ggplot(cat_df, aes(x="n", y="cat", color="direction", group="direction"))
        + geom_line(size=1.0)
        + geom_point(size=2.5)
        + scale_color_manual(
            values={"Sensitiser": PUBLICATION_COLORS[1], "Resistor": PUBLICATION_COLORS[0]},
            name="Direction",
        )
        + labs(
            x="N (top genes)",
            y="CAT(N)",
            title="Concordance at the top — Elling 2024",
        )
        + theme_publication()
    )
    dest = out_dir / "cat_curve.png"
    p.save(str(dest), dpi=300, width=5, height=4)
    log.info("Saved %s", dest)


def plot_discordance_scatter(discordant_df: pd.DataFrame, scores_long: pd.DataFrame, out_dir: Path) -> None:
    """Volcano-style: Δ (in-vivo − in-vitro) vs in-vitro score."""
    wide = scores_long.pivot_table(
        index="gene_symbol", columns="context", values="score_norm", aggfunc="mean"
    )
    wide.columns.name = None
    wide = wide.dropna(subset=["in_vitro", "in_vivo"])

    plot_df = wide.join(discordant_df[["concordance_label"]], how="inner").reset_index()
    plot_df["delta"] = plot_df["in_vivo"] - plot_df["in_vitro"]

    p = (
        ggplot(plot_df, aes(x="in_vitro", y="delta", color="concordance_label"))
        + geom_point(alpha=0.4, size=0.8)
        + geom_hline(yintercept=0, linetype="dashed", color="grey", size=0.5)
        + scale_color_manual(values=CONCORDANCE_COLORS, name="Concordance")
        + labs(
            x="In vitro NormZ",
            y="Δ (in vivo − in vitro)",
            title="Discordance scatter — Elling 2024",
        )
        + theme_publication()
    )
    dest = out_dir / "discordance_scatter.png"
    p.save(str(dest), dpi=300, width=6, height=4)
    log.info("Saved %s", dest)


def plot_roc_curves(model_results: pd.DataFrame, out_dir: Path) -> None:
    """ROC-style bar chart (mean AUROC ± SD per model + baseline) across CV folds."""
    # Aggregate per-fold AUROC by model
    agg = model_results.groupby("model")[["auroc", "baseline_auroc"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["model", "auroc_mean", "auroc_sd", "baseline_auroc_mean", "baseline_auroc_sd"]

    # Build long form for a bar chart
    rows = []
    for _, r in agg.iterrows():
        rows.append({"label": r["model"].replace("_", " ").title(), "auroc_mean": r["auroc_mean"], "auroc_sd": r["auroc_sd"]})
    baseline_auroc = agg["baseline_auroc_mean"].mean()
    rows.append({"label": "Naive baseline", "auroc_mean": baseline_auroc, "auroc_sd": agg["baseline_auroc_sd"].mean()})

    plot_df = pd.DataFrame(rows)

    from plotnine import geom_bar, geom_errorbar, position_dodge
    p = (
        ggplot(plot_df, aes(x="label", y="auroc_mean", fill="label"))
        + geom_bar(stat="identity", width=0.6)
        + geom_errorbar(aes(ymin="auroc_mean - auroc_sd", ymax="auroc_mean + auroc_sd"), width=0.2)
        + geom_hline(yintercept=0.5, linetype="dashed", color="grey", size=0.5)
        + scale_y_continuous(limits=(0, 1.05))
        + labs(
            x="Model",
            y="Mean AUROC (5-fold CV)",
            title="Predictive model: in-vitro → in-vivo validation",
        )
        + theme_publication()
        + theme(legend_position="none")
    )
    dest = out_dir / "predictive_model_auroc.png"
    p.save(str(dest), dpi=300, width=5, height=4)
    log.info("Saved %s", dest)


def main(results_dir: Path, data_dir: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    agreement_df = load_parquet(str(results_dir / "agreement_metrics.parquet"))
    discordant_df = load_parquet(str(results_dir / "discordant_genes.parquet"))
    model_results = load_parquet(str(results_dir / "predictive_model_results.parquet"))
    scores_long = load_parquet(str(data_dir / "scores_long.parquet"))

    plot_agreement_scatter(discordant_df, scores_long, FIGURES_DIR)
    plot_cat_curve(agreement_df, FIGURES_DIR)
    plot_discordance_scatter(discordant_df, scores_long, FIGURES_DIR)
    plot_roc_curves(model_results, FIGURES_DIR)
    log.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    main(Path(args.results_dir), Path(args.data_dir))
