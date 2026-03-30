"""Generate publication-quality figures for all three Olivieri 2020 benchmark aims.

Reads parquet results from results/olivieri2020/ and saves figures as 300 dpi
PNG and SVG to results/olivieri2020/figures/.

Figures produced:
  aim1_within_screen.{png,svg}  — violin of Pearson/AUROC per model across screens
  aim2_cross_library.{png,svg}  — AUROC by drug/direction for Ridge vs RF
  aim3_lodo.{png,svg}           — AUROC per screen, grouped by library and model
  summary_all_aims.{png,svg}    — AUROC comparison across all three aims
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_jitter,
    geom_point,
    geom_violin,
    ggplot,
    labs,
    position_dodge,
    position_jitter,
    theme,
)

from crispr_al.io import load_parquet
from crispr_al.plotting import theme_publication

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _save(fig, path_stem: Path) -> None:
    for ext in (".png", ".svg"):
        fig.save(str(path_stem.with_suffix(ext)), dpi=300)
    logger.info("Saved %s.{png,svg}", path_stem)


def plot_aim1(results: pd.DataFrame, fig_dir: Path) -> None:
    """Violin plots of Pearson and AUROC distributions across screens."""
    long = results.melt(
        id_vars=["screen", "model"],
        value_vars=["pearson", "auroc"],
        var_name="metric",
        value_name="value",
    )
    fig = (
        ggplot(long, aes("model", "value", fill="model"))
        + geom_violin(alpha=0.7)
        + geom_jitter(position=position_jitter(width=0.1), size=0.5, alpha=0.3)
        + facet_wrap("~metric", scales="free_y")
        + labs(
            title="Aim 1: Within-screen holdout (25 repeats × 30 screens)",
            x="Model", y="Metric value",
        )
        + theme_publication()
        + theme(legend_position="none")
    )
    _save(fig, fig_dir / "aim1_within_screen")


def plot_aim2(results: pd.DataFrame, fig_dir: Path) -> None:
    """AUROC by drug and direction (train→test) for Ridge vs RF."""
    results = results.copy()
    results["direction"] = results["train_screen"] + " → " + results["test_screen"]
    fig = (
        ggplot(results, aes("model", "auroc", fill="model"))
        + geom_violin(alpha=0.7)
        + geom_jitter(position=position_jitter(width=0.1), size=1.5)
        + facet_wrap("~drug")
        + labs(
            title="Aim 2: Cross-library transfer",
            x="Model", y="AUROC",
        )
        + theme_publication()
        + theme(legend_position="none")
    )
    _save(fig, fig_dir / "aim2_cross_library")


def plot_aim3(results: pd.DataFrame, fig_dir: Path) -> None:
    """AUROC per held-out screen, grouped by library and model."""
    fig = (
        ggplot(results, aes("model", "auroc", fill="model"))
        + geom_violin(alpha=0.7)
        + geom_jitter(position=position_jitter(width=0.1), size=1.5)
        + facet_wrap("~library")
        + labs(
            title="Aim 3: Leave-one-drug-out",
            x="Model", y="AUROC",
        )
        + theme_publication()
        + theme(legend_position="none")
    )
    _save(fig, fig_dir / "aim3_lodo")


def plot_summary(aim1: pd.DataFrame, aim2: pd.DataFrame, aim3: pd.DataFrame, fig_dir: Path) -> None:
    """Median AUROC comparison across all three aims."""
    rows = []
    for aim_label, df in [("Aim 1\n(within-screen)", aim1),
                          ("Aim 2\n(cross-library)", aim2),
                          ("Aim 3\n(LODO)", aim3)]:
        for model in ("Ridge", "RF"):
            sub = df[df["model"] == model]
            rows.append({
                "aim": aim_label,
                "model": model,
                "auroc": sub["auroc"].median(),
            })
    summary = pd.DataFrame(rows)

    fig = (
        ggplot(summary, aes("aim", "auroc", color="model"))
        + geom_point(position=position_dodge(width=0.4), size=4)
        + labs(title="Summary: Median AUROC across all aims", x="Aim", y="Median AUROC")
        + theme_publication()
        + theme(axis_text_x=element_text(size=8))
    )
    _save(fig, fig_dir / "summary_all_aims")


def main(results_dir: str) -> None:
    out = Path(results_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    aim1 = load_parquet(str(out / "aim1_within_screen_results.parquet"))
    aim2 = load_parquet(str(out / "aim2_cross_library_results.parquet"))
    aim3 = load_parquet(str(out / "aim3_lodo_results.parquet"))

    plot_aim1(aim1, fig_dir)
    plot_aim2(aim2, fig_dir)
    plot_aim3(aim3, fig_dir)
    plot_summary(aim1, aim2, aim3, fig_dir)
    logger.info("All figures saved to %s", fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="results/olivieri2020",
        help="Directory containing aim result parquet files (default: results/olivieri2020)",
    )
    args = parser.parse_args()
    main(args.results_dir)
