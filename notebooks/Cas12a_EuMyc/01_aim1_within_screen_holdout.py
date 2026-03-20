"""Aim 1 — Within-Screen Holdout Results Notebook.

Loads completed metric JSONs from results/aim1/ and visualises prediction
quality across arms, strategies, and models.
"""
import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT      = Path(__file__).resolve().parents[2]
    RESULTS   = ROOT / "results" / "aim1"
    return ROOT, RESULTS, json, mo, np, pd, plt


@app.cell
def _(RESULTS, json, pd):
    """Load all result JSON files into a flat DataFrame."""
    records = []
    for json_path in sorted(RESULTS.rglob("*.json")):
        if "_predictions" in json_path.stem:
            continue
        try:
            with open(json_path) as f:
                rec = json.load(f)
        except Exception:
            continue

        arm      = json_path.parent.name
        stem     = json_path.stem         # e.g. 001_random_ridge
        parts    = stem.split("_")
        repeat   = int(parts[0])
        strategy = "_".join(parts[1:-1])
        model    = parts[-1]

        sp   = rec.get("split", {})
        metr = rec.get("metrics", {})
        reg  = metr.get("regression", {})
        clf  = metr.get("classification", {})
        rank = metr.get("ranking", {})

        auroc_sens = None
        auprc_sens = None
        for lab in clf.get("labels", []):
            if lab["label"] == "sensitizer":
                auroc_sens = lab["auroc"]
                auprc_sens = lab["auprc"]

        p50 = None
        for km in rank.get("k_metrics", []):
            if km["k"] == 50:
                p50 = km["precision_at_k"]

        records.append({
            "arm":          arm,
            "strategy":     strategy,
            "model":        model,
            "repeat":       repeat,
            "seed":         sp.get("seed"),
            "pearson":      reg.get("pearson"),
            "spearman":     reg.get("spearman"),
            "r2":           reg.get("r2"),
            "rmse":         reg.get("rmse"),
            "auroc_sens":   auroc_sens,
            "auprc_sens":   auprc_sens,
            "p50":          p50,
        })

    results_df = pd.DataFrame(records)
    print(f"Loaded {len(results_df)} records from {RESULTS}")
    results_df.head()
    return results_df,


@app.cell
def _(mo, results_df):
    """Summary table: mean metrics by arm × strategy × model."""
    if results_df.empty:
        mo.md("No results found yet. Run `run_aim1.py` first.")
    else:
        summary = (
            results_df
            .groupby(["arm", "strategy", "model"])[["pearson", "auroc_sens", "p50"]]
            .agg(["mean", "std"])
            .round(3)
        )
        summary
    return summary,


@app.cell
def _(plt, results_df):
    """Box plot: Pearson r by strategy × model (all arms combined)."""
    if results_df.empty:
        return

    arms      = sorted(results_df["arm"].unique())
    strategies = sorted(results_df["strategy"].unique())
    models    = sorted(results_df["model"].unique())

    fig, axes = plt.subplots(1, len(arms), figsize=(4 * len(arms), 5), sharey=True)
    if len(arms) == 1:
        axes = [axes]

    colors = {"ridge": "#4878CF", "rf": "#D65F5F"}

    for ax, arm in zip(axes, arms):
        arm_df = results_df[results_df["arm"] == arm]
        positions = []
        labels    = []
        data_list = []
        col_list  = []
        tick_pos  = []
        tick_label = []

        pos = 0
        for strat in strategies:
            strat_df = arm_df[arm_df["strategy"] == strat]
            group_positions = []
            for model in models:
                vals = strat_df[strat_df["model"] == model]["pearson"].dropna().values
                if len(vals) == 0:
                    continue
                data_list.append(vals)
                col_list.append(colors.get(model, "gray"))
                positions.append(pos)
                group_positions.append(pos)
                pos += 1
            if group_positions:
                tick_pos.append(np.mean(group_positions))
                tick_label.append(strat.replace("_", "\n"))
            pos += 0.5

        if data_list:
            bps = ax.boxplot(data_list, positions=positions, patch_artist=True,
                             widths=0.6, showfliers=False)
            for patch, color in zip(bps["boxes"], col_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_label, fontsize=7)
        ax.set_title(arm, fontsize=9)
        ax.set_ylabel("Pearson r" if arm == arms[0] else "")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], alpha=0.7, label=m) for m in models]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.suptitle("Pearson r by Strategy × Model (Aim 1 Within-Screen Holdout)", fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig,


@app.cell
def _(plt, results_df):
    """AUROC (sensitizer) by strategy × model."""
    if results_df.empty:
        return

    arms       = sorted(results_df["arm"].unique())
    strategies = sorted(results_df["strategy"].unique())
    models     = sorted(results_df["model"].unique())

    fig2, axes2 = plt.subplots(1, len(arms), figsize=(4 * len(arms), 5), sharey=True)
    if len(arms) == 1:
        axes2 = [axes2]

    colors = {"ridge": "#4878CF", "rf": "#D65F5F"}

    for ax, arm in zip(axes2, arms):
        arm_df    = results_df[results_df["arm"] == arm]
        data_list = []
        col_list  = []
        positions = []
        tick_pos  = []
        tick_label = []

        pos = 0
        for strat in strategies:
            strat_df = arm_df[arm_df["strategy"] == strat]
            group_positions = []
            for model in models:
                vals = strat_df[strat_df["model"] == model]["auroc_sens"].dropna().values
                if len(vals) == 0:
                    continue
                data_list.append(vals)
                col_list.append(colors.get(model, "gray"))
                positions.append(pos)
                group_positions.append(pos)
                pos += 1
            if group_positions:
                import numpy as _np
                tick_pos.append(_np.mean(group_positions))
                tick_label.append(strat.replace("_", "\n"))
            pos += 0.5

        if data_list:
            bps = ax.boxplot(data_list, positions=positions, patch_artist=True,
                             widths=0.6, showfliers=False)
            for patch, color in zip(bps["boxes"], col_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_label, fontsize=7)
        ax.set_title(arm, fontsize=9)
        ax.set_ylabel("AUROC (sensitizer)" if arm == arms[0] else "")
        ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--", label="random")
        ax.grid(axis="y", alpha=0.3)

    fig2.suptitle("AUROC (sensitizer) by Strategy × Model (Aim 1 Within-Screen Holdout)", fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig2,


if __name__ == "__main__":
    app.run()
