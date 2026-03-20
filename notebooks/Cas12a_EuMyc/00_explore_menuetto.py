"""Exploratory data analysis of GSE285778_EuMycCountMenuetto.txt.gz.

Visualises the raw crRNA count matrix structure and key quality metrics:
  - Data dimensions and sample decoding
  - Library size per sample (total counts)
  - Guides-per-gene distribution
  - Count distributions per sample (log scale)
  - Replicate correlation within conditions
  - PCA of samples
  - Python-computed gene-level LFC per condition
  - LFC distributions and top hits

Run:
  marimo edit --watch notebooks/Cas12a_EuMyc/00_explore_menuetto.py
"""
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
    # GSE285778 — Menuetto Count Data: Exploratory Analysis

    **Library:** Menuetto (dual crRNA, ~21,743 genes, 2 crRNA/gene)
    **Cell line:** Eµ-MYC lymphoma line #20
    **Conditions:** Input (×6), DMSO (×6), Nutlin-3a (×6), S63845 (×6)

    Sample naming: `D{rep}{cond}` — D=Dual, rep=1–6,
    i=Input · d=DMSO · n=Nutlin-3a · s=S63845
    """)
    return


# ── Imports + paths ───────────────────────────────────────────────────────────

@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    RAW_FILE = Path.cwd() / "data/bulk/menuetto_scherzo_2025/raw/GSE285778_EuMycCountMenuetto.txt.gz"

    return Path, PCA, RAW_FILE, StandardScaler, np, pd, plt, stats


# ── Load count matrix ─────────────────────────────────────────────────────────

@app.cell
def _(RAW_FILE, pd):
    counts_raw = pd.read_csv(RAW_FILE, sep="\t", index_col=0, compression="gzip")
    gene_col   = counts_raw["Gene"]
    sample_cols = [c for c in counts_raw.columns if c != "Gene"]
    count_mat  = counts_raw[sample_cols].astype(float)

    print(f"Shape:        {count_mat.shape}  (guides × samples)")
    print(f"Samples:      {sample_cols}")
    print(f"Unique genes: {gene_col.nunique()}")
    print(f"Guides/gene:  {count_mat.shape[0] / gene_col.nunique():.2f} mean")

    return count_mat, gene_col, sample_cols


# ── Decode sample metadata ────────────────────────────────────────────────────

@app.cell
def _(pd, sample_cols):
    _COND_MAP = {"i": "Input", "d": "DMSO", "n": "Nutlin-3a", "s": "S63845"}
    COND_ORDER = ["Input", "DMSO", "Nutlin-3a", "S63845"]
    COND_COLOURS = {
        "Input":     "#888888",
        "DMSO":      "#4dac26",
        "Nutlin-3a": "#d01c8b",
        "S63845":    "#0571b0",
    }

    sample_meta = pd.DataFrame([
        {"sample": s, "rep": int(s[1]), "condition": _COND_MAP[s[2]]}
        for s in sample_cols
    ]).set_index("sample")

    print(sample_meta.to_string())
    return COND_COLOURS, COND_ORDER, sample_meta


# ── 1. Library size per sample ────────────────────────────────────────────────

@app.cell
def _(COND_COLOURS, COND_ORDER, count_mat, plt, sample_meta):
    from matplotlib.patches import Patch as _Patch

    _lib_sizes = count_mat.sum(axis=0) / 1e6
    _colours = [COND_COLOURS[sample_meta.loc[s, "condition"]] for s in _lib_sizes.index]

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.bar(range(len(_lib_sizes)), _lib_sizes.values, color=_colours,
            edgecolor="white", linewidth=0.5)
    _ax.set_xticks(range(len(_lib_sizes)))
    _ax.set_xticklabels(_lib_sizes.index, rotation=45, ha="right", fontsize=9)
    _ax.set_ylabel("Total counts (millions)")
    _ax.set_title("Library size per sample")
    _ax.axhline(_lib_sizes.mean(), color="black", linestyle="--", linewidth=1,
                label=f"mean = {_lib_sizes.mean():.1f}M")
    _ax.legend(
        handles=[_Patch(color=COND_COLOURS[c], label=c) for c in COND_ORDER],
        loc="upper right", fontsize=8,
    )
    plt.tight_layout()
    _fig


# ── 2. Guides-per-gene + count distribution ───────────────────────────────────

@app.cell
def _(count_mat, gene_col, np, plt):
    _guides_per_gene = gene_col.value_counts()
    _input_cols = [c for c in count_mat.columns if c.endswith("i")]
    _mean_input = count_mat[_input_cols].mean(axis=1)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4))

    _a1.hist(_guides_per_gene.values, bins=range(1, 10),
             color="#5b9bd5", edgecolor="white", align="left")
    _a1.set_xlabel("Guides per gene")
    _a1.set_ylabel("Number of genes")
    _a1.set_title("Guide count per gene")
    _a1.set_xticks(range(1, 9))

    _a2.hist(np.log10(_mean_input.clip(1)), bins=50,
             color="#5b9bd5", edgecolor="white")
    _a2.axvline(np.log10(_mean_input.clip(1)).median(),
                color="red", linestyle="--", label="median")
    _a2.set_xlabel("log10(mean Input count per guide)")
    _a2.set_ylabel("Guides")
    _a2.set_title("Input count distribution (mean over reps)")
    _a2.legend(fontsize=9)

    plt.tight_layout()
    _fig


# ── 3. Count distributions per sample (box plot, log scale) ──────────────────

@app.cell
def _(COND_COLOURS, COND_ORDER, count_mat, np, plt, sample_meta):
    _ordered = []
    for _cond in COND_ORDER:
        _ordered += sample_meta[sample_meta["condition"] == _cond].index.tolist()

    _fig, _ax = plt.subplots(figsize=(12, 5))
    _data = [np.log10(count_mat[s].clip(1)) for s in _ordered]
    _bp = _ax.boxplot(_data, patch_artist=True, showfliers=False,
                      medianprops=dict(color="white", linewidth=1.5))
    for _patch, _s in zip(_bp["boxes"], _ordered):
        _patch.set_facecolor(COND_COLOURS[sample_meta.loc[_s, "condition"]])
        _patch.set_alpha(0.8)

    _ax.set_xticks(range(1, len(_ordered) + 1))
    _ax.set_xticklabels(_ordered, rotation=45, ha="right", fontsize=9)
    _ax.set_ylabel("log10(count + 1)")
    _ax.set_title("Count distribution per sample")

    _n = 0
    for _cond in COND_ORDER:
        _nc = (sample_meta["condition"] == _cond).sum()
        _ax.axvline(_n + _nc + 0.5, color="grey", linewidth=0.8, linestyle=":")
        _ax.text(_n + _nc / 2 + 0.5, _ax.get_ylim()[1] * 0.97,
                 _cond, ha="center", fontsize=8, color=COND_COLOURS[_cond])
        _n += _nc

    plt.tight_layout()
    _fig


# ── 4. Replicate correlation ──────────────────────────────────────────────────

@app.cell
def _(count_mat, np, plt, sample_meta, stats):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))

    for _ax, _cond in zip(_axes, ["Input", "Nutlin-3a", "S63845"]):
        _samps = sample_meta[sample_meta["condition"] == _cond].index.tolist()
        _x = np.log10(count_mat[_samps[0]].clip(1))
        _y = np.log10(count_mat[_samps[1]].clip(1))
        _r, _ = stats.pearsonr(_x, _y)
        _ax.hexbin(_x, _y, gridsize=60, cmap="Blues", mincnt=1, linewidths=0.2)
        _ax.set_xlabel(f"log10 {_samps[0]}")
        _ax.set_ylabel(f"log10 {_samps[1]}")
        _ax.set_title(f"{_cond}: rep1 vs rep2\nPearson r = {_r:.4f}")

    plt.tight_layout()
    _fig


# ── 5. PCA of samples ─────────────────────────────────────────────────────────

@app.cell
def _(COND_COLOURS, COND_ORDER, PCA, StandardScaler, count_mat, np, plt, sample_meta):
    _cpm = count_mat.div(count_mat.sum(axis=0) / 1e6, axis=1)
    _log_cpm = np.log2(_cpm + 1)
    _X = StandardScaler().fit_transform(_log_cpm.T)

    _pca = PCA(n_components=4, random_state=42)
    _pcs = _pca.fit_transform(_X)
    _ev  = _pca.explained_variance_ratio_ * 100
    _sample_list = list(count_mat.columns)

    _fig, (_pa, _pb) = plt.subplots(1, 2, figsize=(11, 5))

    for _ax_pca, (_xi, _yi) in zip([_pa, _pb], [(0, 1), (2, 3)]):
        for _cond in COND_ORDER:
            _grp = sample_meta[sample_meta["condition"] == _cond]
            _idx = [_sample_list.index(s) for s in _grp.index]
            _ax_pca.scatter(_pcs[_idx, _xi], _pcs[_idx, _yi],
                            color=COND_COLOURS[_cond], label=_cond,
                            s=60, edgecolors="white", linewidths=0.5, zorder=3)
            for _s, _i in zip(_grp.index, _idx):
                _ax_pca.annotate(_s, (_pcs[_i, _xi], _pcs[_i, _yi]),
                                 fontsize=6, ha="center", va="bottom", color="grey")
        _ax_pca.set_xlabel(f"PC{_xi+1} ({_ev[_xi]:.1f}%)")
        _ax_pca.set_ylabel(f"PC{_yi+1} ({_ev[_yi]:.1f}%)")
        _ax_pca.axhline(0, color="lightgrey", linewidth=0.5)
        _ax_pca.axvline(0, color="lightgrey", linewidth=0.5)
        if _xi == 0:
            _ax_pca.legend(fontsize=8)

    _pa.set_title("PCA of samples (log-CPM, guide level)")
    plt.tight_layout()
    _fig


# ── 6. Compute gene-level LFC ─────────────────────────────────────────────────

@app.cell
def _(count_mat, gene_col, np, pd, sample_meta):
    _rpm = count_mat.div(count_mat.sum(axis=0) / 1e6, axis=1) + 1

    def _gene_lfc(treat_cond, ctrl_cond):
        _tc = sample_meta[sample_meta["condition"] == treat_cond].index.tolist()
        _cc = sample_meta[sample_meta["condition"] == ctrl_cond].index.tolist()
        _lfc_g = np.log2(_rpm[_tc].mean(axis=1) / _rpm[_cc].mean(axis=1))
        return _lfc_g.groupby(gene_col).median()

    lfc_df = pd.DataFrame({
        "lfc_nutlin": _gene_lfc("Nutlin-3a", "Input"),
        "lfc_s63845": _gene_lfc("S63845",    "Input"),
        "lfc_dmso":   _gene_lfc("DMSO",      "Input"),
    })

    print(f"Gene-level LFC for {len(lfc_df)} genes")
    print(lfc_df.describe().round(3).to_string())
    return (lfc_df,)


# ── 7. LFC distributions ──────────────────────────────────────────────────────

@app.cell
def _(lfc_df, np, plt):
    _COLS = {
        "lfc_nutlin": ("#d01c8b", "Nutlin-3a vs Input"),
        "lfc_s63845": ("#0571b0", "S63845 vs Input"),
        "lfc_dmso":   ("#4dac26", "DMSO vs Input"),
    }
    _bins = np.linspace(-6, 6, 100)

    _fig, _ax = plt.subplots(figsize=(9, 5))
    for _col, (_colour, _label) in _COLS.items():
        _ax.hist(lfc_df[_col].clip(-6, 6), bins=_bins, alpha=0.55,
                 color=_colour, label=_label, density=True)
    _ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    _ax.set_xlabel("Gene-level LFC (median of guides, RPM+1)")
    _ax.set_ylabel("Density")
    _ax.set_title("LFC distributions — Menuetto Eµ-MYC")
    _ax.legend(fontsize=9)
    plt.tight_layout()
    _fig


# ── 8. Ranked LFC with known controls ────────────────────────────────────────

@app.cell
def _(lfc_df, plt):
    # Nutlin-3a (MDM2 inhibitor / p53 activator):
    #   Trp53 KO → resistance (loss of p53 target) → enriched (positive LFC)
    #   Mdm2 KO  → more p53 activity → sensitised → depleted
    # S63845 (MCL-1 inhibitor):
    #   Bax/Bak1 KO → can't execute apoptosis → resistance → enriched
    #   Mcl1 KO   → synthetic lethality with MCL-1 inhibitor → depleted
    _CONTROLS = {
        "lfc_nutlin": {
            "enriched (resist.)": (["Trp53", "Cdkn1a", "Rb1"],   "#d01c8b"),
            "depleted (sensitise)": (["Mdm2", "Mdm4"],            "#0571b0"),
        },
        "lfc_s63845": {
            "enriched (resist.)":  (["Bax", "Bak1", "Bbc3"],     "#d01c8b"),
            "depleted (sensitise)": (["Mcl1", "Bcl2l1", "Bcl2"], "#0571b0"),
        },
    }

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 5))

    for _ax_r, _col in zip([_a1, _a2], ["lfc_nutlin", "lfc_s63845"]):
        _ranked = lfc_df[_col].sort_values()
        _ax_r.scatter(range(len(_ranked)), _ranked.values,
                      s=1, alpha=0.25, color="grey", rasterized=True)

        for _direction, (_genes, _colour) in _CONTROLS[_col].items():
            for _g in _genes:
                if _g in _ranked.index:
                    _r = _ranked.index.get_loc(_g)
                    _v = float(_ranked[_g])
                    _ax_r.scatter(_r, _v, s=30, color=_colour, zorder=5)
                    _ax_r.annotate(_g, (_r, _v), fontsize=7, color=_colour,
                                   xytext=(5, 3), textcoords="offset points")

        _cond_label = _col.replace("lfc_", "").replace("_", "-").title()
        _ax_r.set_xlabel("Gene rank")
        _ax_r.set_ylabel("LFC (vs Input)")
        _ax_r.set_title(f"Ranked LFC — {_cond_label}")
        _ax_r.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    _fig


# ── 9. Top hits table ─────────────────────────────────────────────────────────

@app.cell
def _(lfc_df, mo, pd):
    def _top(series, n=15):
        _dep = series.nsmallest(n).reset_index()
        _dep.columns = ["gene", "lfc"]
        _dep["direction"] = "depleted"
        _enr = series.nlargest(n).reset_index()
        _enr.columns = ["gene", "lfc"]
        _enr["direction"] = "enriched"
        return pd.concat([_enr, _dep]).reset_index(drop=True)

    return mo.vstack([
        mo.md("### Top 15 depleted / enriched — Nutlin-3a"),
        mo.plain_text(_top(lfc_df["lfc_nutlin"]).to_string(index=False)),
        mo.md("### Top 15 depleted / enriched — S63845"),
        mo.plain_text(_top(lfc_df["lfc_s63845"]).to_string(index=False)),
    ])


# ── 10. Summary stats ─────────────────────────────────────────────────────────

@app.cell
def _(count_mat, gene_col, lfc_df, mo, pd):
    _input_cols = [c for c in count_mat.columns if c.endswith("i")]
    _med_cnt = count_mat[_input_cols].mean(axis=1).median()

    _summary = pd.DataFrame({
        "Metric": [
            "Total guides",
            "Unique genes",
            "Mean guides per gene",
            "Median Input count per guide",
            "Genes LFC(nutlin) > 1",
            "Genes LFC(nutlin) < -1",
            "Genes LFC(s63845) > 1",
            "Genes LFC(s63845) < -1",
            "Genes |LFC(dmso)| < 0.2  (expected ~0)",
        ],
        "Value": [
            count_mat.shape[0],
            gene_col.nunique(),
            f"{count_mat.shape[0] / gene_col.nunique():.2f}",
            f"{_med_cnt:.1f}",
            int((lfc_df["lfc_nutlin"] > 1).sum()),
            int((lfc_df["lfc_nutlin"] < -1).sum()),
            int((lfc_df["lfc_s63845"] > 1).sum()),
            int((lfc_df["lfc_s63845"] < -1).sum()),
            int((lfc_df["lfc_dmso"].abs() < 0.2).sum()),
        ],
    })

    mo.vstack([
        mo.md("## Summary"),
        mo.plain_text(_summary.to_string(index=False)),
    ])


if __name__ == "__main__":
    app.run()
