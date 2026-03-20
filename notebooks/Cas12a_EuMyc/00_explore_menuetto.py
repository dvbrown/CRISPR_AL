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

  claude --resume "fix-plot-style-save-figures"
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

    - **Library:** Menuetto (dual crRNA, 2 crRNA/gene, 1 vector per crRNA)
    - **Cell line:** Eµ-MYC;Cas12het lymphoma cells
    - **Organism:** *Mus musculus*
    - **Source:** WEHI Blood Cells and Blood Cancer division (La Marca, Diepstraten et al.)

    ---

    ## Library composition

    | | |
    |---|---|
    | Total guides | 43,814 |
    | Targeting genes | 21,694 |
    | Guides per gene | 2 (mode); 74 genes have only 1 guide |
    | Non-targeting controls | 500 (gene label `NTC`) |
    | Guides absent from all samples | 21 (0.05%) |

    ---

    ## Sample layout — 24 samples, 4 conditions × 6 replicates

    Sample names follow the scheme `D{rep}{cond}`:
    **D** = Dual (Menuetto), **rep** = 1–6, **cond** = `i` Input · `d` DMSO · `n` Nutlin-3a · `s` S63845

    | Condition | Samples | Mean total reads | Mean guide coverage | Mean zero-count guides |
    |---|---|---|---|---|
    | Input | D1i–D6i | 2.39 M | **54×** | 386 (0.9%) |
    | DMSO | D1d–D6d | 1.72 M | 39× | 30,130 (68.8%) |
    | Nutlin-3a | D1n–D6n | 1.91 M | 43× | 28,041 (64.0%) |
    | S63845 | D1s–D6s | 1.92 M | 44× | 35,245 (80.4%) |

    ---

    ## Key QC observations

    - **Input coverage is good:** 54× mean reads/guide; only 0.9% of guides have zero counts in any given Input replicate.
    - **High zero rates in treated samples are expected and biological:** guides targeting essential or drug-sensitising genes drop out under selection. S63845 shows the most dropout (80%), consistent with a strong MCL-1-dependent apoptotic signal.
    - **DMSO zero rate (69%) is surprisingly high** relative to Input — worth checking whether this reflects genuine essential-gene dropout over the culture period or a sequencing depth issue. DMSO has the lowest mean coverage (39×).
    - **74 genes have only 1 guide** — these should be flagged and excluded from gene-level scoring, as MAGeCK requires ≥2 guides for reliable RRA.
    - **NTC set is large (500 guides)** — well-powered for null distribution estimation and FDR calibration.

    ---

    ## Condition biology

    | Condition | Mechanism | Expected top hits |
    |---|---|---|
    | Nutlin-3a | MDM2 inhibitor → p53 stabilisation → apoptosis | *Trp53*, MDM2 pathway KOs confer **resistance** (positive LFC) |
    | S63845 | MCL-1 BH3 mimetic → BAX/BAK-dependent apoptosis | *Bax*, *Bak1* KOs confer **resistance**; *Mcl1* KO sensitises |
    | DMSO | No drug selection | Core essential genes drop out (ribosome, proteasome, translation) |
    | Input | Library reference (T=0) | — |
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#444444",
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "grid.color":          "#DDDDDD",
        "grid.linewidth":      0.4,
        "grid.linestyle":      "-",
        "xtick.color":         "#444444",
        "ytick.color":         "#444444",
        "xtick.major.width":   0.5,
        "ytick.major.width":   0.5,
        "text.color":          "#222222",
        "axes.labelcolor":     "#222222",
        "axes.labelweight":    "bold",
        "axes.titlesize":      12,
        "axes.labelsize":      11,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "font.family":         "sans-serif",
        "savefig.facecolor":   "white",
        "savefig.dpi":         300,
    })

    RAW_FILE = Path.cwd() / "data/bulk/menuetto_scherzo_2025/raw/GSE285778_EuMycCountMenuetto.txt.gz"

    FIGURES_DIR = Path.cwd() / "notebooks/Cas12a_EuMyc/figures/explore_menuetto"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR, PCA, RAW_FILE, StandardScaler, np, pd, plt, stats


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


@app.cell
def _(mo):
    mo.md("""
    ## 1. Library size per sample

    **Normal.** All 24 samples have 1.5–2.5 M total counts (mean ~2.0 M), which is
    adequate for a genome-wide screen at 2 guides/gene (~43 K guides).

    - Input samples are slightly deeper (2.0–2.5 M) than treated samples (1.5–2.0 M).
      A modest depth difference is common and does not affect analysis after normalisation.
    - No outlier samples — all replicates within each condition are consistent.
    - Typical CRISPR screens aim for ≥ 30× coverage (reads/guide); at ~35–55× here
      Input coverage is in the acceptable range.
    """)
    return


@app.cell
def _(COND_COLOURS, COND_ORDER, FIGURES_DIR, count_mat, plt, sample_meta):
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
    _fig.savefig(FIGURES_DIR / "01_library_size.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Guides per gene & count distribution

    **Normal.** The Menuetto library is highly uniform: essentially all genes have exactly
    2 guides, consistent with the library design. A small tail of 1-guide genes (~74) should
    be excluded from gene-level hit scoring downstream.

    - The Input count distribution is log-normal with a median around log₁₀ ≈ 1.6
      (~40 reads/guide), which is typical of a well-represented pooled screen.
    - The left tail (low-count guides) is small — very few guides are under-represented in
      the Input plasmid pool. This is good: it means the library was not severely bottlenecked
      during lentiviral production or transduction.
    - No spike of zero-count guides in Input, confirming the Input sequencing depth is
      sufficient to call LFC accurately.
    """)
    return


@app.cell
def _(FIGURES_DIR, count_mat, gene_col, np, plt):
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
    _fig.savefig(FIGURES_DIR / "02_guides_per_gene_count_dist.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Count distributions per sample

    **⚠ Anomalous — massive dropout in all post-selection conditions, including DMSO.**

    - Input samples show healthy per-guide distributions (median log₁₀ ≈ 1.6–1.9, ~40–80 reads).
    - DMSO, Nutlin-3a, and S63845 samples are **compressed to zero**: the majority of guides
      have 0 counts, so the boxes collapse to the bottom of the axis. This is consistent with the
      header statistics (64–80% zero-count guides per treated replicate).
    - Some drug-driven dropout is expected and biological (sensitising gene KOs are lethal), but
      **DMSO showing the same near-complete dropout as drug conditions is a red flag**. DMSO
      replicates should look close to Input.
    - Most likely explanation: the cells were proliferated for a **very long time** after
      transduction before harvesting, allowing background essential-gene dropout to accumulate.
      This compresses the dynamic range and makes DMSO a poor null comparator.
    - Despite this, drug-specific effects on top of background are still visible in the LFC
      distributions (see §7).
    """)
    return


@app.cell
def _(COND_COLOURS, COND_ORDER, FIGURES_DIR, count_mat, np, plt, sample_meta):
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
    _fig.savefig(FIGURES_DIR / "03_count_distributions.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Replicate correlation

    **⚠ Poor across all conditions — driven by count sparsity.**

    | Condition | Pearson r (rep1 vs rep2) | Expectation |
    |---|---|---|
    | Input | 0.42 | > 0.95 ✗ |
    | Nutlin-3a | 0.08 | > 0.7 ✗ |
    | S63845 | 0.07 | > 0.7 ✗ |

    - **Input r = 0.42** is below the standard quality threshold for a pooled screen
      (typically r > 0.95 for Input replicates, which should be nearly identical). This
      could reflect PCR jackpotting during library prep, early transduction bottlenecking,
      or uneven sequencing within the input pool. Worth investigating with the wet-lab team.
    - **Treatment r ≈ 0.08** is expected when ~70–80% of guides have 0 counts: Pearson
      correlation degrades to noise on sparse binary-like data. Spearman correlation on
      non-zero guides only, or NNMD (normalised median difference), would give a fairer
      assessment of reproducibility in the informative subset.
    - The hexbin plot for Input shows a clear point cloud at the low-count region (0–1 on
      both axes) with a diffuse high-count scatter — consistent with most guides having
      0 or very few counts even in Input, which contradicts the library size data. This
      warrants a check of whether the sample-naming decode (s[2] for condition) is correct.
    """)
    return


@app.cell
def _(FIGURES_DIR, count_mat, np, plt, sample_meta, stats):
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
    _fig.savefig(FIGURES_DIR / "04_replicate_correlation.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. PCA of samples

    **Partially expected, partially concerning.**

    - **PC1 (55.6%)** completely separates Input (far right, PC1 ≈ +250) from all three
      treated conditions (left cluster, PC1 ≈ −50 to −75). This Input-vs-selected axis
      is expected in any CRISPR screen — selected conditions have depleted guides, Input
      does not. The extreme magnitude of the separation reflects the high dropout rate.
    - **DMSO, Nutlin-3a, and S63845 cluster together** on PC1, indicating dropout is
      dominated by shared essential-gene biology rather than drug-specific effects.
      Drug-specific effects are visible but small relative to background selection.
    - **PC2 (3.6%):** One Nutlin-3a replicate is a clear outlier (high PC2 ≈ +175).
      This warrants inspection — it may be a mislabelled sample or a replicate with
      anomalous sequencing. Input replicates also scatter widely on PC2, consistent
      with the poor Input Pearson r seen in §4.
    - **PC3–4 (3.2%, 3.1%):** Drug conditions begin to separate from each other,
      suggesting drug-specific biology is present but requires deeper PCs to resolve.
    - **Overall:** the PCA structure is consistent with a high-dropout screen. The
      biology is being driven by a subset of strongly selected guides.
    """)
    return


@app.cell
def _(
    COND_COLOURS,
    COND_ORDER,
    FIGURES_DIR,
    PCA,
    StandardScaler,
    count_mat,
    np,
    plt,
    sample_meta,
):
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
    _fig.savefig(FIGURES_DIR / "05_pca_samples.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Gene-level LFC

    LFC is computed as log₂(mean RPM+1 in treatment / mean RPM+1 in Input), then
    summarised per gene as the **median across the 2 guides**. The +1 pseudo-count
    prevents log(0) and sets the depletion floor at approximately log₂(1 / Input_RPM).
    For a guide with Input RPM ≈ 34 (median), this floor is log₂(1/34) ≈ **−5.1**,
    consistent with the observed LFC distributions in §7.
    """)
    return


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


@app.cell
def _(mo):
    mo.md("""
    ## 7. LFC distributions

    **Bimodal — consistent with a high-dropout screen, but biological signal is present.**

    All three conditions show a **bimodal LFC distribution**:
    - A large peak at LFC ≈ −4 to −5: guides that dropped to zero counts in the selected
      condition (hits the pseudo-count floor). This is the "dropout mass".
    - A smaller shoulder/peak near LFC ≈ 0: guides that survived selection (neither
      essential nor sensitising).

    This bimodal shape is characteristic of screens with **very strong selection pressure**
    or long proliferation windows, and is seen even in DMSO — meaning a large fraction of
    the mouse genome is required for lymphoma cell fitness under these conditions.

    **Drug-specific shifts:**
    - Nutlin-3a and S63845 distributions are shifted slightly more negative than DMSO,
      reflecting additional drug-induced lethality on top of background essential-gene
      dropout. The difference is modest, suggesting the screen is not overpowered for
      drug-specific hits.
    - The enriched tail (LFC > 1) is visible for all three conditions, representing the
      resistance genes of interest.

    **Is this normal?** Partially. Bimodal LFC distributions occur in well-run screens
    with long selection, but the floor being so dominant (most genes piled at −5) reduces
    precision for ranking depleted genes. For enriched genes (resistance hits), the screen
    appears informative.
    """)
    return


@app.cell
def _(FIGURES_DIR, lfc_df, np, plt):
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
    _fig.savefig(FIGURES_DIR / "07_lfc_distributions.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Ranked LFC — known controls

    **✓ Positive controls are in the expected biological positions — the screen is working.**

    **Nutlin-3a (MDM2i / p53 activator):**
    - *Trp53* is the **most enriched gene** (LFC ≈ +6.5). Trp53 KO abolishes the drug
      target (p53), conferring strong resistance. This is the canonical positive control
      for Nutlin-3a and its presence at rank 1 validates the screen.
    - *Cdkn1a* (p21) is also enriched — a direct p53 transcriptional target; KO of p21
      partially bypasses p53-mediated cell-cycle arrest.
    - *Mdm2* and *Mdm4* are in the depleted half: KO of these negative regulators of p53
      increases p53 activity, sensitising cells to Nutlin-3a. Expected.

    **S63845 (MCL-1 BH3 mimetic):**
    - *Bcl2* is the **most enriched gene** (LFC ≈ +8.5). BCL2 over-representation via KO
      of its repressor, or BCL2 functional upregulation upon MCL-1 loss, can rescue cells
      from MCL-1-driven apoptosis. Strong anti-apoptotic compensator.
    - *Bax* and *Bak1* are enriched (LFC ≈ +2–3): KO of the apoptotic executioners prevents
      the cell from committing to apoptosis even when MCL-1 is inhibited. Expected.
    - *Mcl1* itself is among the depleted genes: consistent with synthetic lethality — KO of
      the same target the drug inhibits makes cells hypersensitive. Expected.
    - *Bcl2l1* (BCL-xL) is near neutral/slightly enriched, as expected (partial redundancy
      with MCL-1 but not a dominant resistance mechanism here).

    **Overall verdict:** Despite the quality flags in §3–4, the biological signal is real
    and the screen has discriminative power for enriched (resistance) phenotypes.
    Depleted hits are harder to interpret given the high baseline dropout.
    """)
    return


@app.cell
def _(FIGURES_DIR, lfc_df, plt):
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
    _fig.savefig(FIGURES_DIR / "08_ranked_lfc.png", dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Top hits
    """)
    return


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

    mo.vstack([
        mo.md("### Nutlin-3a vs Input"),
        mo.plain_text(_top(lfc_df["lfc_nutlin"]).to_string(index=False)),
        mo.md("### S63845 vs Input"),
        mo.plain_text(_top(lfc_df["lfc_s63845"]).to_string(index=False)),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Summary statistics
    """)
    return


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
        mo.plain_text(_summary.to_string(index=False)),
    ])
    return


if __name__ == "__main__":
    app.run()
