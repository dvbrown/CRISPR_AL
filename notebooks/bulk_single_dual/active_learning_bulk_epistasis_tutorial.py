import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Active learning tutorial: bulk CRISPR epistasis (K562)

    This notebook pairs with `notebooks/bulk_single_dual/scripts/al_bulk_epistasis.py` to demonstrate how
    to run an active learning loop that **conditions on single-gene effects** and learns **epistatic
    deviations** on top of an overparameterized additive baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning goals
    - Map single-gene CRISPR effects into an additive baseline for gene pairs.
    - Train a residual model that captures nonlinear epistatic signals.
    - Use uncertainty-aware acquisition and diversity selection to choose new pairs.
    - Track hit rates and learning curves across rounds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Additive baseline (NAIAD-style)
    For a pair $(i, j)$ with single-gene effects $Y_i$ and $Y_j$:

    $$x_{ij} = [Y_i, Y_j] \in \mathbb{R}^2$$
    $$Y_{\text{additive}} = \phi(x_{ij} W_1) A_1^\top$$

    The overparameterized layer $W_1$ provides a high-dimensional additive embedding, and the
    residual model learns deviations from that baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Active learning loop
    1. Seed an initial batch of labeled gene pairs.
    2. Train an ensemble surrogate model (additive + residual).
    3. Compute acquisition scores: $|\mu| + \beta \sigma$.
    4. Apply diversity selection (k-center by default).
    5. Query the GI oracle, append labels, and repeat.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = (NOTEBOOK_DIR / "../../outputs").resolve()
    SCRIPT_PATH = (NOTEBOOK_DIR / "scripts/al_bulk_epistasis.py").resolve()
    ROUNDS_PATH = OUTPUT_DIR / "rounds.csv"
    METRICS_PATH = OUTPUT_DIR / "metrics.csv"
    SELECTED_PATH = OUTPUT_DIR / "selected_pairs.csv"
    return METRICS_PATH, ROUNDS_PATH, SELECTED_PATH, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the compute script

    The active learning loop is executed by a standalone script so that this notebook stays lightweight. If you downloaded the datasets into the default folders, you can run:

    ```bash
      python {SCRIPT_PATH}
    ```

    To override the defaults, provide your local DepMap and Horlbeck data files:

    ```bash
      python {SCRIPT_PATH} \
        --depmap-file /path/to/depmap_single_gene.csv \
        --gi-file /path/to/horlbeck_gi_map.csv
    ```

    Expected outputs:
      - `{ROUNDS_PATH}`
      - `{ROUNDS_PATH.parent / "metrics.csv"}`
      - `{ROUNDS_PATH.parent / "selected_pairs.csv"}`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Notes for running scripts

    - Run commands from the repository root so relative paths resolve correctly.
    - If you want to keep outputs in a different folder, pass `--output-dir`.
    - The script auto-detects CSV vs TSV by file extension; override with `--depmap-sep` or `--gi-sep`.

        Example with extra options:
        ```bash
        python {SCRIPT_PATH} \
          --depmap-file /path/to/depmap_single_gene.tsv \
          --gi-file /path/to/horlbeck_gi_map.tsv \
          --depmap-cell-line K562 \
          --diversity kcenter \
          --n-rounds 6 \
          --output-dir outputs/bulk_epistasis
        ```
    """)
    return


@app.cell
def _(METRICS_PATH, ROUNDS_PATH, SELECTED_PATH, mo, pd):
    if not ROUNDS_PATH.exists():
        mo.md("Outputs not found yet. Run the compute script to generate them.")
        rounds = None
        metrics = None
        selected = None
    else:
        rounds = pd.read_csv(ROUNDS_PATH)
        metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
        selected = pd.read_csv(SELECTED_PATH) if SELECTED_PATH.exists() else None
        mo.md(f"Loaded {len(rounds)} rounds from `{ROUNDS_PATH}`")
    return


app._unparsable_cell(
    r"""
    if rounds is None:
        return
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=rounds, x="round", y="hit_rate_batch", marker="o", label="Batch hit rate"
    )
    sns.lineplot(
        data=rounds,
        x="round",
        y="hit_rate_labeled",
        marker="o",
        label="Labeled hit rate",
    )
    plt.title("Hit rates over active learning rounds")
    plt.xlabel("Round")
    plt.ylabel("Hit rate (|GI| >= threshold)")
    plt.tight_layout()
    plt.show()
    """,
    name="_",
)


app._unparsable_cell(
    r"""
    if rounds is None:
        return
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=rounds, x="round", y="corr_mu_oracle", marker="o")
    plt.title("Correlation between predictions and oracle GI")
    plt.xlabel("Round")
    plt.ylabel("Pearson r")
    plt.tight_layout()
    plt.show()
    """,
    name="_",
)


app._unparsable_cell(
    r"""
    if selected is None:
        return
    selected.head(10)
    """,
    name="_",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps
    - Swap the diversity strategy (`--diversity kmeans` or `--diversity typiclust`).
    - Compare different additive hidden sizes or ensemble sizes.
    - Plot acquisition scores versus oracle GI to visualize calibration.
    """)
    return


if __name__ == "__main__":
    app.run()
