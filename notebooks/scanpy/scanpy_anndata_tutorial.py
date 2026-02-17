import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scanpy AnnData Tutorial (K562 Essential)

    This notebook is a hands-on tutorial for the Scanpy `AnnData` data structure using the K562 essential Perturb-seq dataset.

    Dataset path: `data/real/k562_essential_perturb_seq/v2025-09-03/replogle_k562_essential_perturbation.h5ad`.

    To keep runtime reasonable, we sample `N_CELLS` cells immediately after loading.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning goals
    - Load a large `.h5ad` and sample cells efficiently.
    - Understand the core AnnData attributes and how they relate.
    - Locate perturbation labels and cell identifiers.
    - Perform common manipulations used in ML-ready perturb-seq workflows.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from IPython.display import display

    np.random.seed(7)
    return Path, display, np, pd, sc, sp


@app.cell
def _(Path):
    rel_data_path = Path(
        "data/real/k562_essential_perturb_seq/v2025-09-03/replogle_k562_essential_perturbation.h5ad"
    )
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = (base / rel_data_path).resolve()
        if candidate.exists():
            DATA_PATH = candidate
            break
    else:
        raise FileNotFoundError(
            "Could not find the tutorial dataset. Expected to find "
            f"`{rel_data_path}` from the current working directory or one of its parents. "
            f"Current working directory: `{cwd}`."
        )
    N_CELLS = 1000
    RANDOM_SEED = 7
    return DATA_PATH, N_CELLS, RANDOM_SEED


@app.cell
def _(DATA_PATH, N_CELLS, RANDOM_SEED, np, sc):
    adata_full = sc.read_h5ad(DATA_PATH)

    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(N_CELLS, adata_full.n_obs)
    sample_idx = rng.choice(adata_full.n_obs, size=n_sample, replace=False)

    adata = adata_full[sample_idx].copy()
    del adata_full

    adata
    return (adata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AnnData anatomy

    AnnData is a container that keeps a matrix of measurements aligned with row (cell) and column (gene) metadata. The key contract is: **rows are observations (cells)** and **columns are variables (genes/features)**. Every attribute below must stay aligned with that contract.

    Core attributes:
    - `adata.X`: Main data matrix with shape `(n_cells, n_genes)`. Often sparse (counts) or log-normalized values.
    - `adata.obs`: Per-cell metadata as a pandas DataFrame. Index is the cell ID (`obs_names`).
    - `adata.var`: Per-gene metadata as a pandas DataFrame. Index is the gene ID (`var_names`).
    - `adata.uns`: Unstructured metadata (dict). Stores analysis settings, color maps, etc.
    - `adata.layers`: Additional matrices with the same shape as `X` (raw counts, log1p, corrected).
    - `adata.obsm`: Per-cell multi-dimensional arrays (PCA, UMAP, embeddings). Shape `(n_cells, k)`.
    - `adata.varm`: Per-gene multi-dimensional arrays (gene embeddings, loadings). Shape `(n_genes, k)`.
    - `adata.obsp`: Pairwise cell-cell matrices (neighbors graph, connectivities). Shape `(n_cells, n_cells)`.
    - `adata.raw`: Optional immutable snapshot of `X` and `var` (usually raw counts before normalization).

    Indexing rules and safety:
    - `adata.obs_names` are cell IDs; `adata.var_names` are gene IDs.
    - `adata[obs, var]` slices return a **view** by default. Use `.copy()` if you need an independent object.
    - If `X` is sparse, convert small slices to dense only when needed.
    """)
    return


@app.cell
def _(adata, display):
    print(f"shape: {adata.shape}")
    print(f"X type: {type(adata.X)}")
    display(adata.obs.head())
    return


@app.cell
def _(adata, display):
    display(adata.var.head())
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var columns: {list(adata.var.columns)}")
    print(f"uns keys: {list(adata.uns.keys())}")
    print(f"layers: {list(adata.layers.keys())}")
    print(f"obsm keys: {list(adata.obsm.keys())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell IDs and gene expression access

    Cells are indexed by `adata.obs_names` (often a barcode), and genes are indexed by `adata.var_names` (gene symbols or Ensembl IDs). You can select cells/genes by name or position.

    For expression values, be mindful of sparse vs dense matrices. When `X` is sparse, convert only small subsets to dense arrays.
    """)
    return


@app.cell
def _(adata, sp):
    cell_id = adata.obs_names[0]
    gene_id = adata.var_names[0]

    value = adata[cell_id, gene_id].X
    if sp.issparse(value):
        value = value.toarray()[0, 0]
    else:
        value = float(value)

    value
    return (gene_id,)


@app.cell
def _(adata, gene_id, np, sp):
    gene_expr = adata[:, gene_id].X
    if sp.issparse(gene_expr):
        gene_expr = gene_expr.toarray().ravel()
    else:
        gene_expr = np.ravel(gene_expr)

    gene_expr[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perturbations and guide information

    For Perturb-seq datasets, perturbations or guide identities are usually stored in `adata.obs`. This dataset includes `condition` and `condition_name` columns that encode perturbations (gene-level targets plus control). The `control` column is a binary indicator of control cells.

    If guide-level data were present, you would usually see columns containing `guide`, `sgRNA`, or `barcode`. We check for these columns below.
    """)
    return


@app.cell
def _(adata):
    def find_guide_columns(df):
        keywords = ["guide", "grna", "sgrna", "barcode", "sg", "condition"]
        return [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in keywords)
        ]

    guide_obs_cols = find_guide_columns(adata.obs)
    guide_var_cols = find_guide_columns(adata.var)

    guide_obs_cols, guide_var_cols
    return


@app.cell
def _(adata, display):
    pert_col = "condition" if "condition" in adata.obs.columns else None

    pert_counts = adata.obs[pert_col].value_counts()
    display(pert_counts.head())

    if "control" in adata.obs.columns:
        display(adata.obs["control"].value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common manipulations

    These are the most common AnnData transformations in ML workflows: subsetting, adding metadata, creating normalized layers, and building embeddings. The examples below are intentionally explicit so you can adapt them to your own pipelines.

    Important rule: slicing returns a view by default. Use `.copy()` when creating a dataset that you will mutate.
    """)
    return


@app.cell
def _(adata):
    def pick_column(df, candidates):
        for name in candidates:
            if name in df.columns:
                return name
        return None

    pert_col_1 = pick_column(adata.obs, ["condition", "perturbation", "target_gene"])
    control_col = pick_column(adata.obs, ["control"])
    if control_col:
        control_mask = adata.obs[control_col] == 1
    else:
        control_mask = adata.obs[pert_col_1] == "ctrl"
    adata_ctrl = adata[control_mask].copy()
    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        adata_hvg = adata[:, hvg_mask].copy()
    else:
        adata_hvg = adata
    (adata_ctrl.shape, adata_hvg.shape)
    return (pert_col_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Add per-cell or per-gene metadata

    You can add derived metrics to `obs` or `var`. These are simple pandas columns aligned with cells or genes, which makes later filtering straightforward.
    """)
    return


@app.cell
def _(adata, np, sp):
    adata_1 = adata.copy()
    if sp.issparse(adata_1.X):
        adata_1.obs["n_genes_detected"] = np.asarray(
            (adata_1.X > 0).sum(axis=1)
        ).ravel()
        adata_1.var["mean_expression"] = np.asarray(adata_1.X.mean(axis=0)).ravel()
    else:
        adata_1.obs["n_genes_detected"] = (adata_1.X > 0).sum(axis=1)
        adata_1.var["mean_expression"] = adata_1.X.mean(axis=0)
    adata_1.obs[["n_genes_detected"]].head()
    return (adata_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Layers and normalization

    A common pattern is to store raw counts in `layers["counts"]`, then normalize/log-transform `X` and store the result in another layer. This preserves both representations.
    """)
    return


@app.cell
def _(adata_1, sc):
    adata_proc = adata_1.copy()
    adata_proc.layers["counts"] = adata_proc.X.copy()
    sc.pp.normalize_total(adata_proc, target_sum=10000.0)
    sc.pp.log1p(adata_proc)
    adata_proc.layers["log1p_norm"] = adata_proc.X.copy()
    list(adata_proc.layers.keys())
    return (adata_proc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embeddings in `obsm`

    Embeddings store lower-dimensional representations for each cell. Scanpy uses `obsm` with the convention `X_pca`, `X_umap`, etc. These arrays always align with `obs` (cells).
    """)
    return


@app.cell
def _(adata_proc, sc):
    sc.tl.pca(adata_proc, n_comps=30, svd_solver="arpack")
    adata_proc.obsm["X_pca"].shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ML-ready matrices and splits

    For supervised learning, you typically export a feature matrix `X` and a label vector `y`. Perturb-seq labels are often gene targets (`condition`) or guide IDs when available. This dataset also includes `split` for train/test assignment.
    """)
    return


@app.cell
def _(adata_proc, pert_col_1, sp):
    X = adata_proc.layers["log1p_norm"]
    y = adata_proc.obs[pert_col_1].astype(str).values
    train_test_shapes = None
    if sp.issparse(X):
        X = X.tocsr()
    if "split" in adata_proc.obs.columns:
        split = adata_proc.obs["split"].astype(str).values
        train_mask = split == "train"
        test_mask = split == "test"
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        train_test_shapes = (X_train.shape, X_test.shape)
    train_test_shapes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other ML-relevant structures in pooled CRISPR screens

    Beyond `X` and `y`, common artifacts used in perturb-seq and pooled CRISPR modeling include:
    - **Perturbation design matrix**: a cell-by-perturbation indicator (one-hot or multi-hot for multiplexed guides).
    - **Guide-to-gene mapping**: a table linking sgRNA IDs to target genes, often stored separately.
    - **Covariates**: batch, dose, cell cycle, and technical effects stored in `obs`.
    - **Per-perturbation summaries**: pseudobulk profiles or delta vectors stored in `uns` or external tables.
    - **Prior features**: gene/perturbation embeddings or pathway features stored in `varm` or `uns`.
    - **Graphs**: cell-cell or gene-gene graphs stored in `obsp` for neighborhood or network models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pseudobulk per perturbation

    Pseudobulk aggregation averages expression per perturbation to produce one vector per perturbation. This is useful for downstream modeling and benchmarking. For speed, we demonstrate it on a small gene subset.
    """)
    return


@app.cell
def _(adata_proc, pd, pert_col_1, sp):
    genes = list(adata_proc.var_names[:50])
    expr = adata_proc[:, genes].X
    if sp.issparse(expr):
        expr = expr.toarray()
    expr_df = pd.DataFrame(expr, columns=genes, index=adata_proc.obs_names)
    expr_df["condition"] = adata_proc.obs[pert_col_1].values
    pseudobulk = expr_df.groupby("condition")[genes].sum()
    pseudobulk.head()
    return


@app.cell
def _(Path, adata_proc):
    OUT_PATH = Path("scanpy_anndata_tutorial_sampled.h5ad")
    adata_proc.write(OUT_PATH)
    OUT_PATH
    return


if __name__ == "__main__":
    app.run()
