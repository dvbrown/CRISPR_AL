import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iterative Perturb-seq Tutorial (K562 Essential)

    This notebook walks through how IterPert (iterative Perturb-seq) works experimentally, computationally, and mathematically. We will use the local K562 essential dataset that is already stored in this repo.

    Dataset path: `data/real/k562_essential_perturb_seq/v2025-09-03/replogle_k562_essential_perturbation.h5ad`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning goals
    - Understand the experimental Perturb-seq loop and why iteration matters.
    - Connect perturbation effects to the modeling targets used by IterPert/GEARS.
    - See how kernel-based batch active learning selects the next perturbations.
    - Map the math (effects, kernels, priors) to concrete code and data objects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Experimental overview
    Perturb-seq combines CRISPR perturbations with single-cell RNA-seq. In a classical screen, you perturb a large gene set and measure transcriptional outcomes once. Iterative Perturb-seq instead runs multiple rounds:

    1. **Round 0**: perturb a small seed set, profile scRNA-seq.
    2. **Modeling**: train a predictor of perturbation effects.
    3. **Selection**: choose the next batch of perturbations that is expected to be most informative.
    4. **Wet-lab**: run the selected perturbations, append data, repeat.

    This is an experimental design problem: you trade off experimental budget with information gain. IterPert targets the “active learning under budget” regime where only a small fraction of perturbations are measured per round.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Computational overview
    IterPert wraps three main components:

    1. **Data processing**: build perturbation labels and control baselines.
    2. **Predictive model**: GEARS-style neural network predicts perturbation effects.
    3. **Active learning**: kernel-based batch selection chooses new perturbations.

    IterPert adds *multimodal prior kernels* (protein embeddings, pathway features, etc.) that help guide selection when the labeled set is small. The result is a batch selection method that balances uncertainty, diversity, and prior biological knowledge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mathematical framing
    Let $x_i \in \mathbb{R}^G$ be the expression vector for cell $i$ (across $G$ genes). For perturbation $p$ with cells $C_p$ and control cells $C_{ctrl}$, define:

    $$\mu_p = \frac{1}{|C_p|} \sum_{i \in C_p} x_i, \quad \mu_{ctrl} = \frac{1}{|C_{ctrl}|} \sum_{i \in C_{ctrl}} x_i$$
    $$\Delta_p = \mu_p - \mu_{ctrl}$$

    A model $f_θ(p)$ predicts $\hat{\Delta}_p$. IterPert builds a kernel $K$ over perturbations and selects a batch $B$ that maximizes an acquisition function (e.g., max-distance, BatchBALD). For multiple prior kernels $K_m$ (protein, GO, coexpression, etc.), IterPert forms a normalized average:

    $$\hat{K}_m = D_m^{-1/2} K_m D_m^{-1/2}, \quad K = \frac{1}{M} \sum_{m=1}^M \hat{K}_m$$

    Selection then runs on $K$ (or a learned combination) to choose the next perturbations to profile.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import scanpy as sc
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances

    sns.set_theme(style="whitegrid")
    np.random.seed(7)
    return PCA, Path, np, pairwise_distances, pd, plt, sc, sns, sp


@app.cell
def _(Path, sc):
    DATA_PATH = Path("../../data/real/k562_essential_perturb_seq/v2025-09-03/replogle_k562_essential_perturbation.h5ad").resolve()

    adata = sc.read_h5ad(DATA_PATH)
    adata
    return (adata,)


@app.cell
def _(adata):
    def pick_column(df, candidates):
        for name in candidates:
            if name in df.columns:
                return name
        return None

    condition_col = pick_column(adata.obs, ["condition", "perturbation", "target_gene", "gene"])
    cell_type_col = pick_column(adata.obs, ["cell_type", "cell_line", "celltype"])

    if condition_col is None:
        raise ValueError("Could not find a perturbation column in adata.obs.")

    print(f"Condition column: {condition_col}")
    print(f"Cell type column: {cell_type_col}")
    adata.obs[[condition_col]].head()
    return cell_type_col, condition_col


@app.cell
def _(adata, cell_type_col, condition_col):
    adata_1 = adata.copy()
    obs = adata_1.obs.copy()
    obs['condition_raw'] = obs[condition_col].astype(str)
    control_tokens = {'ctrl', 'control', 'non-targeting', 'non_targeting', 'nt'}
    control_mask = obs['condition_raw'].str.lower().isin(control_tokens)
    if obs['condition_raw'].str.contains('\\+').any():
        obs['condition'] = obs['condition_raw']
    else:
        obs['condition'] = obs['condition_raw'].where(control_mask, obs['condition_raw'] + '+ctrl')
        obs.loc[control_mask, 'condition'] = 'ctrl'
    if cell_type_col:
        obs['cell_type'] = obs[cell_type_col].astype(str)
    adata_1.obs = obs
    pert_counts = adata_1.obs['condition'].value_counts()
    pert_counts.head(10)
    return adata_1, pert_counts


@app.cell
def _(pert_counts, plt, sns):
    top_perts = pert_counts.head(20)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_perts.values, y=top_perts.index, color="#4C72B0")
    plt.title("Top perturbations by cell count")
    plt.xlabel("Number of cells")
    plt.ylabel("Perturbation")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Perturbation effect vectors (demo)
    IterPert/GEARS predict *perturbation effects* (deltas relative to control). Below we compute a small demo matrix of perturbation effects using a subset of genes and perturbations. This keeps the computation manageable while preserving the logic.
    """)
    return


@app.cell
def _(adata_1, np, sp):
    all_perts = np.array([p for p in adata_1.obs['condition'].unique() if p != 'ctrl'])
    rng = np.random.default_rng(7)
    max_perts = min(200, len(all_perts))
    demo_perts = rng.choice(all_perts, size=max_perts, replace=False)
    n_demo_genes = min(300, adata_1.n_vars)
    gene_idx = np.arange(n_demo_genes)
    ctrl_X = adata_1[adata_1.obs['condition'] == 'ctrl', gene_idx].X
    if sp.issparse(ctrl_X):
        ctrl_mean = np.asarray(ctrl_X.mean(axis=0)).ravel()
    else:
        ctrl_mean = ctrl_X.mean(axis=0)
    effect_rows = []
    for pert in demo_perts:
        pert_X = adata_1[adata_1.obs['condition'] == pert, gene_idx].X
        if sp.issparse(pert_X):
            pert_mean = np.asarray(pert_X.mean(axis=0)).ravel()
        else:
            pert_mean = pert_X.mean(axis=0)
        effect_rows.append(pert_mean - ctrl_mean)
    effect_matrix = np.vstack(effect_rows)
    effect_matrix.shape
    return demo_perts, effect_matrix


@app.cell
def _(PCA, effect_matrix):
    pca = PCA(n_components=2, random_state=7)
    embedding = pca.fit_transform(effect_matrix)

    embedding[:5]
    return (embedding,)


@app.cell
def _(demo_perts, embedding, np, pairwise_distances, pd):
    def farthest_point_batch(emb, n_init=10, n_query=20, seed=7):
        rng = np.random.default_rng(seed)
        selected = list(rng.choice(np.arange(emb.shape[0]), size=n_init, replace=False))
        for _ in range(n_query):
            dist = pairwise_distances(emb, emb[selected], metric="euclidean")
            score = dist.min(axis=1)
            score[selected] = -np.inf
            selected.append(int(np.argmax(score)))
        return selected

    selected_idx = farthest_point_batch(embedding, n_init=10, n_query=20)
    selected_mask = np.zeros(embedding.shape[0], dtype=bool)
    selected_mask[selected_idx] = True

    selection_df = pd.DataFrame({
        "perturbation": demo_perts,
        "pc1": embedding[:, 0],
        "pc2": embedding[:, 1],
        "selected": selected_mask
    })
    selection_df.head()
    return (selection_df,)


@app.cell
def _(plt, selection_df, sns):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=selection_df, x="pc1", y="pc2", hue="selected", palette={False: "#A9A9A9", True: "#E15759"})
    plt.title("Demo: max-distance batch selection in PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Prior kernels and IterPert integration
    IterPert augments the model-derived kernel with **prior kernels** (protein embeddings, GO similarities, perturbation signatures). The code below shows the normalization + averaging used conceptually in the paper and repository.
    """)
    return


@app.cell
def _(effect_matrix, embedding, np):
    def normalize_kernel(K):
        diag = np.sqrt(np.diag(K))
        diag[diag == 0] = 1.0
        return K / np.outer(diag, diag)

    K_base = embedding @ embedding.T
    K_base = normalize_kernel(K_base)

    K_prior_1 = normalize_kernel(K_base + 0.05 * np.eye(K_base.shape[0]))
    K_prior_2 = normalize_kernel(np.corrcoef(effect_matrix))

    K_iterpert = np.mean([K_base, K_prior_1, K_prior_2], axis=0)
    K_iterpert[:4, :4]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. How to run IterPert (if installed)
    The official IterPert API expects a dataset formatted like GEARS: `adata.obs` includes a `condition` column with values like `CTRL` or `GENE+ctrl`, and `adata.var` includes a `gene_name` column. The snippet below shows how you would wire the local dataset into the IterPert interface after normalizing the condition labels.
    """)
    return


@app.cell
def _(adata_1):
    if False:
        from iterpert.iterpert import IterPert
        iterpert_adata = adata_1.copy()
        iterpert_adata.obs['condition'] = iterpert_adata.obs['condition']  # Prepare a GEARS-style AnnData object.
        iterpert_adata.obs['cell_type'] = iterpert_adata.obs.get('cell_type', 'K562')
        iterpert_adata.var['gene_name'] = iterpert_adata.var.index.astype(str)
        iterpert_adata.uns.setdefault('log1p', {})
        iterpert_adata.uns['log1p']['base'] = None
        interface = IterPert(weight_bias_track=False, device='cuda:0', seed=1)
        interface.initialize_data(path='/path/to/cache/', dataset_name='replogle_k562_essential_custom', adata=iterpert_adata, batch_size=256)
        interface.initialize_model(epochs=20, hidden_size=64)
        interface.initialize_active_learning_strategy(strategy='IterPert')
        interface.start(n_init_labeled=100, n_round=5, n_query=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. References and next steps
    - IterPert repo: https://github.com/Genentech/iterative-perturb-seq
    - Paper: https://www.biorxiv.org/content/10.1101/2023.12.12.571389v1
    - Dataset metadata: `data/real/k562_essential_perturb_seq/v2025-09-03/metadata.json`

    Suggested extensions:
    - Swap PCA embeddings for a trained GEARS latent embedding.
    - Compare random, max-distance, and BALD selection on the same subset.
    - Add your own prior kernels (GO, protein embeddings) and inspect selection shifts.
    """)
    return


if __name__ == "__main__":
    app.run()
