# Aim 3 — Cross-Condition Transfer: Nutlin-3a ↔ S63845

**File:** `notebooks/Cas12a_EuMyc/03_aim3_cross_condition.py`
**Batch runner:** `notebooks/Cas12a_EuMyc/scripts/run_aim3_aim4.py`

---

## Scientific Question

Can a model trained on gene scores from one drug selection condition (nutlin-3a,
which activates p53) predict gene scores from a mechanistically distinct condition
(S63845, which inhibits MCL-1), and vice versa? Both conditions are run in the
same cell line with the same library.

This tests **biological generalisability** of the feature-based model: do gene
features (expression, pathway membership, co-essentiality) capture condition-
independent properties that predict drug-specific phenotypes?

### Biological context

| Condition | Mechanism | Primary hit | Expected enriched pathway |
|---|---|---|---|
| Nutlin-3a | MDM2 inhibitor → p53 activation → apoptosis | Trp53 KO confers resistance | p53 pathway, DNA damage response |
| S63845 | MCL-1 inhibitor → BAX-dependent apoptosis | Bax KO confers resistance | BCL-2 family, intrinsic apoptosis |
| DMSO | No selection | Essential gene dropout | Ribosome, proteasome, translation |

These are **mechanistically orthogonal** selection pressures. Low cross-condition
transfer is scientifically expected and informative — it tells us what fraction
of the gene score signal is condition-specific vs shared (e.g. essential genes
that drop out under any selection pressure).

---

## Inputs

| Input | Path | Description |
|---|---|---|
| Menuetto gene scores | `data/bulk/menuetto_scherzo_2025/processed/menuetto_gene_scores.parquet` | LFC per gene: nutlin, s63845, dmso |
| Scherzo gene scores | `data/bulk/menuetto_scherzo_2025/processed/scherzo_gene_scores.parquet` | LFC per gene: nutlin, s63845, dmso |
| Feature matrix | `data/bulk/menuetto_scherzo_2025/processed/features_mouse_genes.parquet` | Per-gene feature vectors |

---

## Transfer Matrix

All 12 cross-condition transfers are run (6 per library):

| Transfer ID | Library | Train condition | Predict condition |
|---|---|---|---|
| `M_nutlin→s63845` | Menuetto | Nutlin-3a | S63845 |
| `M_s63845→nutlin` | Menuetto | S63845 | Nutlin-3a |
| `M_nutlin→dmso` | Menuetto | Nutlin-3a | DMSO |
| `M_dmso→nutlin` | Menuetto | DMSO | Nutlin-3a |
| `M_s63845→dmso` | Menuetto | S63845 | DMSO |
| `M_dmso→s63845` | Menuetto | DMSO | S63845 |
| `S_nutlin→s63845` | Scherzo | Nutlin-3a | S63845 |
| `S_s63845→nutlin` | Scherzo | S63845 | Nutlin-3a |
| `S_nutlin→dmso` | Scherzo | Nutlin-3a | DMSO |
| `S_dmso→nutlin` | Scherzo | DMSO | Nutlin-3a |
| `S_s63845→dmso` | Scherzo | S63845 | DMSO |
| `S_dmso→s63845` | Scherzo | DMSO | S63845 |

Plus within-condition controls (same condition, different replicates split
50/50) to establish the within-condition ceiling.

---

## Split Design

For cross-condition transfer, all genes are used (no gene-level holdout).
The split is entirely across conditions:

```
Train set:  all ~21,700 genes, labelled with source condition LFC
Test set:   same ~21,700 genes, labelled with target condition LFC
```

This is a **zero-shot transfer** — the model never sees target condition labels
during training. Run 20 random seeds for the feature-based model to estimate
fitting variance.

```yaml
# splits.yaml
aim3:
  n_repeats: 20
  seeds: [0..19]
  gene_split: null        # all genes used; split is across conditions only
  within_condition_seeds: [100..119]  # separate seeds for within-condition control
```

---

## Method 1 — Direct Score Regression (Cross-Condition)

Directly regress target condition LFC on source condition LFC.
This measures the raw biological correlation between conditions,
independent of any feature model.

```python
def cross_condition_direct(
    source_lfc: np.ndarray,
    target_lfc: np.ndarray,
) -> dict:
    """
    Fit HuberRegressor: target_lfc ~ source_lfc.
    Returns metrics + slope/intercept.
    """
    from sklearn.linear_model import HuberRegressor
    model = HuberRegressor(epsilon=1.35)
    model.fit(source_lfc.reshape(-1, 1), target_lfc)
    y_pred = model.predict(source_lfc.reshape(-1, 1))
    metrics = evaluate_predictions(target_lfc, y_pred)
    metrics["slope"]     = float(model.coef_[0])
    metrics["intercept"] = float(model.intercept_)
    return metrics
```

**Expected:** Low correlation between nutlin and S63845 (r ≈ 0.1–0.3) because
the top hits are different (Trp53 vs Bax). Higher correlation between either
drug and DMSO for the essential gene component (r ≈ 0.3–0.5).

---

## Method 2 — Feature-Based Cross-Condition Transfer

Train a feature-based model on source condition labels, predict target condition.

```python
def feature_cross_condition(
    source_lfc: pd.Series,
    target_lfc: pd.Series,
    features: pd.DataFrame,
    model_type: str = "ridge",
    seed: int = 0,
) -> dict:
    """
    Train on source_lfc using gene features.
    Evaluate predictions against target_lfc.
    All genes are used (no gene-level holdout).
    """
    X = features.loc[source_lfc.index].values
    y_train = source_lfc.values
    y_test  = target_lfc.loc[source_lfc.index].values

    model = fit_model(model_type, X, y_train, seed)
    y_pred = model.predict(X)

    return evaluate_predictions(y_test, y_pred)
```

---

## Method 3 — Shared Signal Decomposition

Decompose each gene's LFC into a **shared component** (present in both
conditions) and a **condition-specific component**:

```python
def decompose_shared_signal(
    nutlin_lfc: np.ndarray,
    s63845_lfc: np.ndarray,
) -> dict:
    """
    Decompose LFC vectors into shared and condition-specific components.

    shared_lfc    = (nutlin_lfc + s63845_lfc) / 2
    nutlin_spec   = nutlin_lfc - shared_lfc
    s63845_spec   = s63845_lfc - shared_lfc

    Returns variance explained by each component.
    """
    shared_lfc  = (nutlin_lfc + s63845_lfc) / 2
    nutlin_spec = nutlin_lfc - shared_lfc
    s63845_spec = s63845_lfc - shared_lfc

    var_total   = np.var(nutlin_lfc) + np.var(s63845_lfc)
    var_shared  = 2 * np.var(shared_lfc)
    var_nutlin  = np.var(nutlin_spec)
    var_s63845  = np.var(s63845_spec)

    return {
        "var_shared_fraction":  var_shared / var_total,
        "var_nutlin_fraction":  var_nutlin / var_total,
        "var_s63845_fraction":  var_s63845 / var_total,
        "shared_lfc":           shared_lfc,
        "nutlin_specific_lfc":  nutlin_spec,
        "s63845_specific_lfc":  s63845_spec,
    }
```

Then test whether features predict the **shared component** better than
the condition-specific components:

```python
# Hypothesis: features predict shared signal better than condition-specific signal
r_shared   = pearsonr(features @ coef, shared_lfc)[0]
r_nutlin   = pearsonr(features @ coef, nutlin_spec)[0]
r_s63845   = pearsonr(features @ coef, s63845_spec)[0]
```

This decomposition is scientifically important: if features primarily capture
the shared (essential gene) signal, then cross-condition transfer will be
limited to predicting which genes are generally important, not which are
specifically sensitising to a given drug.

---

## Gene-Level Analysis: Condition-Specific vs Shared Hits

Classify each gene into one of four categories:

```python
def classify_gene_condition_specificity(
    nutlin_lfc: pd.Series,
    s63845_lfc: pd.Series,
    fdr_nutlin: pd.Series,
    fdr_s63845: pd.Series,
    fdr_threshold: float = 0.1,
) -> pd.Series:
    """
    Classify genes by condition specificity.

    Categories:
        'shared_hit'      : significant in both conditions, same direction
        'nutlin_specific' : significant in nutlin only
        's63845_specific' : significant in S63845 only
        'discordant'      : significant in both, opposite directions
        'non_hit'         : not significant in either
    """
    sig_n = fdr_nutlin  < fdr_threshold
    sig_s = fdr_s63845  < fdr_threshold
    same_dir = np.sign(nutlin_lfc) == np.sign(s63845_lfc)

    categories = pd.Series("non_hit", index=nutlin_lfc.index)
    categories[sig_n & sig_s & same_dir]   = "shared_hit"
    categories[sig_n & sig_s & ~same_dir]  = "discordant"
    categories[sig_n & ~sig_s]             = "nutlin_specific"
    categories[~sig_n & sig_s]             = "s63845_specific"
    return categories
```

Run pathway enrichment on each category using `gseapy` or `clusterProfiler`
to characterise what biological processes are shared vs condition-specific.

---

## Within-Condition Control

To establish the ceiling for cross-condition transfer, compute the
within-condition reproducibility by splitting the 6 replicates 50/50:

```python
def within_condition_reproducibility(
    counts: pd.DataFrame,
    condition_cols: list[str],   # 6 replicate columns for one condition
    n_splits: int = 20,
    seed: int = 0,
) -> list[float]:
    """
    Randomly split 6 replicates into two groups of 3.
    Compute LFC for each half independently.
    Return Pearson r between the two halves across n_splits.

    This is the within-condition ceiling: the maximum r achievable
    given the noise in the screen itself.
    """
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(n_splits):
        idx = rng.permutation(len(condition_cols))
        half1 = [condition_cols[i] for i in idx[:3]]
        half2 = [condition_cols[i] for i in idx[3:]]
        lfc1 = compute_lfc_per_gene(counts, half1, input_cols)
        lfc2 = compute_lfc_per_gene(counts, half2, input_cols)
        r, _ = pearsonr(lfc1["lfc"], lfc2["lfc"])
        correlations.append(r)
    return correlations
```

**Expected within-condition r:** 0.85–0.95 (high reproducibility with 3 replicates).
This sets the absolute ceiling for any transfer experiment.

---

## Output Schema

```
notebooks/Cas12a_EuMyc/results/aim3/
├── direct_cross_condition.csv          # Direct LFC regression, all 12 transfers
├── feature_cross_condition.csv         # Feature model, all 12 transfers × 20 seeds
├── shared_signal_decomposition.csv     # Variance decomposition per library
├── gene_condition_categories.parquet   # Per-gene classification (shared/specific/discordant)
├── within_condition_ceiling.csv        # Half-split reproducibility per condition
└── pathway_enrichment/
    ├── shared_hits_enrichment.csv
    ├── nutlin_specific_enrichment.csv
    └── s63845_specific_enrichment.csv
```

`direct_cross_condition.csv` columns:
```
transfer_id, library, source_condition, target_condition,
pearson_r, spearman_r, auroc_top5, auprc_top5, precision_at_100,
slope, intercept, n_genes
```

---

## Pathway Enrichment of Condition-Specific Hits

```python
import gseapy as gp

def run_enrichment(gene_list: list[str],
                   background: list[str],
                   gene_sets: str = "MSigDB_Hallmark_2020") -> pd.DataFrame:
    """
    Run over-representation analysis on a gene list.
    gene_list:  condition-specific hits (mouse symbols → human orthologs)
    background: all genes in the screen
    """
    # Convert mouse symbols to human orthologs first
    human_hits = [ortholog_map.get(g, g) for g in gene_list]
    human_bg   = [ortholog_map.get(g, g) for g in background]

    enr = gp.enrichr(
        gene_list=human_hits,
        gene_sets=gene_sets,
        background=human_bg,
        outdir=None,
        verbose=False,
    )
    return enr.results[enr.results["Adjusted P-value"] < 0.05]
```

Run for: shared hits, nutlin-specific hits, S63845-specific hits, discordant genes.

---

## Visualisations

### 1. Condition correlation scatter (nutlin LFC vs S63845 LFC)

```python
def plot_condition_scatter(nutlin_lfc, s63845_lfc, categories, r):
    """
    Scatter plot coloured by gene category.
    Highlight known hits: Trp53 (nutlin-specific), Bax (S63845-specific).
    """
    color_map = {
        "shared_hit":      "red",
        "nutlin_specific": "blue",
        "s63845_specific": "orange",
        "discordant":      "purple",
        "non_hit":         "lightgrey",
    }
    ...
```

### 2. Transfer performance heatmap

12 × 3 heatmap (12 transfers × Pearson r / AUROC / Precision@100),
showing which condition pairs transfer best.

### 3. Variance decomposition bar chart

Stacked bar: shared variance fraction vs nutlin-specific vs S63845-specific,
per library (Menuetto and Scherzo).

---

## Expected Results and Interpretation

| Transfer | Expected Pearson r | Interpretation |
|---|---|---|
| Within-condition (ceiling) | 0.85–0.95 | Screen reproducibility |
| Nutlin ↔ S63845 (direct) | 0.10–0.30 | Low — mechanistically distinct |
| Nutlin ↔ DMSO (direct) | 0.30–0.50 | Moderate — shared essential gene signal |
| Feature-based (nutlin → S63845) | 0.10–0.25 | Features capture shared signal only |
| Feature-based (nutlin → DMSO) | 0.25–0.45 | Better — essentiality features help |

Low nutlin ↔ S63845 transfer is the **expected and scientifically correct**
result. It confirms that the two conditions measure distinct biology. The
interesting finding would be if feature-based transfer is *higher* than
direct score regression — this would mean that gene features capture
generalizable properties that transcend the specific drug mechanism.

---

## Failure Modes and Mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| Nutlin ↔ S63845 r > 0.6 | Conditions are not as distinct as expected | Check that correct columns are used; verify known hits (Trp53 vs Bax) are in opposite directions |
| Feature model r ≈ 0 for all transfers | Features only capture essential genes | Add drug-specific features (e.g. BCL-2 family expression for S63845) |
| Pathway enrichment returns no hits | Too few condition-specific genes | Lower FDR threshold to 0.2; use ranked GSEA instead of ORA |
| Within-condition r < 0.7 | Screen quality issue | Check replicate correlation in preprocessing; flag low-quality replicates |
