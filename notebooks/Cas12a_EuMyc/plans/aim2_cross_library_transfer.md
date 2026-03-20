# Aim 2 — Cross-Library Transfer: Menuetto ↔ Scherzo

**File:** `notebooks/Cas12a_EuMyc/02_aim2_cross_library_transfer.py`
**Batch runner:** `notebooks/Cas12a_EuMyc/scripts/run_aim2.py`

---

## Scientific Question

Can gene-level LFC scores from the Menuetto library (2 crRNAs/gene, dual vector)
predict scores from the Scherzo library (4 crRNAs/gene, single vector), and vice
versa, given that both libraries were screened in **the same cell line, same passage,
same drug concentrations, and same sequencing run**?

This is the **upper bound experiment** for cross-dataset transfer. Because biology
is held constant, any prediction gap relative to Aim 1 (within-screen holdout) is
attributable solely to guide-level measurement noise — not to biological differences
between datasets. This quantifies the "measurement floor": the minimum irreducible
error in any cross-dataset transfer scenario.

### Why this matters

- If Aim 2 transfer is near-perfect (r ≈ Aim 1), guide architecture noise is
  negligible and cross-dataset transfer is limited only by feature quality.
- If Aim 2 transfer is substantially worse than Aim 1, guide-level noise is a
  major confounder and must be accounted for in Aim 3 (cross-condition) and
  any future cross-study transfer.

---

## Inputs

| Input | Path | Description |
|---|---|---|
| Menuetto gene scores | `data/bulk/menuetto_scherzo_2025/processed/menuetto_gene_scores.parquet` | LFC per gene, all conditions |
| Scherzo gene scores | `data/bulk/menuetto_scherzo_2025/processed/scherzo_gene_scores.parquet` | LFC per gene, all conditions |
| Feature matrix | `data/bulk/menuetto_scherzo_2025/processed/features_mouse_genes.parquet` | Per-gene feature vectors |

---

## Transfer Directions and Conditions

All 8 combinations are run:

| Transfer ID | Train source | Predict target | Condition |
|---|---|---|---|
| `M→S_nutlin` | Menuetto Nutlin | Scherzo Nutlin | Nutlin-3a |
| `M→S_s63845` | Menuetto S63845 | Scherzo S63845 | S63845 |
| `S→M_nutlin` | Scherzo Nutlin | Menuetto Nutlin | Nutlin-3a |
| `S→M_s63845` | Scherzo S63845 | Menuetto S63845 | S63845 |
| `M→S_dmso` | Menuetto DMSO | Scherzo DMSO | DMSO (essential gene dropout) |
| `S→M_dmso` | Scherzo DMSO | Menuetto DMSO | DMSO |
| `M→S_nutlin_feat` | Menuetto Nutlin (features only) | Scherzo Nutlin | Feature-based prediction |
| `S→M_nutlin_feat` | Scherzo Nutlin (features only) | Menuetto Nutlin | Feature-based prediction |

The first 6 use **direct score regression** (train on source LFC scores as labels,
predict target LFC scores). The last 2 use the **feature-based model** from Aim 1
(gene features → LFC), trained on source scores, evaluated on target scores.

---

## Split Design

Unlike Aim 1, Aim 2 uses **all genes** in both libraries (no holdout within a
library). The train/test split is across libraries:

```
Train set:  all ~21,743 genes from source library (Menuetto or Scherzo)
Test set:   all ~21,721 genes from target library (Scherzo or Menuetto)
Overlap:    ~21,700 genes present in both (used for evaluation)
```

```python
def get_gene_overlap(menuetto_genes: list[str],
                     scherzo_genes: list[str]) -> list[str]:
    """Return genes present in both libraries, sorted for reproducibility."""
    return sorted(set(menuetto_genes) & set(scherzo_genes))
```

**No random repeats needed** for the direct score regression (deterministic).
For the feature-based model, run 10 repeats with different random seeds to
estimate variance from model fitting (not from data splitting).

---

## Method 1 — Direct Score Regression

Train a regression model where:
- **X** = source library LFC scores (one value per gene)
- **y** = target library LFC scores (same genes)

This is a 1D → 1D regression that directly measures how well one library's
scores predict the other's, without any biological features.

```python
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import QuantileTransformer
import numpy as np

def direct_score_regression(
    source_lfc: np.ndarray,
    target_lfc: np.ndarray,
) -> dict:
    """
    Fit a simple linear regression of target ~ source LFC.
    Uses HuberRegressor for robustness to outlier guides.

    Returns metrics dict including slope, intercept, and all
    standard evaluation metrics.
    """
    X = source_lfc.reshape(-1, 1)
    y = target_lfc

    model = HuberRegressor(epsilon=1.35, max_iter=200)
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = evaluate_predictions(y, y_pred)
    metrics["slope"]     = float(model.coef_[0])
    metrics["intercept"] = float(model.intercept_)
    return metrics
```

**Why HuberRegressor:** A small fraction of genes will have very different
scores between libraries due to guide-specific off-target effects. Huber
regression down-weights these outliers, giving a more robust estimate of
the true library-to-library correlation.

Also compute the **raw Pearson/Spearman correlation** between source and
target LFC scores (no model fitting) as the simplest possible benchmark:

```python
from scipy.stats import pearsonr, spearmanr

def raw_correlation(source_lfc: np.ndarray, target_lfc: np.ndarray) -> dict:
    r_p, p_p = pearsonr(source_lfc, target_lfc)
    r_s, p_s = spearmanr(source_lfc, target_lfc)
    return {"pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s}
```

---

## Method 2 — Feature-Based Transfer

Use the same feature-based pipeline as Aim 1, but now:
- Train on source library gene scores (features → source LFC)
- Evaluate predictions against **target library** gene scores

This tests whether the feature-based model generalises across guide architectures.

```python
def feature_based_transfer(
    source_lfc: pd.Series,      # indexed by gene symbol
    target_lfc: pd.Series,      # indexed by gene symbol
    features: pd.DataFrame,     # indexed by gene symbol
    model_type: str = "ridge",
    seed: int = 0,
) -> dict:
    """
    Train on source_lfc using gene features, evaluate on target_lfc.
    Only genes present in both source and target are used for evaluation.
    """
    overlap_genes = source_lfc.index.intersection(target_lfc.index)

    X_train = features.loc[overlap_genes].values
    y_train = source_lfc.loc[overlap_genes].values
    y_test  = target_lfc.loc[overlap_genes].values

    model = fit_model(model_type, X_train, y_train, seed)
    y_pred = model.predict(X_train)  # same genes, different label source

    return evaluate_predictions(y_test, y_pred)
```

---

## Variance Analysis: Scherzo vs Menuetto Score Stability

A key secondary analysis: Scherzo has 4 crRNAs/gene vs Menuetto's 2.
More guides → lower variance in gene-level LFC estimates. Quantify this:

```python
def score_variance_analysis(
    menuetto_scores: pd.DataFrame,   # genes × replicates
    scherzo_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each gene, compute:
    - Inter-replicate variance in Menuetto (2 guides aggregated)
    - Inter-replicate variance in Scherzo (4 guides aggregated)
    - Variance ratio (Menuetto / Scherzo)
    - Whether the gene is a known essential (binary)

    Returns a per-gene DataFrame for downstream analysis.
    """
    m_var = menuetto_scores.var(axis=1).rename("var_menuetto")
    s_var = scherzo_scores.var(axis=1).rename("var_scherzo")
    ratio = (m_var / s_var).rename("var_ratio")

    return pd.concat([m_var, s_var, ratio], axis=1)
```

Use this variance ratio as a **covariate in the transfer regression**:
genes with high Menuetto variance (noisy 2-guide estimates) should be
harder to predict from Scherzo scores, and vice versa.

```python
# Stratified correlation: split genes into variance quartiles
for quartile in [0.25, 0.50, 0.75, 1.00]:
    mask = variance_ratio <= np.quantile(variance_ratio, quartile)
    r_q, _ = pearsonr(source_lfc[mask], target_lfc[mask])
    print(f"Variance quartile ≤{quartile:.0%}: Pearson r = {r_q:.3f}")
```

---

## Comparison to Aim 1 (Upper Bound Analysis)

The central result of Aim 2 is the **transfer gap**:

```
transfer_gap = aim1_pearson_r - aim2_pearson_r
```

Computed for each condition (nutlin, S63845, DMSO) and each direction (M→S, S→M).

```python
def compute_transfer_gap(
    aim1_results: pd.DataFrame,   # from Aim 1 summary CSVs
    aim2_results: pd.DataFrame,   # from Aim 2 results
    condition: str,
) -> dict:
    """
    Compare Aim 2 cross-library transfer to Aim 1 within-screen holdout.

    aim1_pearson_r: mean across 100 repeats for the same condition
    aim2_pearson_r: from direct score regression or feature-based transfer
    """
    aim1_r = aim1_results.query(f"arm_id == 'menuetto_{condition}'")["pearson_r"].mean()
    aim2_r = aim2_results.query(f"transfer_id == 'M→S_{condition}'")["pearson_r"].iloc[0]

    return {
        "condition": condition,
        "aim1_pearson_r": aim1_r,
        "aim2_pearson_r": aim2_r,
        "transfer_gap":   aim1_r - aim2_r,
        "gap_fraction":   (aim1_r - aim2_r) / aim1_r if aim1_r > 0 else None,
    }
```

**Interpretation guide:**

| Transfer gap | Interpretation |
|---|---|
| < 0.05 | Guide architecture noise is negligible; libraries are interchangeable |
| 0.05–0.15 | Moderate guide noise; cross-library transfer is feasible with calibration |
| > 0.15 | Substantial guide noise; library-specific effects dominate |

---

## Output Schema

```
notebooks/Cas12a_EuMyc/results/aim2/
├── direct_score_regression.csv     # Raw correlation + regression metrics, all 8 transfers
├── feature_based_transfer.csv      # Feature model metrics, all transfers × 10 seeds
├── variance_analysis.parquet       # Per-gene variance ratio (Menuetto vs Scherzo)
└── transfer_gap_summary.csv        # Aim1 vs Aim2 comparison table
```

`direct_score_regression.csv` columns:
```
transfer_id, condition, direction, pearson_r, spearman_r, auroc_top5,
auprc_top5, precision_at_100, slope, intercept, n_genes
```

---

## Visualisations

### 1. Scatter plot: Menuetto LFC vs Scherzo LFC (per condition)

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_library_scatter(menuetto_lfc, scherzo_lfc, condition, r):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(menuetto_lfc, scherzo_lfc, s=2, alpha=0.3, color="steelblue")
    ax.set_xlabel("Menuetto LFC")
    ax.set_ylabel("Scherzo LFC")
    ax.set_title(f"{condition}: Menuetto vs Scherzo\nPearson r = {r:.3f}")
    # Highlight known hits (Trp53, Bax, etc.)
    ...
```

### 2. Transfer gap bar chart

Bar chart comparing Aim 1 (within-screen) vs Aim 2 (cross-library) Pearson r
for each condition, with error bars from Aim 1 repeat distribution.

### 3. Variance ratio vs prediction error

Scatter plot: per-gene variance ratio (x) vs |Menuetto LFC − Scherzo LFC| (y).
Expect positive correlation — high-variance genes are harder to transfer.

---

## Batch Runner (`scripts/run_aim2.py`)

```python
"""
Run Aim 2 cross-library transfer for all conditions and directions.
Usage: python scripts/run_aim2.py
"""

TRANSFERS = [
    ("menuetto", "scherzo", "nutlin"),
    ("menuetto", "scherzo", "s63845"),
    ("menuetto", "scherzo", "dmso"),
    ("scherzo",  "menuetto", "nutlin"),
    ("scherzo",  "menuetto", "s63845"),
    ("scherzo",  "menuetto", "dmso"),
]

def main():
    results = []
    for source_lib, target_lib, condition in TRANSFERS:
        source_lfc = load_scores(f"{source_lib}_{condition}")
        target_lfc = load_scores(f"{target_lib}_{condition}")
        overlap    = get_gene_overlap(source_lfc.index, target_lfc.index)

        # Method 1: direct score regression
        m1 = direct_score_regression(
            source_lfc.loc[overlap].values,
            target_lfc.loc[overlap].values,
        )
        m1.update({"transfer_id": f"{source_lib[0].upper()}→{target_lib[0].upper()}_{condition}",
                   "method": "direct", "n_genes": len(overlap)})
        results.append(m1)

        # Method 2: feature-based transfer (10 seeds)
        features = load_features(overlap)
        for seed in range(10):
            m2 = feature_based_transfer(
                source_lfc.loc[overlap], target_lfc.loc[overlap],
                features, model_type="ridge", seed=seed,
            )
            m2.update({"transfer_id": f"{source_lib[0].upper()}→{target_lib[0].upper()}_{condition}",
                       "method": "feature_ridge", "seed": seed, "n_genes": len(overlap)})
            results.append(m2)

    pd.DataFrame(results).to_csv(
        "notebooks/Cas12a_EuMyc/results/aim2/all_transfers.csv", index=False
    )

if __name__ == "__main__":
    main()
```

---

## Expected Results and Interpretation

| Transfer | Expected Pearson r | Notes |
|---|---|---|
| Raw correlation (M vs S, nutlin) | 0.6–0.8 | High — same biology, different guides |
| Direct regression (M→S) | 0.65–0.85 | Slightly better than raw r |
| Feature-based (M→S) | 0.20–0.45 | Lower — features are imperfect proxies |
| Transfer gap (Aim1 − Aim2, feature) | 0.0–0.10 | Should be small if guide noise is low |

The raw Menuetto–Scherzo correlation is expected to be high (0.6–0.8) because
the biology is identical. The feature-based transfer will be lower because
features are imperfect proxies for the true biological signal. The key
comparison is whether the **feature-based transfer gap** (Aim 1 − Aim 2) is
small, which would indicate that guide architecture is not a major confounder.
