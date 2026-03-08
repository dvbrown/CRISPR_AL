# Design A Implementation Handoff: Within-Screen Holdout (Aim 1)

claude --resume 02c2743e-b8b9-455a-86d0-6ba90577f67a

This document is the single source of truth for a coding agent implementing
Design A. All decisions are locked. Do not re-litigate scope — implement
exactly what is specified here.

Read these reference files before starting:

- `notebooks/crispr_screen_transfer/codex_plan.md` — full scientific plan
- `notebooks/crispr_screen_transfer/splits.yaml` — split registry config
- `notebooks/crispr_screen_transfer/metrics.schema.json` — output schema
- `data/README.md` — dataset descriptions and file locations

---

## 1. Objective

Repeatedly sample 2,000 genes from the Chen 2019 genome-wide venetoclax CRISPR
screen, train a regression model on those genes using biological features, and
predict venetoclax sensitivity scores for the remaining ~17,000 holdout genes.
Repeat 25 times with different random splits. Report correlation and hit
recovery metrics with 95% confidence intervals across iterations.

---

## 2. Locked Decisions

| Decision | Value |
|----------|-------|
| Screen | Chen 2019, 16-day exposure: `BIOGRID-ORCS-SCREEN_1393-2.0.18.screen.tab.txt` |
| Train size | 2,000 genes per iteration |
| Repeats | 25 |
| Hit threshold | Top/bottom 5% of holdout genes by `score_norm` |
| K values | 50, 100, 200, 500 |
| Models | Ridge regression; Random Forest regressor |
| Cell line for features | MOLM-13 = ACH-000362 in both CCLE and DepMap |
| Gene ID canonical key | HGNC symbol (strip ` (ENTREZ_ID)` suffix from DepMap/CCLE columns) |
| Gene properties features | DROPPED |
| Druggability features | DROPPED |
| Output directory | `notebooks/crispr_screen_transfer/artifacts/` |

---

## 3. Input Data

All files are under `data/bulk/`. All CSVs are gzip-compressed.

### 3.1 Screen scores — Chen 2019 16-day

```
data/bulk/chen2019_venetoclax/BIOGRID-ORCS-SCREEN_1393-2.0.18.screen.tab.txt
```

Tab-separated, no compression. Columns:

| Column | Description |
|--------|-------------|
| `SCREEN_ID` | Always 1393 |
| `IDENTIFIER_ID` | Entrez Gene ID |
| `IDENTIFIER_TYPE` | Always `ENTREZ_GENE` |
| `OFFICIAL_SYMBOL` | HGNC gene symbol ← **use as gene key** |
| `ALIASES` | Alternative names (ignore) |
| `ORGANISM_ID` | Always 9606 |
| `ORGANISM_OFFICIAL` | Always `Homo sapiens` |
| `SCORE.1` | CRISPR Score (CS): **positive = resistance, negative = sensitivity** |
| `SCORE.2` | p-value for CS |

Load all rows. Drop any row where `OFFICIAL_SYMBOL` is empty or `SCORE.1` is
not a finite float. Expected row count: ~19,109.

**Score normalisation:** z-score `SCORE.1` across all genes in this screen to
produce `score_norm`. Fit z-score parameters on the full gene set (all 19,109),
then apply. Store `score_norm` alongside `OFFICIAL_SYMBOL`.

**Hit labels** (applied after z-scoring, using the full-screen distribution):
- `is_hit_sensitizer = True` if `score_norm` < −1.645 (bottom 5%)
- `is_hit_resistor = True` if `score_norm` > 1.645 (top 5%)

### 3.2 DepMap CRISPR Gene Effect — co-essentiality features

```
data/bulk/depmap_crispr_gene_effect/CRISPRGeneEffect.csv.gz
```

Shape: 1,186 cell lines × 18,435 genes. Row index: DepMap model ID. Column
header format: `SYMBOL (ENTREZ_ID)` e.g. `BCL2 (596)`.

Extract the single row for MOLM-13: **row index `ACH-000362`**.

Parse column names: strip ` (ENTREZ_ID)` suffix → HGNC symbol. This gives a
Series of length 18,435: `{gene_symbol: chronos_score}`.

This row is the **MOLM-13 Chronos profile**. It is NOT used directly as a
feature. Instead it is used to compute co-essentiality (see Section 4.2).

Also load the **full matrix** (all 1,186 rows) to compute pairwise correlations
for co-essentiality features. Load lazily or in chunks if memory is a concern
(full matrix is ~186 MB compressed; ~1.5 GB uncompressed as float64 — consider
float32 or chunked pandas).

### 3.3 CCLE Expression — expression features

```
data/bulk/ccle_expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz
```

Shape: 1,684 cell lines × 19,205 genes. Row index: DepMap model ID. Column
header format: `SYMBOL (ENTREZ_ID)`.

Extract the single row for MOLM-13: **row index `ACH-000362`**.

Parse column names: strip ` (ENTREZ_ID)` suffix → HGNC symbol.

Result: a Series of length 19,205: `{gene_symbol: log2(TPM+1)}`.

This is the **MOLM-13 expression profile**. For each gene in the screen,
its expression value in MOLM-13 is a scalar feature.

### 3.4 Pathway annotations

```
data/bulk/pathway_annotations/NCBI2Reactome_PE_Pathway.txt.gz
data/bulk/pathway_annotations/goa_human.gaf.gz
data/bulk/pathway_annotations/h.all.v2024.1.Hs.symbols.gmt.gz
data/bulk/pathway_annotations/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt.gz
```

See Section 4.3 for parsing and feature construction details.

---

## 4. Feature Engineering

Produce one row per gene, one column per feature. Final matrix:
`gene_features.parquet` — shape (~19,109 genes) × (N features). Index:
`gene_symbol` (HGNC). All features are numeric floats.

### 4.1 Expression feature (1 feature)

```
feature: molm13_log_tpm
value:   CCLE log2(TPM+1) for ACH-000362, for each gene symbol
missing: genes absent from CCLE → fill with 0.0 (unexpressed prior)
```

### 4.2 Co-essentiality features (2 features)

For each screen gene `g`, compute its Pearson correlation with every other gene
across the 1,186 DepMap cell lines. This is the co-essentiality profile.

Summarise into two scalar features:

```
feature: coessential_mean_r_top50
value:   mean Pearson r of gene g with the 50 most correlated other genes
         (i.e. the mean of the top-50 off-diagonal correlations in the
         full gene×gene correlation matrix row for g)

feature: coessential_molm13_chronos
value:   raw Chronos score for gene g in MOLM-13 (ACH-000362) from
         CRISPRGeneEffect — a direct essentiality prior for this cell line
```

Missing: genes absent from DepMap (~8% of screen genes) → fill both with 0.0.

**Implementation note:** Computing the full 18,435 × 18,435 correlation matrix
requires ~2.7 GB float64. Compute in float32 (saves ~50%). If memory is still
constrained, compute row-by-row correlations only for the screen genes present
in DepMap. Do not precompute or cache the full matrix to disk unless explicitly
requested.

### 4.3 Pathway membership features

Build a binary gene × pathway indicator matrix from all four annotation sources,
then reduce to summary scalar features per gene.

#### Reactome (`NCBI2Reactome_PE_Pathway.txt.gz`)

Tab-separated, no header. Relevant columns (0-indexed):

| Index | Content |
|-------|---------|
| 0 | Entrez Gene ID |
| 3 | Reactome Pathway stable ID (e.g. `R-HSA-114608`) |
| 5 | Pathway name |
| 7 | Species — **filter to `Homo sapiens` only** |

Join to screen genes via Entrez ID (use `IDENTIFIER_ID` from the screen file).
Produces a gene → set-of-pathway-IDs mapping.

#### GOA (`goa_human.gaf.gz`)

GAF 2.0/2.2 format. Skip lines starting with `!`. Relevant columns (0-indexed):

| Index | Content |
|-------|---------|
| 2 | Gene symbol (HGNC) ← join key |
| 4 | GO term ID (e.g. `GO:0006915`) |
| 8 | GO aspect: `P` (biological process), `F` (molecular function), `C` (cellular component) |
| 6 | Evidence code — exclude `IEA` (inferred from electronic annotation) to use curated only |

Produces a gene → set-of-GO-term-IDs mapping (filtered to non-IEA).

#### GMT files (Hallmarks + KEGG)

Format: one gene set per line. Tab-separated:
- Column 0: gene set name
- Column 1: description (ignore)
- Columns 2+: HGNC gene symbols

Parse both GMT files. Produces a gene → set-of-geneset-names mapping.

#### Scalar pathway features (derive from the above)

Rather than using the full binary indicator matrix (which is high-dimensional),
compute these scalar summaries per gene:

```
feature: n_reactome_pathways    — count of Reactome pathway memberships
feature: n_go_bp_terms          — count of non-IEA GO Biological Process terms
feature: n_go_mf_terms          — count of non-IEA GO Molecular Function terms
feature: in_hallmark_apoptosis  — 1 if member of HALLMARK_APOPTOSIS, else 0
feature: in_hallmark_oxidative_phosphorylation — 1 if member of
                                    HALLMARK_OXIDATIVE_PHOSPHORYLATION, else 0
feature: n_kegg_pathways        — count of KEGG pathway memberships
```

Missing (gene not in any annotation source) → 0 for all pathway features.

**Total feature count: 1 + 2 + 6 = 9 features per gene.**

Save `gene_features.parquet` with index `gene_symbol` and these 9 columns.
Also save `screen_scores.parquet` with columns `gene_symbol`, `score_raw`
(original `SCORE.1`), `score_norm`, `is_hit_sensitizer`, `is_hit_resistor`.

---

## 5. Split Generation

Generate 25 splits following `splits.yaml` generator `aim1_random_gene_holdout`:

```python
import hashlib, json
import numpy as np

SEED_START = 11001
N_REPEATS = 25
TRAIN_SIZE = 2000
SCREEN_ID = "chen2019_1393"

all_genes = list(screen_scores["gene_symbol"])  # ~19,109 genes

splits = []
for i in range(N_REPEATS):
    seed = SEED_START + i
    rng = np.random.default_rng(seed)
    train_genes = set(rng.choice(all_genes, size=TRAIN_SIZE, replace=False))
    test_genes = [g for g in all_genes if g not in train_genes]

    split_payload = json.dumps({
        "generator_id": "aim1_random_gene_holdout",
        "screen_id": SCREEN_ID,
        "seed": seed,
        "train_genes": sorted(train_genes),
    }, sort_keys=True).encode()
    split_hash = hashlib.sha256(split_payload).hexdigest()[:16]

    splits.append({
        "split_id": f"aim1_random_{SCREEN_ID}_r{i+1:03d}",
        "generator_id": "aim1_random_gene_holdout",
        "family": "random_gene_holdout",
        "aim": "aim1_venetoclax",
        "metrics_profile": "aim1_transfer",
        "seed": seed,
        "repeat_index": i + 1,
        "train_screen_id": SCREEN_ID,
        "test_screen_id": SCREEN_ID,
        "split_hash": split_hash,
        "train_genes": sorted(train_genes),
        "test_genes": sorted(test_genes),
    })
```

Save the manifest (all fields except `train_genes`/`test_genes`) to
`notebooks/crispr_screen_transfer/artifacts/split_manifest.csv`.

Save full splits (including gene lists) to
`notebooks/crispr_screen_transfer/artifacts/splits/aim1_random_{SCREEN_ID}_r{i:03d}.json`
for auditability.

---

## 6. Model Training and Prediction

For each split:

### 6.1 Prepare train/test arrays

```python
X_train = gene_features.loc[train_genes].values   # (2000, 9)
y_train = screen_scores.set_index("gene_symbol").loc[train_genes, "score_norm"].values

X_test  = gene_features.loc[test_genes].values    # (~17109, 9)
y_test  = screen_scores.set_index("gene_symbol").loc[test_genes, "score_norm"].values
```

Leakage check: assert `set(train_genes) & set(test_genes) == set()`.

### 6.2 Feature scaling

Fit `StandardScaler` on `X_train` only. Transform both `X_train` and `X_test`.
This is the "train-only normalization" required by leakage controls.

### 6.3 Models

**Ridge regression**
```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

**Random Forest**
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                               random_state=seed, n_jobs=-1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

Run both models on every split. Store predictions separately.

---

## 7. Metrics

Compute the following for each (split × model) combination. All metrics are
on the **test set only**.

### 7.1 Regression

```python
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pearson  = pearsonr(y_test, y_pred).statistic
spearman = spearmanr(y_test, y_pred).statistic
r2       = r2_score(y_test, y_pred)
rmse     = mean_squared_error(y_test, y_pred) ** 0.5
mae      = mean_absolute_error(y_test, y_pred)
```

### 7.2 Ranking — Precision@K and Recall@K

Hit sets are defined on the **full screen** (all 19,109 genes) using the
z-score thresholds from Section 3.1. Within the test set, a gene is a
"true positive" if it was labelled a hit in the full-screen distribution.

```python
K_VALUES = [50, 100, 200, 500]

# For sensitizers (bottom 5% = most negative score_norm)
true_sensitizers = set(test genes where is_hit_sensitizer == True)
pred_rank_sensitizer = test genes sorted by y_pred ascending  # most negative first

for K in K_VALUES:
    top_k_pred = set(pred_rank_sensitizer[:K])
    precision_at_k_sensitizer = len(top_k_pred & true_sensitizers) / K
    recall_at_k_sensitizer    = len(top_k_pred & true_sensitizers) / max(len(true_sensitizers), 1)

# Repeat symmetrically for resistors (top 5%, sort y_pred descending)
```

### 7.3 Classification — AUROC and AUPRC

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Sensitizers: label = 1 if is_hit_sensitizer, score = -y_pred (more negative = more likely)
auroc_sensitizer = roc_auc_score(y_test_sensitizer_labels, -y_pred)
auprc_sensitizer = average_precision_score(y_test_sensitizer_labels, -y_pred)

# Resistors: label = 1 if is_hit_resistor, score = +y_pred
auroc_resistor = roc_auc_score(y_test_resistor_labels, y_pred)
auprc_resistor = average_precision_score(y_test_resistor_labels, y_pred)
```

### 7.4 Naive baseline

Compute once (not per split): predict `y_pred = 0.0` for all test genes
(i.e. predict the global mean). Report the same metrics. This is the floor
against which model performance is compared.

---

## 8. Output Artifacts

### 8.1 Per-split metrics JSON

For each (split × model), write one JSON file validated against
`metrics.schema.json`:

```
notebooks/crispr_screen_transfer/artifacts/metrics/
    aim1_random_chen2019_1393_r001_ridge.json
    aim1_random_chen2019_1393_r001_rf.json
    ...
    aim1_random_chen2019_1393_r100_rf.json
```

JSON structure must match `metrics.schema.json` exactly. Required fields:

```json
{
  "schema_version": "1.0.0",
  "run_id": "<split_id>_<model_name>",
  "timestamp_utc": "<ISO 8601>",
  "code_commit": "<git rev-parse --short HEAD>",
  "split": {
    "split_id": "aim1_random_chen2019_1393_r001",
    "generator_id": "aim1_random_gene_holdout",
    "family": "random_gene_holdout",
    "aim": "aim1_venetoclax",
    "metrics_profile": "aim1_transfer",
    "seed": 11001,
    "repeat_index": 1,
    "train_screen_id": "chen2019_1393",
    "test_screen_id": "chen2019_1393",
    "split_hash": "<16-char hex>"
  },
  "data_counts": {
    "train_row_count": 2000,
    "test_row_count": <N>,
    "n_unique_train_genes": 2000,
    "n_unique_test_genes": <N>,
    "n_overlap_genes_train_test": 0
  },
  "leakage_checks": {
    "disjoint_gene_label_rows": true,
    "normalization_fit_on_train_only": true,
    "split_hash_logged": true
  },
  "metrics": {
    "regression": {
      "pearson": <float>,
      "spearman": <float>,
      "r2": <float>,
      "rmse": <float>,
      "mae": <float>
    },
    "ranking": {
      "k_metrics": [
        {"k": 50,  "precision_at_k": <float>, "recall_at_k": <float>},
        {"k": 100, "precision_at_k": <float>, "recall_at_k": <float>},
        {"k": 200, "precision_at_k": <float>, "recall_at_k": <float>},
        {"k": 500, "precision_at_k": <float>, "recall_at_k": <float>}
      ]
    },
    "classification": {
      "labels": [
        {"label": "sensitizer", "auroc": <float>, "auprc": <float>, "positive_rate": <float>},
        {"label": "resistor",   "auroc": <float>, "auprc": <float>, "positive_rate": <float>}
      ]
    }
  }
}
```

### 8.2 Aggregated results table

After all 25 splits, aggregate per model:

```
notebooks/crispr_screen_transfer/artifacts/design_a_results_ridge.csv
notebooks/crispr_screen_transfer/artifacts/design_a_results_rf.csv
notebooks/crispr_screen_transfer/artifacts/design_a_results_baseline.csv
```

Each row = one split. Columns: `split_id`, `seed`, `repeat_index`, then one
column per metric. Compute and append mean/std/CI rows at the bottom.

CIs: bootstrap 1,000× across the 100 per-split point estimates, 95% (α=0.05),
as specified in `splits.yaml` `confidence_interval` block.

### 8.3 Intermediate artefacts (save to avoid recomputation)

```
notebooks/crispr_screen_transfer/artifacts/
    gene_features.parquet        — gene × 9 features
    screen_scores.parquet        — gene_symbol, score_raw, score_norm, hit labels
    splits/                      — per-split JSON gene lists
    split_manifest.csv           — split metadata (no gene lists)
```

---

## 9. Implementation Order

1. Load and normalise Chen 2019 screen scores → `screen_scores.parquet`
2. Build expression feature (CCLE, ACH-000362)
3. Build co-essentiality features (DepMap, ACH-000362 + full matrix)
4. Build pathway features (Reactome, GOA, Hallmarks, KEGG)
5. Assemble and save `gene_features.parquet`
6. Generate and save all 100 splits + `split_manifest.csv`
7. Run Ridge on all 100 splits; write per-split JSON; write aggregated CSV
8. Run Random Forest on all 100 splits; repeat
9. Compute naive baseline; write CSV
10. Validate all metric JSONs against `metrics.schema.json`

---

## 10. Environment

Python 3.10+. Required packages:

```
pandas >= 2.0
numpy >= 1.26
scipy >= 1.12
scikit-learn >= 1.4
pyarrow          # for parquet I/O
jsonschema       # for metrics.schema.json validation
```

No GPU required. The co-essentiality matrix step is the most compute-intensive;
expect 2–10 minutes depending on available RAM. Use `n_jobs=-1` for Random
Forest.

---

## 11. What This Handoff Does NOT Cover

- Design B (cross-screen transfer between Chen and Sharon) — separate handoff
- Aim 2 (in vitro vs in vivo) — separate handoff
- Visualisation / figures — separate step after results are validated
- Hyperparameter search beyond the alphas specified for Ridge

---

## 12. Acceptance Criteria

The implementation is complete when:

1. `gene_features.parquet` exists with 9 feature columns and ~19,109 gene rows
2. `screen_scores.parquet` exists with `score_norm`, `is_hit_sensitizer`,
   `is_hit_resistor` columns
3. `split_manifest.csv` has exactly 100 rows with unique `split_hash` values
4. 200 metric JSON files exist (100 splits × 2 models) plus 1 baseline file
5. All metric JSONs pass `jsonschema.validate` against `metrics.schema.json`
6. All `leakage_checks` fields are `true` in every JSON
7. `n_overlap_genes_train_test` is 0 in every JSON
8. Aggregated CSVs contain mean Pearson r, mean AUROC, and 95% CI for both
   models and the naive baseline
