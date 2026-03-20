# Aim 1 — Within-Screen Holdout: Focused → Genome-Wide Prediction

**File:** `notebooks/Cas12a_EuMyc/01_aim1_within_screen_holdout.py`
**Batch runner:** `notebooks/Cas12a_EuMyc/scripts/run_aim1.py`

---

## Scientific Question

Can a model trained on a randomly sampled subset of ~2,000 genes predict the
LFC scores of the remaining ~19,700 genes in the same screen? And does the
**choice of which 2,000 genes to include** — random vs pathway-stratified vs
biologically-informed — materially change prediction quality?

This aim has two nested questions:

1. **Prediction quality:** How well can gene features extrapolate from a focused
   training set to genome-wide scores? (baseline transferability)
2. **Training set design:** Does smarter gene selection improve hit recovery and
   generalisation to unseen pathways, for the same sequencing budget?

The random-sampling baseline sets the floor; the selection strategy experiments
test whether a researcher can do better by being deliberate about which genes
to include in a focused library.

---

## Inputs

| Input | Path | Description |
|---|---|---|
| Menuetto gene scores | `data/bulk/menuetto_scherzo_2025/processed/menuetto_gene_scores.parquet` | LFC + FDR per gene, all conditions |
| Scherzo gene scores | `data/bulk/menuetto_scherzo_2025/processed/scherzo_gene_scores.parquet` | LFC per gene, all conditions |
| iMDF gene scores | `data/bulk/menuetto_scherzo_2025/processed/imdf_gene_scores_by_timepoint.parquet` | LFC per timepoint vs T0 |
| Feature matrix | `data/bulk/menuetto_scherzo_2025/processed/features_mouse_genes.parquet` | Per-gene feature vectors |
| Reactome gene sets | `data/bulk/pathway_annotations/NCBI2Reactome_PE_Pathway.txt.gz` | Gene → Reactome pathway membership |
| MSigDB Hallmarks | `data/bulk/pathway_annotations/h.all.v2024.1.Hs.symbols.gmt.gz` | Gene → Hallmark gene set membership |
| Ortholog map | `data/bulk/menuetto_scherzo_2025/processed/mouse_human_orthologs.parquet` | Mouse → human symbol mapping |

---

## Screen Arms Tested

Run Aim 1 independently for each of the following 5 conditions:

| Arm ID | Library | Condition | Label column |
|---|---|---|---|
| `menuetto_nutlin` | Menuetto | Nutlin-3a vs Input | `lfc_nutlin` |
| `menuetto_s63845` | Menuetto | S63845 vs Input | `lfc_s63845` |
| `scherzo_nutlin` | Scherzo | Nutlin-3a vs Input | `lfc_nutlin` |
| `scherzo_s63845` | Scherzo | S63845 vs Input | `lfc_s63845` |
| `imdf_t3` | Menuetto (iMDF) | T3 vs T0 | `lfc_t3` |

---

## Selection Strategies

Six strategies are implemented, all using the same **n_train = 2,000 gene budget**.
They are defined in `src/crispr_al/splits.py` and registered in `splits.yaml`.

### Strategy 0 — Random (baseline)

Pure random sampling. No biological knowledge used. Run 100 repeats.

```python
def split_random(
    gene_list: list[str],
    n_train: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Randomly partition genes into training and holdout sets.
    Returns (train_genes, holdout_genes).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(gene_list))
    return [gene_list[i] for i in idx[:n_train]], \
           [gene_list[i] for i in idx[n_train:]]
```

---

### Strategy 1 — Reactome-Stratified

Ensure every Reactome pathway (above a minimum size threshold) contributes at
least one gene to the training set. Remaining budget filled randomly.

**Rationale:** A model trained on genes from only a few pathways will have
poor feature coverage for unseen pathways. Stratification forces the training
set to span the full biological space represented in the feature matrix.

```python
def split_reactome_stratified(
    gene_list: list[str],
    reactome_membership: dict[str, list[str]],  # pathway_id → [gene_symbols]
    n_train: int,
    seed: int,
    min_pathway_size: int = 10,
    max_pathway_size: int = 500,
) -> tuple[list[str], list[str]]:
    """
    Step 1: Filter pathways to those with min_pathway_size ≤ size ≤ max_pathway_size
            and at least 1 member in gene_list.
    Step 2: For each qualifying pathway, sample 1 gene (without replacement
            across pathways — each gene can only be selected once).
    Step 3: Fill remaining budget (n_train - n_pathway_seeds) randomly
            from unselected genes.

    Parameters
    ----------
    min_pathway_size : exclude tiny pathways (noisy, over-specific)
    max_pathway_size : exclude huge pathways (e.g. "metabolism") that would
                       dominate the seed set

    Returns
    -------
    (train_genes, holdout_genes)
    """
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)

    # Filter pathways
    qualifying = {
        pid: [g for g in genes if g in gene_set]
        for pid, genes in reactome_membership.items()
        if min_pathway_size <= len([g for g in genes if g in gene_set]) <= max_pathway_size
    }

    # Greedy 1-per-pathway sampling (shuffle pathway order for reproducibility)
    pathway_ids = sorted(qualifying.keys())
    rng.shuffle(pathway_ids)

    selected = set()
    for pid in pathway_ids:
        candidates = [g for g in qualifying[pid] if g not in selected]
        if candidates:
            chosen = rng.choice(candidates)
            selected.add(chosen)
        if len(selected) >= n_train:
            break

    # Fill remaining budget randomly
    remaining = [g for g in gene_list if g not in selected]
    n_fill = n_train - len(selected)
    if n_fill > 0:
        fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
        selected.update(fill)

    train_genes   = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes
```

**Expected n_pathway_seeds:** Reactome has ~2,000 human pathways, but after
filtering to 10–500 genes and requiring mouse ortholog coverage, expect
~800–1,200 qualifying pathways. The seed set will therefore be ~800–1,200
genes, with the remaining ~800–1,200 filled randomly.

**Key property:** Every qualifying Reactome pathway has ≥1 representative in
the training set. The model sees at least one example from each pathway,
which should improve generalisation to holdout genes in those pathways.

---

### Strategy 2 — Hallmark-Stratified

Same logic as Strategy 1 but using the 50 MSigDB Hallmark gene sets.
Much coarser (50 pathways → 50 seed genes), so the seed set is small
and the random fill dominates.

```python
def split_hallmark_stratified(
    gene_list: list[str],
    hallmark_membership: dict[str, list[str]],  # hallmark_name → [gene_symbols]
    n_train: int,
    seed: int,
    n_per_hallmark: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Sample n_per_hallmark genes from each of the 50 Hallmark gene sets.
    Total seed genes: up to 50 × n_per_hallmark = 250 genes.
    Fill remaining budget randomly.

    n_per_hallmark = 5 gives 250 seed genes out of 2,000 total (12.5%).
    This is intentionally modest — Hallmarks are broad and overlapping.
    """
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)
    selected = set()

    hallmark_names = sorted(hallmark_membership.keys())
    rng.shuffle(hallmark_names)

    for hname in hallmark_names:
        candidates = [g for g in hallmark_membership[hname]
                      if g in gene_set and g not in selected]
        n_sample = min(n_per_hallmark, len(candidates))
        if n_sample > 0:
            chosen = rng.choice(candidates, size=n_sample, replace=False)
            selected.update(chosen)

    # Fill remaining budget randomly
    remaining = [g for g in gene_list if g not in selected]
    n_fill = n_train - len(selected)
    if n_fill > 0:
        fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
        selected.update(fill)

    train_genes   = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes
```

---

### Strategy 3 — Apoptosis/p53 Seeded (nutlin-3a informed)

Fix a core set of genes from the **apoptosis** and **p53 pathway** Hallmark
gene sets as mandatory training genes, then fill the remaining budget randomly.

**Rationale:** A researcher designing a focused library for a nutlin-3a screen
would naturally include known p53 pathway and apoptosis genes. This strategy
tests whether this domain knowledge translates into better prediction of the
full genome-wide screen.

```python
# Hallmark gene sets used as seeds for nutlin-3a / p53 screens
APOPTOSIS_P53_HALLMARKS = [
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_DNA_REPAIR",          # p53 activates DNA repair genes
    "HALLMARK_G2M_CHECKPOINT",      # p53-mediated cell cycle arrest
]

def split_apoptosis_p53_seeded(
    gene_list: list[str],
    hallmark_membership: dict[str, list[str]],
    n_train: int,
    seed: int,
    seed_hallmarks: list[str] = APOPTOSIS_P53_HALLMARKS,
    max_seed_fraction: float = 0.20,  # cap seed genes at 20% of budget = 400 genes
) -> tuple[list[str], list[str]]:
    """
    Include ALL genes from the specified Hallmark gene sets as mandatory
    training genes (up to max_seed_fraction × n_train).
    Fill remaining budget randomly.

    max_seed_fraction caps the seed set so it doesn't dominate the training
    set — we want to test whether seeding *helps*, not whether a pure
    apoptosis library predicts apoptosis genes well (trivially true).

    Returns (train_genes, holdout_genes).
    """
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)
    max_seeds = int(n_train * max_seed_fraction)

    # Collect all genes from seed hallmarks
    seed_genes = set()
    for hname in seed_hallmarks:
        seed_genes.update(g for g in hallmark_membership.get(hname, [])
                          if g in gene_set)

    # Cap seed set if it exceeds max_seed_fraction
    if len(seed_genes) > max_seeds:
        seed_genes = set(rng.choice(sorted(seed_genes), size=max_seeds, replace=False))

    # Fill remaining budget randomly
    remaining = [g for g in gene_list if g not in seed_genes]
    n_fill = n_train - len(seed_genes)
    fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
    selected = seed_genes | set(fill)

    train_genes   = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes
```

**Expected seed set size:** HALLMARK_APOPTOSIS (~160 genes) + HALLMARK_P53_PATHWAY
(~200 genes) + HALLMARK_DNA_REPAIR (~150 genes) + HALLMARK_G2M_CHECKPOINT (~200 genes)
= ~500–600 unique genes after deduplication. Capped at 400 (20% of 2,000).

**Important:** This strategy is **condition-specific** — it is only meaningful
for the nutlin-3a arms. For S63845 (MCL-1 inhibitor), use Strategy 4 instead.
For iMDF dropout, use Strategy 1 (Reactome-stratified) as the informed strategy.

---

### Strategy 4 — BCL-2/Apoptosis Seeded (S63845 informed)

Analogous to Strategy 3 but seeded with BCL-2 family and intrinsic apoptosis
pathway genes, relevant for the MCL-1 inhibitor S63845.

```python
BCL2_APOPTOSIS_HALLMARKS = [
    "HALLMARK_APOPTOSIS",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",  # survival signalling
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",  # anti-apoptotic survival
    "HALLMARK_MYC_TARGETS_V1",           # Eµ-MYC model context
]

def split_bcl2_seeded(
    gene_list: list[str],
    hallmark_membership: dict[str, list[str]],
    n_train: int,
    seed: int,
    seed_hallmarks: list[str] = BCL2_APOPTOSIS_HALLMARKS,
    max_seed_fraction: float = 0.20,
) -> tuple[list[str], list[str]]:
    """Same logic as split_apoptosis_p53_seeded, different seed hallmarks."""
    return split_apoptosis_p53_seeded(
        gene_list, hallmark_membership, n_train, seed,
        seed_hallmarks=seed_hallmarks,
        max_seed_fraction=max_seed_fraction,
    )
```

---

### Strategy 5 — Reactome-Stratified + Apoptosis Oversampled

Combines Strategies 1 and 3: pathway-stratified sampling with the apoptosis/p53
pathway oversampled (sample 5 genes per apoptosis pathway instead of 1).

```python
def split_reactome_apoptosis_oversampled(
    gene_list: list[str],
    reactome_membership: dict[str, list[str]],
    n_train: int,
    seed: int,
    apoptosis_pathway_ids: list[str],   # Reactome IDs for apoptosis pathways
    n_per_apoptosis: int = 5,
    n_per_other: int = 1,
    min_pathway_size: int = 10,
    max_pathway_size: int = 500,
) -> tuple[list[str], list[str]]:
    """
    Reactome-stratified sampling with oversampling of apoptosis pathways.

    For apoptosis/p53 Reactome pathways: sample n_per_apoptosis genes each.
    For all other qualifying pathways: sample n_per_other gene each.
    Fill remaining budget randomly.

    apoptosis_pathway_ids: list of Reactome pathway IDs corresponding to
        apoptosis, p53 regulation, intrinsic apoptosis, etc.
        (see REACTOME_APOPTOSIS_IDS constant below)
    """
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)
    selected = set()

    for pid, genes in reactome_membership.items():
        members = [g for g in genes if g in gene_set and g not in selected]
        pathway_size = len([g for g in genes if g in gene_set])
        if not (min_pathway_size <= pathway_size <= max_pathway_size):
            continue

        n_sample = n_per_apoptosis if pid in apoptosis_pathway_ids else n_per_other
        n_sample = min(n_sample, len(members))
        if n_sample > 0:
            chosen = rng.choice(members, size=n_sample, replace=False)
            selected.update(chosen)

        if len(selected) >= n_train:
            break

    # Fill remaining budget randomly
    remaining = [g for g in gene_list if g not in selected]
    n_fill = n_train - len(selected)
    if n_fill > 0:
        fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
        selected.update(fill)

    train_genes   = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes


# Reactome pathway IDs for apoptosis / p53 regulation
# (human IDs; mapped to mouse orthologs before use)
REACTOME_APOPTOSIS_IDS = [
    "R-HSA-109581",   # Apoptosis
    "R-HSA-5633007",  # Regulation of TP53 Expression and Degradation
    "R-HSA-3700989",  # Transcriptional Regulation by Small Molecules (p53)
    "R-HSA-6796648",  # TP53 Regulates Transcription of Genes Involved in G1 Cell Cycle Arrest
    "R-HSA-111452",   # Activation of BH3-only proteins
    "R-HSA-114452",   # Activation of BIM and translocation to mitochondria
    "R-HSA-139915",   # Activation of NOXA and translocation to mitochondria
    "R-HSA-5633007",  # BCL-2 family members and regulation of apoptosis
]
```

---

## Strategy Registry (`splits.yaml`)

```yaml
aim1:
  n_train: 2000
  n_repeats: 100
  strategies:

    random:
      function: split_random
      n_repeats: 100
      description: "Pure random sampling — baseline"

    reactome_stratified:
      function: split_reactome_stratified
      n_repeats: 50
      params:
        min_pathway_size: 10
        max_pathway_size: 500
      description: "1 gene per qualifying Reactome pathway, random fill"

    hallmark_stratified:
      function: split_hallmark_stratified
      n_repeats: 50
      params:
        n_per_hallmark: 5
      description: "5 genes per Hallmark gene set, random fill"

    apoptosis_p53_seeded:
      function: split_apoptosis_p53_seeded
      n_repeats: 50
      arms: [menuetto_nutlin, scherzo_nutlin]   # only meaningful for nutlin arms
      params:
        seed_hallmarks: [HALLMARK_APOPTOSIS, HALLMARK_P53_PATHWAY,
                         HALLMARK_DNA_REPAIR, HALLMARK_G2M_CHECKPOINT]
        max_seed_fraction: 0.20
      description: "Apoptosis/p53 Hallmark genes seeded, random fill"

    bcl2_seeded:
      function: split_bcl2_seeded
      n_repeats: 50
      arms: [menuetto_s63845, scherzo_s63845]   # only meaningful for S63845 arms
      params:
        seed_hallmarks: [HALLMARK_APOPTOSIS, HALLMARK_TNFA_SIGNALING_VIA_NFKB,
                         HALLMARK_PI3K_AKT_MTOR_SIGNALING, HALLMARK_MYC_TARGETS_V1]
        max_seed_fraction: 0.20
      description: "BCL-2/survival Hallmark genes seeded, random fill"

    reactome_apoptosis_oversampled:
      function: split_reactome_apoptosis_oversampled
      n_repeats: 50
      arms: [menuetto_nutlin, menuetto_s63845, scherzo_nutlin, scherzo_s63845]
      params:
        n_per_apoptosis: 5
        n_per_other: 1
        min_pathway_size: 10
        max_pathway_size: 500
      description: "Reactome-stratified with 5x oversampling of apoptosis pathways"
```

---

## Feature Matrix

### Construction (from `00_download_and_preprocess.py`)

Each gene is represented as a fixed-length feature vector. All features are
derived from the gene identity alone (not from the screen data), so there is
no data leakage between training and holdout sets.

```python
FEATURE_GROUPS = {
    "depmap_chronos": {
        "description": "Pan-cancer mean Chronos score across all DepMap cell lines",
        "source": "data/bulk/depmap_crispr_gene_effect/CRISPRGeneEffect.csv.gz",
        "n_features": 1,
        "dtype": float,
        "missing_strategy": "median_impute",
    },
    "ccle_expression": {
        "description": "log2(TPM+1) expression in closest available cell line context",
        "source": "data/bulk/ccle_expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz",
        "n_features": 1,
        "dtype": float,
        "missing_strategy": "zero_impute",
    },
    "reactome_pathways": {
        "description": "Binary membership in Reactome pathways (human ortholog)",
        "source": "data/bulk/pathway_annotations/NCBI2Reactome_PE_Pathway.txt.gz",
        "n_features": "variable (~2000 pathways after min-gene filtering)",
        "dtype": bool,
        "missing_strategy": "zero",
    },
    "hallmark_genesets": {
        "description": "Binary membership in MSigDB Hallmark gene sets (50 sets)",
        "source": "data/bulk/pathway_annotations/h.all.v2024.1.Hs.symbols.gmt.gz",
        "n_features": 50,
        "dtype": bool,
        "missing_strategy": "zero",
    },
    "is_core_essential": {
        "description": "Binary flag: gene in Hart 2015 core essential list",
        "source": "published list (embed as static file)",
        "n_features": 1,
        "dtype": bool,
        "missing_strategy": "zero",
    },
    "coessentiality_pc": {
        "description": "Top-10 PCA components of DepMap co-essentiality profile",
        "source": "derived from CRISPRGeneEffect.csv.gz",
        "n_features": 10,
        "dtype": float,
        "missing_strategy": "zero_vector",
    },
}
```

**Critical:** PCA for co-essentiality features must be fitted on **all genes**
before any train/holdout split. The PCA transform is then applied to both
training and holdout genes. This is the only preprocessing step that sees
all genes; all other features are gene-intrinsic.

### Mouse → Human ortholog mapping

```python
def load_ortholog_map(
    path: str = "data/bulk/menuetto_scherzo_2025/processed/mouse_human_orthologs.parquet"
) -> pd.DataFrame:
    """
    Columns: mouse_symbol | human_symbol | orthology_confidence
    One-to-one orthologs only (confidence == 1).
    Genes without a high-confidence ortholog receive NaN for all
    human-derived features (imputed downstream with median).
    """
    return pd.read_parquet(path)
```

**Coverage expectation:** ~85–90% of ~21,700 mouse genes will have a
high-confidence human ortholog. The remaining ~10–15% are imputed with
median values and flagged with a binary `has_ortholog` feature.

**Pathway membership for mouse genes:** All pathway lookups use the human
ortholog symbol. Mouse genes without an ortholog are treated as pathway-absent
(all zeros in the pathway feature columns).

---

## Models

### Model 1 — Ridge Regression

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
```

`alpha` selected by 5-fold cross-validation on training set only,
from grid `[0.01, 0.1, 1.0, 10.0, 100.0]`.

### Model 2 — Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=seed,
)
```

Fixed defaults for speed across many repeats × strategies.

### Baseline — Mean LFC

```python
def mean_lfc_baseline(train_lfc: pd.Series, holdout_genes: list[str]) -> pd.Series:
    """Predict the training set mean LFC for all holdout genes."""
    return pd.Series(train_lfc.mean(), index=holdout_genes)
```

---

## Evaluation Metrics

### Standard metrics (all strategies)

```python
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hit_threshold_quantile: float = 0.95,
) -> dict:
    """
    Compute all Aim 1 metrics on holdout genes.

    Returns: pearson_r, pearson_p, spearman_r, spearman_p,
             auroc_top5, auprc_top5, precision_at_100, n_holdout
    """
    pearson_r, pearson_p   = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    threshold  = np.quantile(np.abs(y_true), hit_threshold_quantile)
    hit_labels = (np.abs(y_true) >= threshold).astype(int)

    auroc = roc_auc_score(hit_labels, np.abs(y_pred))
    auprc = average_precision_score(hit_labels, np.abs(y_pred))

    top100_true = set(np.argsort(np.abs(y_true))[-100:])
    top100_pred = set(np.argsort(np.abs(y_pred))[-100:])
    precision_at_100 = len(top100_true & top100_pred) / 100

    return {
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "auroc_top5": auroc, "auprc_top5": auprc,
        "precision_at_100": precision_at_100,
        "n_holdout": len(y_true),
    }
```

### Pathway-level generalisation metrics (new for selection strategies)

For each repeat, compute prediction quality **stratified by pathway membership**
of the holdout genes. This answers: does pathway-stratified training improve
prediction specifically for genes in pathways that were represented in training?

```python
def evaluate_by_pathway(
    y_true: pd.Series,                          # indexed by gene symbol
    y_pred: pd.Series,                          # indexed by gene symbol
    train_genes: list[str],
    pathway_membership: dict[str, list[str]],   # pathway_id → [gene_symbols]
    min_holdout_genes: int = 10,
) -> pd.DataFrame:
    """
    For each pathway, compute Pearson r between y_true and y_pred
    restricted to holdout genes in that pathway.

    Also records whether the pathway had ≥1 representative in the training set
    (seen_in_training = True/False), enabling the key comparison:
        seen pathways vs unseen pathways.

    Returns a DataFrame with columns:
        pathway_id, pathway_name, n_holdout_genes, seen_in_training,
        pearson_r, auroc_top5, precision_at_100
    """
    holdout_genes = set(y_true.index)
    train_set     = set(train_genes)
    rows = []

    for pid, members in pathway_membership.items():
        holdout_members = [g for g in members if g in holdout_genes]
        if len(holdout_members) < min_holdout_genes:
            continue

        seen = any(g in train_set for g in members)
        r, _ = pearsonr(y_true.loc[holdout_members].values,
                        y_pred.loc[holdout_members].values)
        rows.append({
            "pathway_id":        pid,
            "n_holdout_genes":   len(holdout_members),
            "seen_in_training":  seen,
            "pearson_r":         r,
        })

    return pd.DataFrame(rows)
```

**Key comparison:** For each strategy, compute:
- Mean Pearson r for pathways **seen** in training (≥1 training gene)
- Mean Pearson r for pathways **unseen** in training (0 training genes)

For random sampling, both will be similar (random chance of coverage).
For Reactome-stratified sampling, seen pathways should have higher r.
If unseen pathway r is also higher, the model has genuinely generalised.

```python
def summarise_pathway_generalisation(
    pathway_metrics: pd.DataFrame,
) -> dict:
    seen   = pathway_metrics[pathway_metrics["seen_in_training"]]
    unseen = pathway_metrics[~pathway_metrics["seen_in_training"]]
    return {
        "n_seen_pathways":    len(seen),
        "n_unseen_pathways":  len(unseen),
        "mean_r_seen":        seen["pearson_r"].mean(),
        "mean_r_unseen":      unseen["pearson_r"].mean(),
        "generalisation_gap": seen["pearson_r"].mean() - unseen["pearson_r"].mean(),
    }
```

### Hit recovery by pathway (for biologically-informed strategies)

For the apoptosis/p53-seeded and BCL-2-seeded strategies, compute hit recovery
specifically within the seeded pathways vs outside them:

```python
def evaluate_hit_recovery_by_pathway_group(
    y_true: pd.Series,
    y_pred: pd.Series,
    seed_genes: set[str],           # genes from the seed hallmarks
    hit_threshold_quantile: float = 0.95,
) -> dict:
    """
    Compare AUROC and Precision@100 for:
    (a) holdout genes that are in the seed pathway group
    (b) holdout genes that are NOT in the seed pathway group

    This tests whether seeding improves prediction of the seeded pathway
    specifically, or generalises to the whole genome.
    """
    holdout_in_seed  = [g for g in y_true.index if g in seed_genes]
    holdout_out_seed = [g for g in y_true.index if g not in seed_genes]

    metrics_in  = evaluate_predictions(y_true.loc[holdout_in_seed].values,
                                       y_pred.loc[holdout_in_seed].values,
                                       hit_threshold_quantile)
    metrics_out = evaluate_predictions(y_true.loc[holdout_out_seed].values,
                                       y_pred.loc[holdout_out_seed].values,
                                       hit_threshold_quantile)
    return {
        "auroc_in_seed_pathway":  metrics_in["auroc_top5"],
        "auroc_out_seed_pathway": metrics_out["auroc_top5"],
        "precision100_in_seed":   metrics_in["precision_at_100"],
        "precision100_out_seed":  metrics_out["precision_at_100"],
        "n_holdout_in_seed":      len(holdout_in_seed),
        "n_holdout_out_seed":     len(holdout_out_seed),
    }
```

---

## Output Schema

```
notebooks/Cas12a_EuMyc/results/aim1/
├── {arm_id}/
│   ├── repeat_{seed:03d}_{strategy}_{model}.json   # per-repeat metrics
│   ├── summary_{strategy}_{model}.csv              # aggregated across repeats
│   └── pathway_generalisation_{strategy}_{model}.parquet  # per-pathway metrics
├── strategy_comparison.csv                          # cross-strategy summary table
└── pathway_generalisation_summary.csv               # seen vs unseen pathway summary
```

`strategy_comparison.csv` columns:
```
arm_id, strategy, model, mean_pearson_r, std_pearson_r,
mean_auroc_top5, mean_precision_at_100,
mean_r_seen_pathways, mean_r_unseen_pathways, generalisation_gap
```

---

## Main Loop

```python
from src.crispr_al.splits import STRATEGY_REGISTRY
from src.crispr_al.models import fit_ridge, fit_rf
from src.crispr_al.metrics import evaluate_predictions, evaluate_by_pathway

def run_aim1(
    arm_id: str,
    strategy_name: str,
    n_train: int = 2000,
    n_repeats: int = 100,
):
    scores   = load_scores(arm_id)
    features = load_features(scores.index)
    pathway_membership = load_pathway_membership(scores.index)  # Reactome + Hallmarks

    split_fn   = STRATEGY_REGISTRY[strategy_name]
    gene_list  = scores.index.tolist()
    results_dir = pathlib.Path(
        f"notebooks/Cas12a_EuMyc/results/aim1/{arm_id}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(n_repeats):
        train_genes, holdout_genes = split_fn(gene_list, n_train=n_train, seed=seed)

        X_train = features.loc[train_genes].values
        y_train = scores.loc[train_genes].values
        X_hold  = features.loc[holdout_genes].values
        y_hold  = scores.loc[holdout_genes]   # keep as Series for pathway eval

        for model_name, model in [
            ("ridge", fit_ridge(X_train, y_train, seed)),
            ("rf",    fit_rf(X_train, y_train, seed)),
            ("mean",  None),
        ]:
            y_pred_vals = (np.full(len(holdout_genes), y_train.mean())
                           if model_name == "mean"
                           else model.predict(X_hold))
            y_pred = pd.Series(y_pred_vals, index=holdout_genes)

            # Standard metrics
            metrics = evaluate_predictions(y_hold.values, y_pred.values)
            metrics.update({
                "aim": 1, "arm_id": arm_id, "strategy": strategy_name,
                "model": model_name, "seed": seed,
                "n_train": n_train, "n_holdout": len(holdout_genes),
            })

            # Pathway-level generalisation
            pw_metrics = evaluate_by_pathway(
                y_hold, y_pred, train_genes, pathway_membership
            )
            pw_summary = summarise_pathway_generalisation(pw_metrics)
            metrics.update(pw_summary)

            out_path = results_dir / f"repeat_{seed:03d}_{strategy_name}_{model_name}.json"
            out_path.write_text(json.dumps(metrics, indent=2))
```

---

## Batch Runner (`scripts/run_aim1.py`)

```python
"""
Run Aim 1 for all arms × strategies × models in parallel.
Usage: python scripts/run_aim1.py [--n-workers 8]
"""
import argparse
from multiprocessing import Pool
from itertools import product

ARMS = ["menuetto_nutlin", "menuetto_s63845",
        "scherzo_nutlin",  "scherzo_s63845", "imdf_t3"]

# Strategy → arms it applies to (condition-specific strategies are restricted)
STRATEGY_ARMS = {
    "random":                        ARMS,
    "reactome_stratified":           ARMS,
    "hallmark_stratified":           ARMS,
    "apoptosis_p53_seeded":          ["menuetto_nutlin", "scherzo_nutlin"],
    "bcl2_seeded":                   ["menuetto_s63845", "scherzo_s63845"],
    "reactome_apoptosis_oversampled": ["menuetto_nutlin", "menuetto_s63845",
                                       "scherzo_nutlin",  "scherzo_s63845"],
}

def run_one(args):
    arm_id, strategy_name = args
    n_repeats = 100 if strategy_name == "random" else 50
    run_aim1(arm_id, strategy_name, n_train=2000, n_repeats=n_repeats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    combos = [
        (arm, strategy)
        for strategy, arms in STRATEGY_ARMS.items()
        for arm in arms
    ]
    print(f"Total (arm, strategy) combinations: {len(combos)}")

    with Pool(args.n_workers) as pool:
        pool.map(run_one, combos)

if __name__ == "__main__":
    main()
```

Total combinations: 5 arms × random + 5 × reactome + 5 × hallmark +
2 × apoptosis_p53 + 2 × bcl2 + 4 × reactome_apoptosis = **23 (arm, strategy) pairs**.
At 50–100 repeats × 3 models each, ~3,450 total model fits.

---

## Visualisations

### 1. Strategy comparison box plot (primary figure)

```python
def plot_strategy_comparison(summary_df: pd.DataFrame, metric: str = "pearson_r"):
    """
    Box plot: metric (y) × strategy (x), faceted by arm_id.
    Each box shows distribution across repeats.
    Colour by model (ridge / rf / mean).
    """
```

### 2. Pathway generalisation: seen vs unseen

```python
def plot_pathway_generalisation(summary_df: pd.DataFrame):
    """
    Paired bar chart: mean_r_seen vs mean_r_unseen for each strategy.
    Faceted by arm_id.
    Smaller generalisation_gap = better generalisation.
    """
```

### 3. Hit recovery within vs outside seed pathway (biologically-informed strategies)

```python
def plot_seed_pathway_hit_recovery(results_df: pd.DataFrame):
    """
    Bar chart: AUROC for holdout genes inside vs outside the seed pathway,
    for apoptosis_p53_seeded and bcl2_seeded strategies.
    Compare to random baseline.
    """
```

### 4. Training set composition heatmap

```python
def plot_training_set_composition(
    train_genes_per_strategy: dict[str, list[str]],
    hallmark_membership: dict[str, list[str]],
):
    """
    Heatmap: strategies (rows) × Hallmark gene sets (columns).
    Cell value = fraction of Hallmark gene set members in training set.
    Shows how each strategy covers the biological space.
    """
```

---

## Expected Results and Interpretation

### Prediction quality (Pearson r on holdout)

| Strategy | Expected Pearson r | Notes |
|---|---|---|
| Mean baseline | ~0.0 | No predictive power |
| Random | 0.20–0.40 | Baseline |
| Reactome-stratified | 0.20–0.42 | Marginal improvement from coverage |
| Hallmark-stratified | 0.20–0.40 | Similar to random (50 seeds is small) |
| Apoptosis/p53 seeded (nutlin arm) | 0.22–0.45 | Modest improvement if p53 genes are informative |
| BCL-2 seeded (S63845 arm) | 0.20–0.42 | Similar |
| Reactome + apoptosis oversampled | 0.22–0.45 | Best expected for drug arms |

### Pathway generalisation

| Strategy | Expected generalisation gap | Interpretation |
|---|---|---|
| Random | ~0 (seen ≈ unseen) | No systematic coverage |
| Reactome-stratified | Small positive (seen > unseen) | Coverage helps slightly |
| Apoptosis/p53 seeded | Large positive for apoptosis pathways | Seeding improves in-pathway prediction |

### Key scientific question

If the apoptosis/p53-seeded strategy improves AUROC for **holdout genes outside
the seed pathway** (not just inside), this is evidence that the model has learned
a generalizable representation of drug sensitivity — not just memorised the seed
pathway. This would be the most interesting positive result.

If improvement is confined to the seed pathway, the model is simply interpolating
within a known pathway, which is less interesting but still practically useful.

---

## Failure Modes and Mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| All strategies ≈ random | Features don't predict LFC | Revisit feature engineering before comparing strategies |
| Reactome-stratified worse than random | Pathway seeds are uninformative genes | Try sampling the highest-variance gene per pathway instead of random |
| Apoptosis seeding hurts performance | Seed genes are outliers that bias the model | Cap seed fraction at 10% instead of 20%; check if seed genes have extreme LFC |
| Pathway generalisation gap = 0 for all | Pathway membership is not predictive | This is a valid null result — report it |
| Mouse genes missing pathway annotations | Low ortholog coverage | Report coverage statistics; flag genes with no pathway annotation |
