# Research Plan: Active Learning Simulation on Menuetto/Scherzo CRISPR Screens

## Overview

This research plan replaces the previous venetoclax-focused design (Chen 2019 / Sharon 2019),
which hit a roadblock in cross-dataset transfer due to batch effects and biological confounds
between the two AML cell lines. We pivot to a single, richly structured public dataset that
provides multiple clean transfer axes within a controlled experimental system.

**Primary dataset:** Jin, Deng, La Marca et al. (2025), *Nature Communications*
— "Advancing the genetic engineering toolbox by combining AsCas12a knock-in mice with
ultra-compact screening." PMID 39885149. GEO: **GSE285778**.

This dataset contains three independent screens run from the same lab, same cell system,
and same sequencing pipeline:

| Screen arm | Cell model | Library | Conditions | Timepoints | Replicates |
|---|---|---|---|---|---|
| **A** — Enrichment | Eµ-MYC lymphoma (#20) | Menuetto (~21,743 genes, 2 crRNA/gene) | Nutlin-3a, S63845, DMSO | 1 endpoint | 6 |
| **B** — Enrichment | Eµ-MYC lymphoma (#20) | Scherzo (~21,721 genes, 4 crRNA/gene) | Nutlin-3a, S63845, DMSO | 1 endpoint | 6 |
| **C** — Dropout | iMDF (immortalised fibroblasts) | Menuetto | Untreated growth | **4 timepoints** (T0, T4, T8, T12 days) | 3 |

Arms A and B were run in **the same cell line, same passage, same drug concentrations,
same sequencing run** — the only variable is guide RNA architecture (dual vs quad vector).
This makes A↔B the cleanest possible cross-dataset transfer test: biology is held constant,
only the measurement instrument changes.

---

## Scientific Questions

1. **Aim 1 — Focused → Genome-wide prediction (within-screen holdout):**
   Can a model trained on a random subset of ~2,000 genes predict the remaining ~19,000
   gene scores in the same screen? Tested independently in Arms A, B, and C.

2. **Aim 2 — Cross-library transfer (Menuetto ↔ Scherzo):**
   Can gene-level scores learned from one guide architecture (Menuetto, 2 crRNAs/gene)
   predict scores in the other (Scherzo, 4 crRNAs/gene), given identical biology?
   This establishes an upper bound on transferability and isolates guide-level noise
   from biological noise.

3. **Aim 3 — Cross-condition transfer:**
   Can a model trained on the nutlin-3a arm predict the S63845 arm (and vice versa)?
   These are mechanistically distinct selection pressures (p53 activation vs MCL-1
   inhibition) in the same cell line — a test of whether gene features generalise
   across drug contexts.

4. **Aim 4 — Temporal active learning simulation (dropout screen):**
   Using the iMDF dropout arm (4 timepoints), simulate an iterative active learning
   loop where the model queries the most informative genes at each timepoint and
   evaluates whether early queries improve prediction of late-timepoint essentiality.

---

## Repo Location

**New folder:** `notebooks/Cas12a_EuMyc/`

This is a clean separation from the existing `notebooks/crispr_screen_transfer/`
(which retains the Chen/Sharon venetoclax work). The new folder follows the same
conventions: a `scripts/` subfolder for batch runners, a `splits.yaml` for
canonical split configs, and Marimo notebooks for interactive analysis.

```
notebooks/Cas12a_EuMyc/
├── 00_download_and_preprocess.py     # Data download, MAGeCK scoring, feature engineering
├── 01_aim1_within_screen_holdout.py  # Aim 1: focused → genome-wide (Arms A, B, C)
├── 02_aim2_cross_library_transfer.py # Aim 2: Menuetto ↔ Scherzo transfer
├── 03_aim3_cross_condition.py        # Aim 3: nutlin-3a ↔ S63845 transfer
├── 04_aim4_temporal_al.py            # Aim 4: iMDF dropout AL simulation
├── splits.yaml                       # Canonical split seeds and n_train values
└── scripts/
    ├── run_aim1.py                   # Batch runner for Aim 1
    ├── run_aim2.py                   # Batch runner for Aim 2
    └── run_aim3_aim4.py              # Batch runner for Aims 3 & 4
```

Data lives at:
```
data/bulk/menuetto_scherzo_2025/
├── raw/
│   ├── GSE285778_EuMycCountMenuetto.txt.gz   # crRNA counts, Eµ-MYC × Menuetto
│   ├── GSE285778_EuMycCountScherzo.txt.gz    # crRNA counts, Eµ-MYC × Scherzo
│   ├── GSE285778_iMDFCount.txt.gz            # crRNA counts, iMDF dropout
│   └── GSE285778_InVivoCount.txt.gz          # crRNA counts, in vivo (reference only)
└── processed/
    ├── menuetto_gene_scores.parquet          # Gene-level LFC + FDR, all conditions
    ├── scherzo_gene_scores.parquet           # Gene-level LFC + FDR, all conditions
    ├── imdf_gene_scores_by_timepoint.parquet # Gene-level LFC per timepoint vs T0
    └── features_mouse_genes.parquet          # Feature matrix (see below)
```

---

## Data Download Instructions (for AI agent)

### Step 1 — Download raw count files from GEO

GEO accession: **GSE285778** (public as of 2025-01-03).
All four supplementary files are available via HTTPS from the GEO FTP mirror.

```python
import urllib.request, os, pathlib

RAW_DIR = pathlib.Path("data/bulk/menuetto_scherzo_2025/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285778/suppl/"

FILES = [
    "GSE285778_EuMycCountMenuetto.txt.gz",
    "GSE285778_EuMycCountScherzo.txt.gz",
    "GSE285778_iMDFCount.txt.gz",
    "GSE285778_InVivoCount.txt.gz",
]

for fname in FILES:
    dest = RAW_DIR / fname
    if not dest.exists():
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(GEO_BASE + fname, dest)
        print(f"  -> saved to {dest}")
    else:
        print(f"  -> already exists: {dest}")
```

**Expected file sizes (approximate):**
- `EuMycCountMenuetto.txt.gz` — ~1.0 MB
- `EuMycCountScherzo.txt.gz` — ~0.6 MB
- `iMDFCount.txt.gz` — ~1.4 MB
- `InVivoCount.txt.gz` — ~0.3 MB (reference only, not used in AL simulation)

### Step 2 — Parse count files

The GEO supplementary files are tab-separated count matrices. Expected format:

```
crRNA_id  | gene  | EuMyc_Menuetto_Input_Rep1 | EuMyc_Menuetto_Input_Rep2 | ...
```

Parse with:

```python
import pandas as pd, gzip

def load_count_matrix(path: str) -> pd.DataFrame:
    """Load a GEO supplementary crRNA count matrix."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    return df

menuetto_counts = load_count_matrix("data/bulk/menuetto_scherzo_2025/raw/GSE285778_EuMycCountMenuetto.txt.gz")
scherzo_counts  = load_count_matrix("data/bulk/menuetto_scherzo_2025/raw/GSE285778_EuMycCountScherzo.txt.gz")
imdf_counts     = load_count_matrix("data/bulk/menuetto_scherzo_2025/raw/GSE285778_iMDFCount.txt.gz")
```

**Verify sample names match GEO metadata:**
- Menuetto Eµ-MYC: Input×6, DMSO×6, Nutlin×6, S63845×6 = 24 columns
- Scherzo Eµ-MYC: same structure = 24 columns
- iMDF: T0×3, T1×3, T2×3, T3×3 = 12 columns

### Step 3 — Compute gene-level scores with MAGeCK

Use MAGeCK `test` to compute gene-level log fold change and FDR for each
condition comparison. Install via conda: `conda install -c bioconda mageck`.

```bash
# Menuetto: Nutlin-3a vs Input
mageck test \
  --count-table data/bulk/menuetto_scherzo_2025/raw/GSE285778_EuMycCountMenuetto.txt.gz \
  --treatment-id  EuMyc_Menuetto_Nutlin_Rep1,EuMyc_Menuetto_Nutlin_Rep2,...Rep6 \
  --control-id    EuMyc_Menuetto_Input_Rep1,...Rep6 \
  --output-prefix data/bulk/menuetto_scherzo_2025/processed/menuetto_nutlin \
  --gene-lfc-method median \
  --adjust-method fdr

# Repeat for: Menuetto S63845 vs Input
# Repeat for: Scherzo Nutlin vs Input
# Repeat for: Scherzo S63845 vs Input
# Repeat for: iMDF T1 vs T0, T2 vs T0, T3 vs T0
```

**Key MAGeCK output columns used:**
- `id` — gene symbol (mouse)
- `lfc` — median log2 fold change across guides
- `neg|score`, `neg|fdr` — depletion (essential / sensitiser)
- `pos|score`, `pos|fdr` — enrichment (resistance / dropout suppressor)

Alternatively, compute LFC directly in Python using the `pydeseq2` or
`limma-voom` approach (as used in the original paper):

```python
import numpy as np

def compute_lfc_per_gene(counts: pd.DataFrame,
                          treatment_cols: list[str],
                          control_cols: list[str],
                          gene_col: str = "gene") -> pd.DataFrame:
    """
    Compute median log2 fold change per gene from raw crRNA counts.
    Adds a pseudocount of 1 before normalisation.
    """
    # Normalise to reads-per-million within each sample
    rpm = counts.div(counts.sum(axis=0) / 1e6, axis=1)
    rpm = rpm + 1  # pseudocount

    treat_mean = rpm[treatment_cols].mean(axis=1)
    ctrl_mean  = rpm[control_cols].mean(axis=1)
    lfc_guide  = np.log2(treat_mean / ctrl_mean)

    # Aggregate to gene level
    gene_lfc = (
        lfc_guide
        .groupby(counts[gene_col])
        .median()
        .rename("lfc")
        .reset_index()
    )
    return gene_lfc
```

### Step 4 — Feature engineering (mouse genes)

Because this dataset is murine, features must be derived from mouse resources
or via human ortholog mapping. Use the following strategy:

**Option A — Mouse DepMap (preferred if available):**
DepMap includes some mouse cell line data; check for Eµ-MYC or related
lymphoma lines. If absent, fall back to Option B.

**Option B — Human ortholog mapping (recommended):**
Map mouse gene symbols to human orthologs via Ensembl BioMart, then
apply existing CCLE/DepMap features from the human ortholog.

```python
# Download mouse→human ortholog table from Ensembl BioMart
import pandas as pd

BIOMART_URL = (
    "https://www.ensembl.org/biomart/martservice?query="
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<!DOCTYPE Query>"
    "<Query virtualSchemaName='default' formatter='TSV' header='1' "
    "uniqueRows='1' count='' datasetConfigVersion='0.6'>"
    "<Dataset name='mmusculus_gene_ensembl' interface='default'>"
    "<Attribute name='external_gene_name'/>"
    "<Attribute name='hsapiens_homolog_associated_gene_name'/>"
    "<Attribute name='hsapiens_homolog_orthology_confidence'/>"
    "</Dataset></Query>"
)
# Use gget or pybiomart as a cleaner alternative:
# pip install gget
import gget
orthologs = gget.info(["Trp53"], species="mus_musculus")  # example
```

**Recommended feature set per gene:**

| Feature | Source | Notes |
|---|---|---|
| Human ortholog DepMap Chronos score (MOLM-13 or pan-cancer mean) | DepMap 25Q3 | Already downloaded |
| Human ortholog expression log-TPM | CCLE 25Q2 | Already downloaded |
| Pathway membership (Reactome, GO Hallmarks) | MSigDB / GOA | Already downloaded |
| Mouse gene essentiality (Hart 2015 core essential list) | Published list | Binary flag |
| crRNA GC content mean | Computed from library | Guide quality proxy |
| Number of paralogs (mouse) | Ensembl | Redundancy proxy |

Save the final feature matrix as:
```
data/bulk/menuetto_scherzo_2025/processed/features_mouse_genes.parquet
```

---

## Aim 1 — Within-Screen Holdout

**Question:** Can a model trained on ~2,000 randomly sampled genes predict
the remaining ~19,700 gene scores in the same screen?

**Design:**
- For each of 100 iterations (seeded):
  1. Randomly sample 2,000 genes as training set
  2. Remaining ~19,700 genes = holdout
  3. Train Ridge Regression and Random Forest on training gene features + LFC labels
  4. Predict holdout gene LFC scores
  5. Compute Pearson r, Spearman r, AUROC (top 5% hits), Precision@100

- Run independently for:
  - Menuetto × Nutlin-3a
  - Menuetto × S63845
  - Scherzo × Nutlin-3a
  - Scherzo × S63845
  - iMDF × T3 vs T0 (single timepoint label)

**Expected output:** Distribution of Pearson r across 100 iterations per
condition, with 95% CI. Baseline = mean LFC of training set (intercept-only model).

---

## Aim 2 — Cross-Library Transfer (Menuetto ↔ Scherzo)

**Question:** Can gene scores from Menuetto predict Scherzo scores (and vice versa),
given that biology is held constant and only guide architecture differs?

**Design:**
- Train on all Menuetto genes → predict all Scherzo genes (same condition)
- Train on all Scherzo genes → predict all Menuetto genes (same condition)
- Evaluate on the ~21,700-gene overlapping set
- Compare to Aim 1 within-screen performance as upper bound

**Key insight:** Any gap between Aim 1 (within-screen) and Aim 2 (cross-library)
performance is attributable to guide-level noise, not biological noise. This
quantifies the "measurement floor" for cross-dataset transfer.

**Important caveat to track:** Scherzo has 4 crRNAs/gene vs Menuetto's 2.
Scherzo gene scores will have lower variance (more guides = more stable estimates).
Track score variance per gene across both libraries and report as a covariate.

---

## Aim 3 — Cross-Condition Transfer

**Question:** Can a model trained on nutlin-3a scores predict S63845 scores
(and vice versa) within the same library?

**Design:**
- Train on Menuetto × Nutlin → predict Menuetto × S63845
- Train on Menuetto × S63845 → predict Menuetto × Nutlin
- Repeat for Scherzo
- Evaluate on all ~21,700 genes

**Biological interpretation:** Nutlin-3a selects for p53 pathway loss; S63845
selects for MCL-1 dependency loss (BAX is the key hit). These are mechanistically
distinct. Low transfer performance here is scientifically expected and informative —
it tells us how much of the gene score signal is condition-specific vs shared
(e.g. essential genes drop out in both).

---

## Aim 4 — Temporal Active Learning Simulation (iMDF Dropout)

**Question:** In an iterative active learning loop, can early-timepoint queries
improve prediction of late-timepoint essentiality scores?

**Design:**
The iMDF dropout screen has 4 timepoints: T0 (day 0), T1 (day 4), T2 (day 8),
T3 (day 12). Gene-level LFC vs T0 becomes progressively more informative as
essential genes drop out.

**AL simulation loop:**

```
Initialise: label_set = random sample of n_seed=200 genes (LFC at T3)
            unlabelled_pool = remaining ~21,500 genes

For each AL round r = 1 ... R:
    1. Train model on label_set (features → T3 LFC)
    2. Score all unlabelled genes by acquisition function
    3. Query top-k=100 genes (highest uncertainty or expected improvement)
    4. Add queried genes + their T3 labels to label_set
    5. Evaluate: Pearson r on held-out test set (fixed 2,000 genes)

Compare against:
    - Random acquisition (baseline)
    - Greedy (highest predicted |LFC|)
    - Uncertainty sampling (highest model variance)
```

**Temporal extension:** Use T1 and T2 LFC as cheap proxy labels for early
rounds (lower cost, noisier signal), then switch to T3 labels in later rounds.
This simulates a real screen where early timepoints are cheaper to collect.

**Expected output:** Learning curves (Pearson r vs number of labelled genes)
for each acquisition strategy, with 95% CI across 20 random seeds.

---

## Evaluation Metrics

| Metric | Description | Used in |
|---|---|---|
| Pearson r | Linear correlation of predicted vs actual LFC | All aims |
| Spearman r | Rank correlation (robust to outliers) | All aims |
| AUROC | Classify top 5% hits vs rest | Aims 1–3 |
| Precision@100 | Fraction of true top-100 hits in predicted top-100 | Aims 1–3 |
| Learning curve AUC | Area under Pearson r vs n_labelled curve | Aim 4 |
| Score variance ratio | Scherzo/Menuetto variance per gene | Aim 2 |

---

## Registry Entry (add to `manifests/registry.yaml`)

```yaml
- name: menuetto_scherzo_2025
  type: real
  version: GSE285778
  source:
    local_path: data/bulk/menuetto_scherzo_2025
    target_subdir: real/menuetto_scherzo_2025/vGSE285778
    checksum_file: checksums.sha256
  source_url: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285778/suppl/
  description: >
    Jin, Deng, La Marca et al. (2025) Nat Commun. Genome-wide Cas12a CRISPR screens
    in Eµ-MYC mouse lymphoma cells (Menuetto dual-library and Scherzo quad-library)
    and iMDF dropout screen (4 timepoints). GEO GSE285778. Used as primary substrate
    for active learning simulation (Aims 1–4).
  publication:
    pmid: 39885149
    doi: 10.1038/s41467-025-56282-2
    journal: Nature Communications
    year: 2025
  organism: Mus musculus
  cell_models:
    - name: EuMyc_lymphoma_line20
      type: transformed_lymphoma
      screen_arms: [menuetto_enrichment, scherzo_enrichment]
    - name: iMDF_enAsCas12aKI_KI
      type: immortalised_fibroblast
      screen_arms: [menuetto_dropout]
  libraries:
    - name: Menuetto
      type: dual_crRNA
      n_genes: 21743
      n_constructs: 44000
      guides_per_gene: 2
    - name: Scherzo
      type: quad_crRNA
      n_genes: 21721
      n_constructs: 23000
      guides_per_gene: 4
  files:
    - path: raw/GSE285778_EuMycCountMenuetto.txt.gz
      format: tsv.gz
      size_bytes: 1048576
    - path: raw/GSE285778_EuMycCountScherzo.txt.gz
      format: tsv.gz
      size_bytes: 638976
    - path: raw/GSE285778_iMDFCount.txt.gz
      format: tsv.gz
      size_bytes: 1468006
    - path: raw/GSE285778_InVivoCount.txt.gz
      format: tsv.gz
      size_bytes: 273408
  downloaded_at: null  # set after download
  notes: >
    Cross-library transfer (Menuetto <-> Scherzo) is the primary Aim 2 test.
    Same cell line, same passage, same drug concentrations — only guide
    architecture differs. iMDF dropout arm provides 4-timepoint temporal
    structure for Aim 4 AL simulation. In vivo counts (InVivoCount) are
    downloaded for reference but not used in AL simulation (monoclonal
    tumour design precludes standard gene-score regression).
```

---

## Folder Creation Checklist

Run once to scaffold the new workstream:

```bash
# From repo root
mkdir -p notebooks/Cas12a_EuMyc/scripts
mkdir -p data/bulk/menuetto_scherzo_2025/raw
mkdir -p data/bulk/menuetto_scherzo_2025/processed

touch notebooks/Cas12a_EuMyc/00_download_and_preprocess.py
touch notebooks/Cas12a_EuMyc/01_aim1_within_screen_holdout.py
touch notebooks/Cas12a_EuMyc/02_aim2_cross_library_transfer.py
touch notebooks/Cas12a_EuMyc/03_aim3_cross_condition.py
touch notebooks/Cas12a_EuMyc/04_aim4_temporal_al.py
touch notebooks/Cas12a_EuMyc/splits.yaml
touch notebooks/Cas12a_EuMyc/scripts/run_aim1.py
touch notebooks/Cas12a_EuMyc/scripts/run_aim2.py
touch notebooks/Cas12a_EuMyc/scripts/run_aim3_aim4.py
```

Add `data/bulk/menuetto_scherzo_2025/raw/` and
`data/bulk/menuetto_scherzo_2025/processed/` to `.gitignore`
(raw data files should not be committed).

---

## Key Differences from Previous Plan

| Aspect | Previous (Chen/Sharon) | New (Menuetto/Scherzo) |
|---|---|---|
| Organism | Human (AML) | Mouse (B-cell lymphoma) |
| Cross-dataset transfer | Different cell lines (MOLM-13 vs MOLM-13-R1) | Same cell line, different guide architecture |
| Transfer confounds | Batch effects + biological differences | Guide-level noise only |
| Timepoints | 2–4 (Sharon) | **4 (iMDF dropout)** |
| Data availability | BioGRID ORCS (processed scores only) | **GEO GSE285778 (raw counts)** |
| Feature engineering | Human CCLE/DepMap direct | Human ortholog mapping required |
| Upper bound on transfer | Unknown | Quantifiable (Aim 2 = same biology) |

---

## References

1. Jin W, Deng Y, La Marca JE et al. Advancing the genetic engineering toolbox by
   combining AsCas12a knock-in mice with ultra-compact screening.
   *Nat Commun* 16, 974 (2025). https://doi.org/10.1038/s41467-025-56282-2. PMID 39885149.

2. GEO Series GSE285778. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285778.
   Submitted 2025-01-03. Public.
