# Data Directory

All data files are gitignored. This README documents what lives here,
why it exists, and how it is used in CRISPR perturbation modelling.

---

## Layout

```
data/bulk/<dataset_name>/   — raw/processed external datasets
tests/data/                 — small tracked fixtures for unit tests
```

---

## data/bulk — Dataset Catalogue

### chen2019_venetoclax/

| Field | Value |
|-------|-------|
| Source | BioGRID ORCS, Dataset 408 |
| Publication | Chen X et al. 2019, *Cancer Discovery*, PubMed 31048321 |
| Cell line | MOLM-13 (AML) |
| Library | Brunello (genome-wide, ~19,115 genes) |
| Method | Differential CRISPR Score (CRISPRn/Cas9) |
| Drug | Venetoclax 2.0 µM |

**Files:**
- `BIOGRID-ORCS-SCREEN_1392-*.screen.tab.txt` — 8-day exposure
- `BIOGRID-ORCS-SCREEN_1393-*.screen.tab.txt` — 16-day exposure

**Key columns:** `OFFICIAL_SYMBOL` (HGNC), `IDENTIFIER_ID` (Entrez),
`SCORE.1` (CRISPR Score: positive = resistance, negative = sensitivity).

**Use in modelling:**
Primary genome-wide venetoclax CRISPR screen. Used as the training
source for **Aim 1 Design A** (within-screen holdout: sample 2,000
genes, predict remaining ~17,000) and as one leg of **Design B**
(train on Chen → predict Sharon, and vice versa). The 2-screen time
course allows assessment of score stability across exposure lengths.

---

### sharon2019_venetoclax/

| Field | Value |
|-------|-------|
| Source | BioGRID ORCS, Dataset 406 |
| Publication | Sharon D et al. 2019, *Science Translational Medicine*, PubMed 31666400 |
| Cell line | MOLM-13-R1 (venetoclax-resistant AML derivative) |
| Library | TKO v1 (genome-wide, ~17,237 genes) |
| Method | MAGeCK (CRISPRn/Cas9) |
| Drug | Venetoclax 400 nM |

**Files:**
- `BIOGRID-ORCS-SCREEN_1401-*.screen.tab.txt` — 8-day exposure
- `BIOGRID-ORCS-SCREEN_1402-*.screen.tab.txt` — 16-day exposure
- `BIOGRID-ORCS-SCREEN_1403-*.screen.tab.txt` — 22-day exposure
- `BIOGRID-ORCS-SCREEN_1404-*.screen.tab.txt` — 29-day exposure

**Key columns:** `OFFICIAL_SYMBOL`, `IDENTIFIER_ID`, `SCORE.1`
(MAGeCK neg score = re-sensitization), `SCORE.2` (neg FDR),
`SCORE.3` (pos score = resistance), `SCORE.5` (log2 fold change).

**Use in modelling:**
Independent genome-wide venetoclax screen in the resistant cell line.
Primary target for **Aim 1 Design B** cross-study transfer. The
4-screen time course makes it a rich test of whether predictions hold
across multiple exposure windows.

**Gene overlap with Chen 2019:** 17,091 shared genes (of 17,230 in
Sharon screen 1401); 2,018 Chen-only genes; 139 Sharon-only genes.
Design B operates on the ~17,000-gene shared space.

---

### depmap_crispr_gene_effect/

| Field | Value |
|-------|-------|
| Source | DepMap Public 25Q3 |
| File | `CRISPRGeneEffect.csv.gz` (413 MB → 186 MB gzipped) |
| Dimensions | 1,186 cell lines × 18,435 genes |
| Score | Chronos gene effect (negative = essential; ~−1 = lethal) |

**Column format:** `SYMBOL (ENTREZ_ID)` — e.g. `BCL2 (596)`.
Row index: DepMap model ID (ACH-XXXXXX).

**Use in modelling:**
Provides **co-essentiality features** for holdout gene prediction.
For each holdout gene, its Chronos score profile across ~1,000 cell
lines is correlated with the profiles of training genes. High
correlation implies functional relatedness and is a strong predictor
of shared venetoclax sensitivity. Also used to classify genes as
common essentials, selective essentials, or non-essential — a
useful stratification feature.

**MOLM-13 ACH ID: ACH-000362** (stripped name: `MOLM13`). Present in both
this file and CCLE expression. The historical ID ACH-001187 is incorrect.

---

### ccle_expression/

| Field | Value |
|-------|-------|
| Source | DepMap Public 25Q2 |
| File | `OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz` (231 MB gzipped) |
| Dimensions | 1,684 cell lines × 19,205 genes |
| Score | log2(TPM + 1) RNA expression |

**Column format:** `SYMBOL (ENTREZ_ID)` — same as DepMap CRISPR.
Row index: DepMap model ID (ACH-XXXXXX). 1,112 cell lines overlap
with the CRISPR gene effect file.

**Use in modelling:**
Provides **expression-level features** for holdout gene prediction.
A gene's expression in the target cell line context is a basic
prior on whether it is active and therefore likely to have a
CRISPR phenotype. Used as a per-gene scalar feature (log-TPM in
the relevant AML context) and to filter unexpressed genes before
modelling.

**MOLM-13 ACH ID: ACH-000362** (`MOLM13`) — present in this file. Use for
both Chen 2019 and Sharon 2019 features (MOLM-13-R1 has no separate CCLE entry).

---

### pathway_annotations/

| File | Source | Description |
|------|--------|-------------|
| `NCBI2Reactome_PE_Pathway.txt.gz` (7 MB) | Reactome (Nov 2024) | Entrez Gene → Reactome pathway mapping for all species; filter to `Homo sapiens` rows |
| `goa_human.gaf.gz` (15 MB) | Gene Ontology (Jan 2025) | GO Annotation File for human proteins: gene → GO term (BP/MF/CC) with evidence codes |
| `h.all.v2024.1.Hs.symbols.gmt.gz` (21 KB) | MSigDB Hallmarks | 50 curated hallmark gene sets (HGNC symbols) — high-level biological themes |
| `c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt.gz` (26 KB) | MSigDB C2/KEGG | 186 canonical KEGG pathway gene sets (HGNC symbols) |

**Use in modelling:**
Pathway membership is converted to **binary feature vectors** per
gene: gene × pathway indicator matrix. A holdout gene sharing
pathway membership with training sensitizers is predicted to also
sensitize. Features are derived from one or more of these sources
and optionally combined. Reactome and GO provide fine-grained
coverage; Hallmarks and KEGG provide interpretable coarse features
useful for feature importance analysis.

**Gene ID alignment:** BioGRID ORCS uses `OFFICIAL_SYMBOL` (HGNC).
DepMap/CCLE use `SYMBOL (ENTREZ_ID)`. Reactome uses Entrez IDs;
GOA uses UniProt accessions mapped via gene symbol; GMT files use
HGNC symbols. Harmonization step: strip Entrez suffix from DepMap
columns and join on HGNC symbol as canonical key.

---

## Verification Summary (as of 2026-03-08)

| Check | Result |
|-------|--------|
| Chen 2019 genes in DepMap | 17,568 / 19,109 (92%) |
| Sharon 2019 genes in DepMap | 16,360 / 17,230 (95%) |
| Chen ↔ Sharon gene overlap | 17,091 genes (Design B scope) |
| MOLM-13 ACH ID | **ACH-000362** (`MOLM13`) — present in both DepMap 25Q3 and CCLE 25Q2 |
| MOLM-13-R1 in DepMap/CCLE | Absent — use MOLM-13 (ACH-000362) as proxy |
| Gene ID format (DepMap/CCLE) | `SYMBOL (ENTREZ_ID)` |
| Gene ID format (BioGRID ORCS) | `OFFICIAL_SYMBOL` (HGNC) + `IDENTIFIER_ID` (Entrez) |

---

### depmap_model_metadata/

| File | Size | Source |
|------|------|--------|
| `Model.csv.gz` | 123 KB | DepMap portal |

Maps ACH model IDs to cell line names, lineage, primary disease, RRID, and
OncoTree annotations. 2,116 models total; 63 AML. Used to confirm MOLM-13
ACH ID (ACH-000362) and to enumerate AML lines available in CCLE/DepMap
for context-specific feature selection.

---

### menuetto_scherzo_2025/

| Field | Value |
|-------|-------|
| Source | Internal (Menuetto/Scherzo experimental datasets) |
| Cell line | EuMyc mouse B-cell lymphoma model |
| Library | Cas12a sgRNA library (genotoxic screen panel) |
| Method | Cas12a CRISPR screen |
| Format | Processed parquet/CSV files in `processed/` subdirectory |

**Files:**
- `processed/` — Gene-level and sgRNA-level summary files

**Use in modelling:**
Primary data source for the `notebooks/Cas12a_EuMyc/` workstream. Used
to benchmark within-screen holdout prediction (Aim 1) and to develop
stratified/seeded split strategies (`splits.py` EuMyc functions). Mouse
gene symbols are mapped to human orthologues via
`features.map_mouse_to_human_orthologues()` before feature assembly.

---

### olivieri2020/

| Field | Value |
|-------|-------|
| Source | Olivieri et al. 2020 (GEO / publication supplementary) |
| Publication | Olivieri M et al. 2020, *Molecular Cell*, PubMed 33147444 |
| Cell line | RPE1-hTERT (non-transformed human retinal pigment epithelium) |
| Library | TKOv2 / TKOv3 (genome-wide, ~18,000 genes) |
| Method | Cas12a CRISPR screen |
| Drug | Multiple genotoxic agents and DNA damage response inhibitors |
| Format | NormZ matrix (parquet) |

**Use in modelling:**
Benchmark dataset for the `notebooks/RPE1-hTERT_genotoxic/` workstream.
Loaded via `screen.load_olivieri_normz()`. Because RPE1-hTERT is not
in DepMap/CCLE, only the 6 pathway features are used (`features.build_olivieri_features()`).
Leave-one-drug-out (LODO) splits are generated via `splits.generate_lodo_splits()`.

---

### elling2024/

| Field | Value |
|-------|-------|
| Source | Elling et al. 2024 (GEO supplementary files) |
| Publication | Elling U et al. 2024 |
| Method | CRISPR screen (Cas9) |
| Format | GEO supplementary TSV/CSV (gene and score columns auto-detected) |

**Files:**
- GEO supplementary files downloaded and processed by `scripts/elling2024/`

**Use in modelling:**
Loaded via `screen.load_elling_scores()`, which auto-detects gene symbol
and score columns. Intended for cross-screen transfer benchmarks
extending beyond the Chen/Sharon venetoclax pair.

---

## BIOGRID-ORCS-SCREEN_INDEX file

`data/bulk/BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt` — index
of all BioGRID ORCS screens, used to look up screen metadata
(cell line, organism, drug, library type) by SCREEN_ID.
