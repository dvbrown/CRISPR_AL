# Repository Architecture

This document describes the structure, data flow, and design decisions of the
`CRISPR_AL` codebase.

## Overview

The repository supports benchmarking and active learning for CRISPR perturbation
campaign design. The core task is predicting which genes, when knocked out, will
sensitize or resist a drug treatment — given only a small labelled set of genes
from one or more historical CRISPR screens.

All reusable logic lives in `src/crispr_al/`. Notebooks in `notebooks/` run
experiments and visualise results. Nextflow pipelines under each notebook
workstream coordinate batch runs on SLURM.

---

## Source Package: `src/crispr_al/`

Seven modules, each with a single responsibility:

```
src/crispr_al/
├── screen.py      # Load and normalise CRISPR screens
├── splits.py      # Reproducible train/test split generation
├── features.py    # Gene feature matrix (9 features)
├── models.py      # Model training (Ridge, RandomForest)
├── metrics.py     # Metric computation and JSON schema validation
├── io.py          # Parquet, JSON, and split-manifest I/O
└── plotting.py    # Publication-quality plotnine theme and colour palette
```

### `screen.py` — Screen loading and normalisation

Loads CRISPR screen data from BioGRID ORCS (TSV) and other sources.
Normalises scores to z-scores and assigns binary hit labels using a
z-score threshold (±1.645, corresponding to p < 0.05 one-tailed).

Supported screens:
- **Chen 2019** — MOLM-13, venetoclax, Brunello library (CRISPR Score)
- **Sharon 2019** — MOLM-13-R1, venetoclax, TKOv1 (LFC / MAGeCK)
- **Olivieri 2020** — RPE1-hTERT, genotoxic agents, TKOv2/v3 (NormZ, parquet)
- **Elling 2024** — GEO supplementary files (auto-detected column names)

### `splits.py` — Split generation

Generates reproducible train/test gene splits via seeded random number
generators. Each split is identified by a SHA-256 hash of its payload for
audit and deduplication.

Seed ranges are reserved per design to prevent accidental overlap:

| Design | Seed range | Description |
|--------|-----------|-------------|
| Design A | 11001– | Within-screen holdout (Chen 2019) |
| Design B forward | 21001– | Chen → Sharon transfer |
| Design B reverse | 22001– | Sharon → Chen transfer |

Additional split strategies for EuMyc workstream:
- Random partition
- Reactome-stratified (one gene per qualifying pathway)
- Hallmark-stratified (n genes per Hallmark gene set)
- Apoptosis/p53-seeded (mandatory seeds from apoptosis/p53 Hallmarks)
- BCL-2-seeded (BCL-2/survival Hallmarks as seeds)
- Reactome apoptosis oversampled

### `features.py` — Gene feature matrix

Builds a 9-column feature matrix indexed by `gene_symbol`:

| Feature | Source | Description |
|---------|--------|-------------|
| `expression_molm13` | CCLE | log2(TPM+1) in MOLM-13 (ACH-000362) |
| `coessential_mean_r_top50` | DepMap CRISPR | Mean Pearson r with top-50 co-essential genes |
| `coessential_molm13_chronos` | DepMap CRISPR | Chronos score in MOLM-13 |
| `n_reactome_pathways` | Reactome | Number of Reactome pathways |
| `n_go_bp_terms` | Gene Ontology | Number of GO biological process terms (non-IEA) |
| `n_go_mf_terms` | Gene Ontology | Number of GO molecular function terms (non-IEA) |
| `in_hallmark_apoptosis` | MSigDB Hallmarks | Binary: member of HALLMARK_APOPTOSIS |
| `in_hallmark_oxidative_phosphorylation` | MSigDB Hallmarks | Binary: member of HALLMARK_OXIDATIVE_PHOSPHORYLATION |
| `n_kegg_pathways` | KEGG | Number of KEGG canonical pathways |

Missing genes receive zero-imputation. For the Olivieri 2020 workstream
(RPE1-hTERT cells, not in DepMap/CCLE), only the 6 pathway features are used
(`build_olivieri_features()`).

Mouse-to-human ortholog mapping is available via
`map_mouse_to_human_orthologues()` for EuMyc workstream data.

### `models.py` — Model training

Two models, both trained on z-scored features (StandardScaler fit on train
only to prevent leakage):

| Model | Implementation | Notes |
|-------|---------------|-------|
| Ridge | `RidgeCV` | Cross-validated alpha selection from [0.1, 1.0, 10.0, 100.0] |
| Random Forest | `RandomForestRegressor` | 200 trees, sqrt features, single-threaded |

### `metrics.py` — Metric computation

Computes three families of metrics against held-out genes:

**Regression:** Pearson r, Spearman ρ, R², RMSE, MAE

**Ranking (Precision/Recall@K):** K ∈ {50, 100, 200, 500}
- Sensitizers: ranked by ascending predicted score
- Resistors: ranked by descending predicted score

**Classification:** AUROC and AUPRC for sensitizer and resistor tasks

Bootstrap 95% CIs (BCa method) are computed for all metrics.
All outputs are validated against `metrics.schema.json` before being written
to disk.

### `io.py` — I/O utilities

- `save_parquet` / `load_parquet` — DataFrame persistence
- `save_metrics_json` / `load_metrics_json` — Individual metric records
- `save_split_manifest` — CSV of split metadata (gene lists excluded)
- `save_split_files` — Individual JSON files per split
- `get_code_commit()` — Git short hash for reproducibility tagging in outputs

### `plotting.py` — Visualisation

- `theme_publication()` — Minimal white-background plotnine theme
- `PUBLICATION_COLORS` — 9-colour hex palette for categorical data
- `scale_fill_publication()` / `scale_color_publication()` — Discrete scales

Apply `theme_publication()` to every plotnine figure. For matplotlib figures,
apply the matching rcParams block (see `notebooks/Cas12a_EuMyc/00_explore_menuetto.py`).

---

## Notebooks: `notebooks/`

Each subdirectory is a self-contained workstream with its own scripts,
results, figures, and (where applicable) a Nextflow pipeline.

```
notebooks/
├── crispr_screen_transfer_hold/   # Design A & B benchmarks
│   ├── design_a_analysis.py       # Marimo notebook — Design A
│   ├── splits.yaml                # Canonical split config
│   ├── metrics.schema.json        # JSON schema for all metric outputs
│   ├── artifacts/                 # Intermediate outputs (features, etc.)
│   ├── results/
│   │   ├── design_a/              # Design A metric JSONs and CSVs
│   │   └── design_b/              # Design B metric JSONs and CSVs
│   └── nextflow/
│       ├── pipeline_a/            # Design A Nextflow pipeline
│       └── pipeline_b/            # Design B Nextflow pipeline
├── Cas12a_EuMyc/                  # Cas12a genotoxic screens (EuMyc mouse model)
│   ├── 00_explore_menuetto.py     # Data exploration
│   ├── 01_aim1_within_screen_holdout.py
│   ├── figures/
│   ├── plans/
│   ├── results/
│   └── scripts/
├── RPE1-hTERT_genotoxic/          # Olivieri 2020 benchmark
│   ├── olivieri2020_analysis.py
│   ├── figures/
│   ├── plans/
│   └── results/
├── crispr_star/                   # CRISPR-STAR data exploration
├── bulk_single_dual/              # Bulk epistasis tutorial
├── iterpert/                      # IterPert tutorial (K562)
└── scanpy/                        # scRNA-seq / AnnData tutorial
```

---

## Nextflow Pipelines

Pipelines are co-located with their notebook workstream, not in a top-level
`pipelines/` directory. Each pipeline is self-contained.

### Design A pipeline (`nextflow/pipeline_a/`)

Five-loop autonomous research workflow:

1. **Baseline** — 25 splits × 2 models; generates 50 metric JSON files
2. **Ablation** — Leave-one-out feature ablation (Ridge only; 225 runs)
3. **Reduced model** — Top features from ablation + Ridge + RF
4. **Calibration** — Score distributions, rank correlations, RF feature importances
5. **Report** — Summary Markdown synthesis

### Design B pipeline (`nextflow/pipeline_b/`)

Cross-screen transfer:
- Chen → Sharon: 30 repeats × 2 models
- Sharon → Chen: 30 repeats × 2 models
- 2 overlap-only baselines
- 6 aggregated CSVs
- Total: 122 output files

### Pipeline configuration

Both pipelines resolve data paths relative to the repo root and write results
to their workstream's `results/` directory. SLURM and local execution profiles
are provided. Nextflow work directories write to `work/design_a/` and
`work/design_b/` at the repo root (gitignored).

---

## Data: `data/`

Input datasets are gitignored and must be fetched separately (see
`scripts/fetch_dataset.py` and `manifests/registry.yaml`). Small test fixtures
in `tests/data/` are tracked.

```
data/bulk/
├── chen2019_venetoclax/           # BioGRID ORCS Dataset 408 (CRISPR Score TSV)
├── sharon2019_venetoclax/         # BioGRID ORCS Dataset 406 (MAGeCK TSV, 4 timepoints)
├── ccle_expression/               # CCLE OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz
├── depmap_crispr_gene_effect/     # DepMap CRISPRGeneEffect.csv.gz (Chronos)
├── depmap_model_metadata/         # DepMap Model.csv.gz (cell line → ACH ID)
├── pathway_annotations/           # Reactome, GO (GAF), Hallmarks GMT, KEGG GMT
├── menuetto_scherzo_2025/         # Cas12a EuMyc screens (processed parquet/CSV)
├── olivieri2020/                  # Olivieri 2020 NormZ matrices (parquet)
└── elling2024/                    # Elling 2024 GEO supplementary files
```

Key identifiers:
- MOLM-13 DepMap ID: **ACH-000362** (used for expression and co-essentiality features)
- Chen 2019 / Sharon 2019 gene overlap: **17,091 genes** (Design B scope)

---

## Tests: `tests/`

One test file per source module:

| Test file | Module under test |
|-----------|------------------|
| `test_screen.py` | `screen.py` |
| `test_splits.py` | `splits.py` |
| `test_features.py` | `features.py` |
| `test_models.py` | `models.py` |
| `test_metrics.py` | `metrics.py` |
| `test_cross_screen.py` | Cross-screen transfer integration |

`conftest.py` provides synthetic fixtures (`tiny_screen_df`, `tiny_features_df`,
`tiny_screen_normalized`) used across multiple test files.

Run with: `pytest tests/`

---

## Scripts: `scripts/`

| Script | Purpose |
|--------|---------|
| `activate_env.sh` | Activate the project micromamba environment |
| `fetch_dataset.py` | Download datasets via the registry manifest |
| `verify_checksums.py` | Verify downloaded dataset integrity |
| `generate_synthetic.py` | Generate synthetic CRISPR screen data for testing |
| `create_gears_iterpert_tiny.py` | Create tiny GEARS/IterPert fixture for tests |
| `elling2024/` | Processing scripts for Elling 2024 GEO data |
| `olivieri2020/` | Processing scripts for Olivieri 2020 data |

---

## Docs: `docs/`

| File | Purpose |
|------|---------|
| `plan.md` | Goals, milestones, conventions |
| `overview.md` | Active learning concepts, acquisition strategies |
| `architecture.md` | This document |
| `repo_list/repo_list.md` | Curated candidate external repositories |
| `evals/template.md` | Evaluation template for external repositories |
| `evals/summary.md` | Comparison of evaluated repositories |
| `evals/olivieri2020.md` | Evaluation note for Olivieri 2020 dataset |
| `decision-log/` | Architectural decision records |

---

## Key Design Decisions

**Feature scaler fit on train only.** `StandardScaler` is fit exclusively on
the training set before transforming both train and test. This is enforced in
`models.scale_features()` and tested.

**Split hashing for audit.** Every split is identified by a SHA-256 hash of
its sorted gene lists and metadata. This allows deduplication across runs and
links metric JSON files back to exact data partitions.

**Screen-level z-score normalisation is not leakage.** `zscore_normalize()`
operates over all genes in the screen. This is equivalent to centering and
scaling the outcome variable using population statistics, not information from
the test set labels.

**Zero-imputation for out-of-matrix genes.** Sharon 2019 contains genes absent
from the DepMap/CCLE feature matrix. These receive zero-imputed features. This
is noted in `splits.py` and tested in `test_cross_screen.py`.

**One pipeline directory per design.** Pipelines are co-located with their
notebook workstream rather than in a top-level `pipelines/` directory, making
each workstream independently navigable.

**One micromamba environment per external repo.** External codebases under
evaluation have conflicting dependencies. Each gets its own isolated prefix
under `repos/.envs/<repo>/`.
