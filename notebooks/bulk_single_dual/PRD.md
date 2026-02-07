# PRD: Active Learning Tutorial for Bulk CRISPR Epistasis (K562)

- Status: Draft
- Updated: 2026-02-07
- Owner: OpenCode
- Location: `notebooks/bulk_single_dual/`

## 1) Problem statement
We need a lightweight, reproducible tutorial that starts from a **single-gRNA bulk CRISPR screen** and uses **active learning (AL)** to prioritize **gene pairs** for epistasis testing. The tutorial must be practical for local runs and teach the core AL loop: surrogate modeling, acquisition, and diversity-aware batch selection.

## 2) Goals
- Teach a full AL workflow using real bulk CRISPR data.
- Train on **single-gene** data from a single cell line, then recommend **gene pairs** to test for epistasis.
- Keep runtime reasonable on a laptop while retaining a realistic gene count (5k-20k genes).
- Provide a clear separation between heavy compute (Python script) and education/visualization (marimo notebook).

## 3) Non-goals
- Building a production-grade AL system or scalable distributed compute.
- Discovering true biological mechanisms beyond the tutorial example.
- Exhaustive hyperparameter optimization.

## 4) Target audience
- Computational biologists or data scientists new to active learning for CRISPR screens.
- Readers comfortable with Python, pandas, and basic ML concepts.

## 5) Default dataset choices
### 5.1 Single-gene training data (bulk)
- **Dataset:** DepMap/Achilles CRISPR gene effect (Avana)
- **Cell line:** K562
- **Size target:** ~10k-20k genes for one cell line
- **Use:** gene-level features for pair modeling; not pairwise labels

### 5.2 Pairwise epistasis data (evaluation oracle)
- **Dataset:** Horlbeck et al. 2018 CRISPRi genetic interaction map
- **Cell line:** K562
- **Use:** simulate wet-lab measurements during AL rounds and evaluate ranking quality

### 5.3 Rationale
- K562 exists in both datasets, enabling consistent biology.
- Avana is a standard bulk CRISPR benchmark.
- Horlbeck 2018 is a canonical pairwise GI dataset with public usage in the field.

## 6) Data access and layout
- Follow repo layout in `data/` and registry conventions in `manifests/registry.yaml`.
- Single-gene data target path:
  - `data/real/depmap_avana_gene_effect/vYYYY-MM-DD/`
- Pairwise data target path:
  - `data/real/horlbeck_2018_gi_k562/vYYYY-MM-DD/`

### Required files (per dataset version)
- `metadata.json`
- `checksums.sha256`
- Raw dataset file(s) in CSV/TSV format

## 7) Data schema (tutorial assumptions)
### 7.1 Single-gene table (K562 only)
Minimal columns expected for the tutorial:
- `gene`: gene symbol
- `effect`: gene effect score (bulk fitness)
- Optional: `mean_effect`, `std_effect`, `guide RNA ID` if provided

### 7.2 Pairwise table (K562 only)
Minimal columns expected for the tutorial:
- `gene_a`, `gene_b`: gene symbols
- `gi_score`: genetic interaction score
- Optional: `p_value`, `effect_observed`, `effect_expected`

## 8) Method defaults
### 8.1 Feature construction
- **Gene features** from the single-gene screen:
  - Primary: `effect`
  - Optional: co-essentiality or pathway priors if available
- **Pair features** derived from gene features:
  - Concatenation: `[g_a_features, g_b_features]`
  - Symmetric transforms: sum, product, absolute difference

### 8.2 Surrogate model
- **Default:** tree ensemble (RandomForest or LightGBM)
- **Uncertainty:** per-tree variance (std across ensemble predictions)

### 8.3 Acquisition function
- **Default:** UCB (mean + beta * std)
- Optional: qNEI as an alternative

### 8.4 Diversity picker (batch selection)
- **Default:** greedy k-center / max-min over pair embeddings
- Rationale: ensures batch coverage and reduces redundancy

## 9) Active learning loop (tutorial behavior)
1) Load K562 single-gene effects and build gene features.
2) Build candidate gene pairs (limited to genes with non-missing effect).
3) Seed an initial labeled set of pairs (small random batch from Horlbeck).
4) Train surrogate on labeled pairs using pair features.
5) Score all candidate pairs with acquisition function.
6) Apply diversity picker to form the next batch.
7) Query the oracle (Horlbeck) for labels; update the training set.
8) Repeat for N rounds; track metrics.

## 10) Tutorial deliverables
### 10.1 Heavy-compute script
- Location:
  - Entry point: `notebooks/bulk_single_dual/scripts/al_bulk_epistasis.py`
  - Core logic: `src/al_bulk_epistasis/`
- Rationale: keep a notebook-specific CLI in `notebooks/<subfolder>/scripts` while the reusable, testable pipeline lives in `src/`.
- Responsibilities:
  - Load and preprocess datasets
  - Build pair features and candidate set
  - Run AL loop and save results

### 10.2 Marimo notebook
- Location: `notebooks/bulk_single_dual/active_learning_bulk_epistasis_tutorial.py`
- Responsibilities:
  - Explain AL concepts and dataset rationale
  - Examine data structures of the various objects used in the analysis
  - Visualize pair scores, uncertainty, and diversity selections
  - Plot performance curves across rounds

### 10.3 Output artifacts
- Location:  `notebooks/bulk_single_dual/outputs/`
- Files:
  - `rounds.csv` (round, batch indices, acquisition scores)
  - `metrics.csv` (hit rate, top-k enrichment, GI correlation)
  - `selected_pairs.csv`

## 11) Evaluation and metrics
- **Top-k enrichment:** overlap of selected pairs with strongest GI scores
- **Hit rate:** fraction of selected pairs above GI threshold
- **Learning curve:** metric vs. AL round

## 12) Constraints and risks
- **Dataset access/licensing:** DepMap and Horlbeck downloads must be documented; add fallback notes.
- **Candidate pair explosion:** mitigate by pre-filtering genes (non-missing, optional high-variance) and limiting pair pool by only those with functional effect in single gene dataset.
- **No combinatorial synthetic dataset in repo:** tutorial relies on real data; note in documentation.

## 13) Acceptance criteria
- Script runs end-to-end on a single cell line with 10k-20k genes in reasonable time (seconds to an minutes).
- Notebook reproduces core plots and explains the AL loop clearly.
- Results artifacts are written under `notebooks/bulk_single_dual/outputs/` with deterministic seeds.

## 14) Milestones
1) Confirm dataset versions and download instructions.
2) Draft compute script skeleton and data schema checks.
3) Build marimo notebook outline and visualizations.
4) Validate end-to-end run on a small seed batch. Randomly select 100 genes for the validation run.

## 15) References
- DepMap/Achilles CRISPR gene effect (Avana)
- Horlbeck et al. 2018 CRISPRi genetic interaction map (K562)
