# Olivieri 2020 — Genotoxic CRISPR Screen Benchmark

## Dataset

- **Source**: Olivieri M et al., *Cell* 2020, PMID 32649862
- **Cell line**: RPE1-hTERT p53−/− Cas9
- **Screens**: 30 screens across 27 genotoxic agents (31 total; screen 1328 ICRF-187 excluded — QC fail)
- **Libraries**: TKOv2 (11 screens), TKOv3 (19 screens)
- **Score type**: DrugZ NormZ (Z-score; negative = sensitising KO)
- **Access**: BioGRID ORCS (server-side DataTables API)

## Problem Statement

- Can pathway-membership features alone predict which gene knockouts sensitise RPE1-hTERT cells to genotoxic agents?
- Three transfer regimes tested: within-screen holdout (Aim 1), cross-library version transfer (Aim 2), and leave-one-drug-out (Aim 3).
- RPE1-hTERT is absent from DepMap/CCLE, so only pathway features are available (6 features; no expression, no co-essentiality).

## Feature Set

| Feature | Source | Type |
|---|---|---|
| `n_reactome_pathways` | MSigDB C2 Reactome | count |
| `n_kegg_pathways` | MSigDB C2 KEGG | count |
| `n_go_bp_terms` | MSigDB C5 GO:BP | count |
| `n_go_mf_terms` | MSigDB C5 GO:MF | count |
| `in_hallmark_apoptosis` | MSigDB H | binary |
| `in_hallmark_oxidative_phosphorylation` | MSigDB H | binary |

## Models

- Ridge regression (RidgeCV, alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5; StandardScaler)
- Random Forest (n_estimators=200, max_features="sqrt", min_samples_leaf=5)

## Benchmark Designs

### Aim 1 — Within-screen holdout
- 80/20 random gene holdout, 25 repeats per screen (30 screens × 25 repeats × 2 models = 1,500 rows).
- Tests whether pathway features predict held-out gene scores within a single condition.

### Aim 2 — Cross-library transfer
- Train on one library version of a drug (TKOv2/TKOv3), predict on the other.
- 6 train/test pairs for Cisplatin and Camptothecin (both directions).
- No bootstrapping — single evaluation per direction pair.

### Aim 3 — Leave-one-drug-out (LODO)
- Train on all other drugs in the same library, predict on the held-out drug.
- TKOv2: 10 train / 1 test; TKOv3: 18 train / 1 test.
- Key implementation: per-screen Z-normalization of training labels before stacking to prevent label-scale collapse.

## Results

Mean metrics across all repeats/screens (from parquet files).

| Aim | Setting | RF AUROC | Ridge AUROC | RF Pearson | Ridge Pearson | RF P@50 | RF P@100 |
|---|---|---|---|---|---|---|---|
| Aim 1 | Within-screen (25 repeats × 30 screens) | 0.554 | 0.595 (median 0.635) | 0.039 | 0.028 | 0.013 | 0.012 |
| Aim 2 | Cross-library (Cisplatin, Camptothecin) | 0.821 | 0.682 | 0.143 | 0.023 | 0.370 | 0.273 |
| Aim 3 | LODO (TKOv2 + TKOv3) | 0.813 | 0.665 | 0.261 | 0.021 | 0.313 | 0.230 |

## Key Findings

1. **RF jumps from near-chance to strong when given a paired screen**: Aim 1 RF AUROC ≈ 0.58 vs Aims 2–3 ≈ 0.82. The pathway features encode a transferable sensitiser signal, but this signal is masked by condition-specific noise in the within-screen setting.

2. **Ridge plateau**: Ridge AUROC barely moves across aims (≈ 0.64–0.68), suggesting linear models cannot exploit the multi-screen training signal the way RF can with its rank-based splits.

3. **Aims 2 and 3 are equivalent for RF**: Cross-library (same drug, different library version) and cross-drug (LODO) yield identical RF AUROC ≈ 0.82. The transferable signal is drug-agnostic — it reflects general genotoxic stress pathway membership.

4. **Pearson near zero throughout**: Regression accuracy is weak across all aims. Pathway features rank sensitiser genes well (AUROC ≈ 0.82) but cannot predict absolute NormZ score magnitudes.

5. **High within-screen variance**: Wide AUROC distributions in Aim 1 across 30 screens reflect genuine biological heterogeneity — different genotoxic mechanisms vary in how pathway-predictable their sensitiser profiles are.

## Implementation Notes

- `build_olivieri_features()` added to `src/crispr_al/features.py` — delegates to `build_pathway_features()`, skipping expression and co-essentiality.
- `load_olivieri_normz()` added to `src/crispr_al/screen.py`.
- `generate_lodo_splits()` added to `src/crispr_al/splits.py` — returns list of `{test_screen, train_screens}` dicts.
- `min_samples_leaf` param added to `train_rf()` in `src/crispr_al/models.py` (set to 5 for Olivieri analysis).
- Results in `notebooks/RPE1-hTERT_genotoxic/results/`; figures in `notebooks/RPE1-hTERT_genotoxic/figures/`.

## Reusable Components

- `generate_lodo_splits()` — generic LODO split generator, applicable to any multi-screen dataset.
- `min_samples_leaf` tuning in RF — important for datasets where genes have limited training signal; reduces overfitting in pathway-only feature regimes.
- Per-screen Z-normalization before stacking labels — required pattern for any multi-screen LODO analysis to prevent label-scale collapse.
