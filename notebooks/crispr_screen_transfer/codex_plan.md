# Instructions

This plan operationalizes `research_plan_crispr_screen_prediction.md` into an
execution-ready workflow for evaluating CRISPR screen transferability.

## 0. Objectives

1. Aim 1 (Venetoclax): quantify how well focused CRISPR screens predict
   genome-wide venetoclax effects.
2. Aim 2 (Context transfer): quantify how well in vitro genome-wide screens
   predict in vivo dependencies.

Primary success criteria:

- Stable cross-run estimates with confidence intervals.
- Strong rank recovery of top sensitizers/resistors.
- Clear characterization of discordant in vitro vs in vivo genes.

## 1. Data Specification

### 1.1 Aim 1 Datasets (Venetoclax)

| Study | Cell lines | Library size | Condition | Access |
|---|---|---|---|---|
| Chen et al. 2019 | MOLM-13 | Genome-wide (~19,115 genes) | Venetoclax | BioGRID ORCS |
| Sharon et al. 2019 | MOLM-13-R1 | Genome-wide (~17,237 genes) | Venetoclax | BioGRID ORCS |

Supporting features:

- DepMap CRISPR essentiality
- CCLE expression
- PRISM drug response
- BeatAML 2.0 venetoclax response

### 1.2 Aim 2 Datasets (In Vitro vs In Vivo)

| Study | Cancer type | Cell line/model | Library size | In vivo model | Access |
|---|---|---|---|---|---|
| Lin et al. 2022 | AML | MV4-11, U937, PDX16-01 | 200 genes | Orthotopic xenograft | Supplementary tables |
| CRISPR-StAR (Elling et al. 2024) | Melanoma | Yumm1.7 450R | ~22,000 genes | Subcutaneous tumor | GEO: GSE262309 |
| Yan et al. 2023 | TNBC | SUM159 | ~19,000 genes | Subcutaneous xenograft | Supplementary files |
| He et al. 2024 (MEN1) | Lung | A549 | ~1,000 genes | Xenograft | GEO: GSE194349 |

### 1.3 Required Harmonized Schema

Minimum per-screen gene-level table:

- `gene_id`
- `screen_id`
- `context` (`in_vitro`/`in_vivo`/`venetoclax`)
- `score_raw`
- `score_norm` (z-score or rank-normalized)
- `is_hit_sensitizer`
- `is_hit_resistor`

Additional metadata fields (recommended for transfer benchmarking):

- `study_id`
- `model_system` (cell line, xenograft, PDX)
- `tissue`
- `drug_condition`
- `library_type` (focused or genome-wide)
- `split_group` (used to enforce leakage-safe train/test partitions)

### 1.4 Transfer Split Registry (State-Inspired)

Define all evaluation splits in a machine-readable registry (`splits.yaml` or
`splits.toml`) to make runs reproducible and comparable.

Required split families:

1. `random_gene_holdout`:
   Design A repeated random holdout for Aim 1.
2. `context_zeroshot`:
   train on all in vitro studies except target context, test on held-out in
   vivo context.
3. `study_zeroshot`:
   leave-one-study-out transfer for Aim 2.
4. `fewshot_context`:
   adapt to a held-out context with small labeled support sets (for example
   25/50/100 genes) and evaluate on remaining genes.

Leakage controls:

- no shared gene-label rows between train and test for a split;
- no derived normalization statistics computed on held-out test rows;
- log split hash and random seed for every run.

## 2. Aim 1: Focused -> Genome-Wide Venetoclax Prediction

### 2.1 Task Formulation

Let `y_g` be the normalized venetoclax effect for gene `g`, and let `x_g` be
the feature vector (expression, pathway, PPI, co-essentiality, annotations).
Learn a model `f` such that `f(x_g) ~ y_g`.

### 2.2 Design A (Within-Screen Holdout)

For each iteration `r` in `{1, ..., 100}`:

1. Randomly sample `|G_train^(r)| = 2000` genes from one
   genome-wide screen.
2. Set holdout genes
   `G_test^(r) = G_all - G_train^(r)` (typically `~16000` genes).
3. Train model on `G_train^(r)`, predict on `G_test^(r)`.
4. Aggregate metrics across `r` with mean and confidence intervals.

### 2.3 Design B (Cross-Screen Transfer)

Train on 2,000 randomly sampled genes from Chen 2019 → predict Sharon 2019
(and vice versa). Tests cross-study generalization between two independent
genome-wide screens on related cell lines (MOLM-13 vs MOLM-13-R1):

1. Overlap-only baseline: direct correlation on intersecting genes.
2. Non-overlap transfer: train on Chen 2019 sample, predict Sharon 2019 genes
   not in training set, using shared feature space.
3. Repeat with Chen↔Sharon swapped.
4. Compare to random baseline from Design A.

### 2.4 Feature Set

Use:

1. CCLE expression
2. Pathway membership (GO/KEGG/Reactome)
3. DepMap co-essentiality features
4. Gene properties (length, GC, isoform count)
5. Functional annotations (essentiality class, druggability)

### 2.5 Baseline Models

- Ridge regression
- Random forest regressor

Optional extension:

- Gradient boosting (XGBoost/LightGBM) for nonlinearity checks.

### 2.6 Metrics (Machine-Readable)

Correlation/regression:

$$
r_{\mathrm{Pearson}} = \mathrm{corr}(y, \hat{y}), \quad
\rho_{\mathrm{Spearman}} = \mathrm{corr}(\mathrm{rank}(y), \mathrm{rank}(\hat{y}))
$$

$$
R^2 = 1 - \frac{\sum_g (y_g-\hat{y}_g)^2}{\sum_g (y_g-\bar{y})^2}
$$

Top-hit recovery:

$$
\mathrm{Precision@}K
=
\frac{| \mathrm{TopK}(\hat{y}) \cap \mathrm{TopK}(y) |}{K}
$$

$$
\mathrm{Recall@}K
=
\frac{| \mathrm{TopK}(\hat{y}) \cap \mathrm{TopK}(y) |}{|\mathrm{TopK}(y)|}
$$

Classification quality:

- AUROC for sensitizer/resistor classification
- AUPRC for class-imbalanced hit recovery

Hit definitions:

- Sensitizers: lower-tail venetoclax scores
- Resistors: upper-tail venetoclax scores
- Primary threshold: top/bottom 5% (alternative: FDR \(<0.05\))

### 2.7 Transfer Regimes and Baseline Ladder

To separate interpolation from true transfer, report a baseline ladder:

1. `naive_global_mean`:
   predict by global mean/rank.
2. `overlap_only`:
   evaluate only intersecting genes (direct transfer sanity check).
3. `feature_transfer_zeroshot`:
   no target labels; predict with shared features only.
4. `feature_transfer_fewshot`:
   fine-tune/calibrate with small target support sets.

For `fewshot` settings, report performance as a function of support size
(`n_support` curve), with 95% CIs across seeds.

## 3. Aim 2: In Vitro -> In Vivo Prediction

### 3.1 Harmonization

For each paired study:

1. Extract gene-level effect scores (LFC/MAGeCK or reported score).
2. Normalize within screen (z-score or rank-based).
3. Align to unified gene IDs.

### 3.2 Agreement Metrics

Global agreement:

$$
r_{\mathrm{vitro,vivo}} = \mathrm{corr}(y^{\mathrm{vitro}}, y^{\mathrm{vivo}})
$$

Concordance-at-the-top (CAT):

$$
\mathrm{CAT}(N)
=
\frac{| \mathrm{Top}N(y^{\mathrm{vitro}}) \cap \mathrm{Top}N(y^{\mathrm{vivo}}) |}{N}
$$

Hit overlap:

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

with \(A\), \(B\) as significant hit sets (for example FDR \(<0.05\)).

### 3.3 Discordance Rules

Define discordant genes using:

$$
\Delta_g = y^{\mathrm{vivo}}_g - y^{\mathrm{vitro}}_g
$$

Classify as discordant if:

- \(|\Delta_g| > 1\) SD and opposite-direction effects, or
- significant in one context but not the other.

Categories:

- In vivo-specific
- In vitro-specific
- Concordant

### 3.4 Predictive Model for In Vivo Validation

Goal: predict whether an in vitro hit validates in vivo.

Features:

- expression
- pathway classes (immune/metabolism/TME)
- protein localization
- DepMap essentiality pattern
- in vitro effect size

Models:

- Logistic regression
- Random forest classifier

Evaluation:

- Leave-one-study-out CV (train on 3, test on 1)
- AUROC/AUPRC and calibration

### 3.5 Context Transfer Matrix (State-Inspired)

Construct a transfer matrix where each row is a train context/study and each
column is a test context/study. Each cell stores:

- Pearson/Spearman
- CAT@N
- Jaccard of hit sets
- AUROC/AUPRC (if framed as hit classification)

Include both:

- `zeroshot` (no target labels)
- `fewshot` (small adaptation sets)

This matrix is the primary artifact for comparing where transfer succeeds or
fails.

### 3.6 Optional Active-Learning Extension (NAIAD-Inspired)

Optional prospective module after retrospective benchmarking:

1. Train ensemble on current labeled genes.
2. Score unlabeled genes by acquisition rule (for example uncertainty or
   uncertainty-weighted expected gain).
3. Select top `B` genes for next validation batch.
4. Refit model and track gain in CAT@N / hit-recovery vs random selection.

Use this only as a follow-on objective; keep core aims fully retrospective.

## 4. Execution Workflow

1. Data acquisition:
   download all primary/supporting datasets; log source versions.
2. Preprocessing:
   compute gene-level scores if needed, normalize per screen, harmonize IDs,
   quality filters.
3. Split registry:
   materialize and version all `random`, `zeroshot`, and `fewshot` splits with
   fixed seeds and split hashes using
   `notebooks/crispr_screen_transfer/splits.yaml`.
4. Aim 1 Design A:
   run 100 random 2000-gene sampling iterations.
5. Aim 1 Design B:
   run cross-screen transfer analysis with overlap and non-overlap subsets.
6. Aim 2:
   compute agreement/discordance metrics and pathway enrichment of discordant
   genes.
7. Predictive modeling:
   run classifier for in vivo validation likelihood.
8. Benchmark packaging:
   aggregate per-split metrics into transfer matrices and `fewshot` learning
   curves; compute mean, CI, and seed variance; validate each metrics artifact
   against `notebooks/crispr_screen_transfer/metrics.schema.json`.
9. Reporting:
   compile summary tables, CI plots, PR/ROC, CAT curves, and failure cases.

## 5. Deliverables

1. Correlation summary tables (Aim 1 and Aim 2).
2. Precision/recall and ROC/PR curves for hit recovery.
3. CAT and Jaccard overlap analyses.
4. Discordant gene lists with pathway enrichment annotations.
5. Feature-importance summaries for regression/classification models.
6. Reproducible run artifacts (seeds, configs, metrics, plots).
7. Transfer matrix heatmaps (`zeroshot` and `fewshot`) with per-cell metrics.
8. `n_support` adaptation curves showing marginal benefit of few-shot labels.
9. Optional active-learning gain curves versus random query baseline.

## 6. Timeline Estimate

| Step | Aim 1 | Aim 2 |
|---|---|---|
| Data download | 1-2 h | 2-3 h |
| Preprocessing | 2-3 h | 2-3 h |
| Main analysis | 4-5 h | 4-5 h |
| Visualization/reporting | 2-3 h | 2-3 h |
| Total | ~10-13 h | ~10-14 h |

## 7. Data Verification Findings (2026-03-08)

These findings were confirmed by inspecting the actual downloaded files and must
inform preprocessing and feature engineering decisions.

### 7.1 Gene Overlap (Design B scope)

| Metric | Value |
|--------|-------|
| Chen 2019 unique genes | 19,109 |
| Sharon 2019 unique genes (screen 1401) | 17,230 |
| Shared genes (Design B operating set) | **17,091** |
| Chen-only genes | 2,018 |
| Sharon-only genes | 139 |

Design B cross-screen transfer operates on the ~17,000-gene shared space.
The 2,018 Chen-only genes are excluded from Design B evaluation.

### 7.2 Gene Coverage in DepMap CRISPR (co-essentiality features)

| Screen | Genes in DepMap 25Q3 (18,435 total) | Coverage |
|--------|--------------------------------------|----------|
| Chen 2019 | 17,568 / 19,109 | 92% |
| Sharon 2019 | 16,360 / 17,230 | 95% |

Genes absent from DepMap will have missing co-essentiality features.
Impute with zero correlation or drop from co-essentiality feature only
(retain other features).

### 7.3 MOLM-13 ACH ID — Corrected

**MOLM-13 is present in both DepMap and CCLE under ACH-000362** (stripped name:
`MOLM13`). The historical ID ACH-001187 is incorrect — do not use it.

MOLM-13-R1 (the resistant line used in Sharon 2019) is not a separate entry in
DepMap/CCLE; it is a lab-derived subline not independently profiled. Use MOLM-13
(ACH-000362) as the expression and co-essentiality reference for both screens.

Full AML line inventory in CCLE + DepMap is in
`data/bulk/depmap_model_metadata/Model.csv.gz` (2,116 models total, 63 AML).
AML lines present in both CCLE expression and DepMap CRISPR (for co-essentiality
context if ever needed): HEL, HEL9217, MV411, OCIAML2, THP1, NOMO1, HDMYZ,
SET2, EOL1, KASUMI1, NB4, OCIAML3, MOLM13, SKM1, TF1, U937, F36P, KO52,
AML193, M07E, P31FUJ, CMK115, MONOMAC1, MOLM14, MUTZ8, OCIAML4, OCIM2,
SHI1, SKNO1, CMS.

### 7.4 Gene ID Formats — Alignment Required

| Dataset | Gene ID format | Example |
|---------|---------------|---------|
| BioGRID ORCS (Chen, Sharon) | `OFFICIAL_SYMBOL` (HGNC) + `IDENTIFIER_ID` (Entrez) | `BCL2`, `596` |
| DepMap CRISPR gene effect | Column header: `SYMBOL (ENTREZ_ID)` | `BCL2 (596)` |
| CCLE expression | Column header: `SYMBOL (ENTREZ_ID)` | `BCL2 (596)` |
| Reactome (`NCBI2Reactome_PE_Pathway.txt`) | Entrez Gene ID in col 1 | `596` |
| GOA (`goa_human.gaf.gz`) | UniProt accession; gene symbol in col 3 | `P10415` / `BCL2` |
| MSigDB GMT files | HGNC symbol | `BCL2` |

**Canonical key for joining:** HGNC symbol (strip ` (ENTREZ)` suffix from
DepMap/CCLE column names; use `OFFICIAL_SYMBOL` from BioGRID ORCS).
Fall back to Entrez ID for any symbol mismatches.

### 7.5 Data Files on Disk

All files gzipped. Located under `data/bulk/`:

| Path | Size | Release |
|------|------|---------|
| `depmap_crispr_gene_effect/CRISPRGeneEffect.csv.gz` | 186 MB | DepMap 25Q3 |
| `ccle_expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv.gz` | 231 MB | DepMap 25Q2 |
| `pathway_annotations/NCBI2Reactome_PE_Pathway.txt.gz` | 7 MB | Reactome Nov 2024 |
| `pathway_annotations/goa_human.gaf.gz` | 15 MB | GOA Jan 2025 |
| `pathway_annotations/h.all.v2024.1.Hs.symbols.gmt.gz` | 21 KB | MSigDB 2024.1 |
| `pathway_annotations/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt.gz` | 26 KB | MSigDB 2024.1 |
| `chen2019_venetoclax/BIOGRID-ORCS-SCREEN_139{2,3}-*.txt` | — | BioGRID ORCS 2.0.18 |
| `sharon2019_venetoclax/BIOGRID-ORCS-SCREEN_140{1,2,3,4}-*.txt` | — | BioGRID ORCS 2.0.18 |

## 8. References

- [Research Plan Source](./research_plan_crispr_screen_prediction.md)
- [CRISPR-StAR (GSE262309)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262309)
- [MEN1 Study (GSE194349)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194349)
- [STATE Framework](https://github.com/ArcInstitute/state)
- [NAIAD Framework](https://github.com/NeptuneBio/NAIAD)
