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
| ZNF740 (Zhang et al. 2024) | OCI-AML2, MOLM-13 | 1,426 genes (7 sgRNAs/gene) | Venetoclax vs DMSO | GEO: GSE267342 |
| Chen et al. 2019 | AML cells | Genome-wide | Venetoclax | BioGRID ORCS |
| Sharon et al. 2019 | MOLM-13 | Genome-wide | Venetoclax | BioGRID ORCS |

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

## 2. Aim 1: Focused -> Genome-Wide Venetoclax Prediction

### 2.1 Task Formulation

Let \(y_g\) be normalized venetoclax effect for gene \(g\), and \(x_g\) the
feature vector (expression, pathway, PPI, co-essentiality, annotations).
Learn \(f(x_g) \approx y_g\).

### 2.2 Design A (Within-Screen Holdout)

For each iteration \(r \in \{1,\dots,100\}\):

1. Randomly sample \(|G_{\mathrm{train}}^{(r)}|=2000\) genes from one
   genome-wide screen.
2. Set holdout genes
   \(G_{\mathrm{test}}^{(r)} = G \setminus G_{\mathrm{train}}^{(r)}\)
   (typically \(\sim16000\)).
3. Train model on \(G_{\mathrm{train}}^{(r)}\), predict on
   \(G_{\mathrm{test}}^{(r)}\).
4. Aggregate metrics across \(r\) with mean and confidence intervals.

### 2.3 Design B (Cross-Screen Transfer)

Train on focused Screen A (for example ZNF740) and evaluate on genome-wide
Screen B (for example Sharon which is the same cell line):

1. Overlap-only baseline: direct correlation on intersecting genes.
2. Non-overlap transfer: train on A, predict B with shared feature space.
3. Compare to random baseline from Design A.

### 2.4 Feature Set

Use:

1. CCLE expression
2. Pathway membership (GO/KEGG/Reactome)
3. STRING PPI neighborhood features
4. DepMap co-essentiality features
5. Gene properties (length, GC, isoform count)
6. Functional annotations (essentiality class, druggability)

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

## 4. Execution Workflow

1. Data acquisition:
   download all primary/supporting datasets; log source versions.
2. Preprocessing:
   compute gene-level scores if needed, normalize per screen, harmonize IDs,
   quality filters.
3. Aim 1 Design A:
   run 100 random 2000-gene sampling iterations.
4. Aim 1 Design B:
   run cross-screen transfer analysis with overlap and non-overlap subsets.
5. Aim 2:
   compute agreement/discordance metrics and pathway enrichment of discordant
   genes.
6. Predictive modeling:
   run classifier for in vivo validation likelihood.
7. Reporting:
   compile summary tables, CI plots, PR/ROC, CAT curves, and failure cases.

## 5. Deliverables

1. Correlation summary tables (Aim 1 and Aim 2).
2. Precision/recall and ROC/PR curves for hit recovery.
3. CAT and Jaccard overlap analyses.
4. Discordant gene lists with pathway enrichment annotations.
5. Feature-importance summaries for regression/classification models.
6. Reproducible run artifacts (seeds, configs, metrics, plots).

## 6. Timeline Estimate

| Step | Aim 1 | Aim 2 |
|---|---|---|
| Data download | 1-2 h | 2-3 h |
| Preprocessing | 2-3 h | 2-3 h |
| Main analysis | 4-5 h | 4-5 h |
| Visualization/reporting | 2-3 h | 2-3 h |
| Total | ~10-13 h | ~10-14 h |

## 7. References

- [Research Plan Source](./research_plan_crispr_screen_prediction.md)
- [ZNF740 Study (GSE267342)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267342)
- [CRISPR-StAR (GSE262309)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262309)
- [MEN1 Study (GSE194349)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194349)
