# Research Plan: Predictive Power of CRISPR Screens

## Overview

This research plan addresses two fundamental questions about the predictive power of CRISPR screens:

1. **Aim 1**: Can a focused ~2,000 gene CRISPR screen for venetoclax resistance/sensitivity predict genome-wide perturbation effects?
2. **Aim 2**: Can in vitro genome-wide CRISPR screens predict in vivo CRISPR screen results?

---

## Aim 1: Focused vs Genome-Wide CRISPR Screen Prediction (Venetoclax)

### Scientific Question

**If I run a focused ~2,000 gene CRISPR screen with venetoclax selection, how well can I predict what a genome-wide CRISPR screen with venetoclax would find?**

This has practical implications for:
- Cost-effective screen design
- Prioritizing genes for follow-up
- Understanding the information content of focused libraries

### Experimental Designs

We will test two complementary designs:

#### Design A: Within-Screen Holdout (Same Dataset)

**Concept**: Split a genome-wide venetoclax CRISPR screen into training (2,000 genes) and holdout (~16,000 genes) sets. Train a model on the 2,000 genes and predict the remaining genes.

```
Genome-wide venetoclax CRISPR screen
              │
              ▼
┌─────────────────────────────────────┐
│  ~18,000 genes with venetoclax     │
│  resistance/sensitivity scores      │
└─────────────────────────────────────┘
              │
         Split genes
              │
       ┌──────┴──────┐
       ▼             ▼
   Training       Holdout
   2,000 genes    ~16,000 genes
       │             │
       ▼             │
    Build            │
    Model ───────────┘
       │
       ▼
  Predict holdout gene scores
       │
       ▼
  Evaluate prediction accuracy
```

**What this tests**: Can biological features (expression, pathway, PPI networks) extrapolate from a focused screen to genome-wide predictions within the same experimental context?

**Strengths**:
- Same technical noise, same cell state
- Tests pure predictive power of features
- No confounding from batch effects

**Limitations**:
- Requires informative predictive features
- May not reflect real-world cross-study scenarios

#### Design B: Cross-Screen Transfer (Different Dataset)

**Concept**: Train on 2,000 randomly sampled genes from Chen 2019 → predict Sharon 2019 (and vice versa). Tests cross-study generalization between two independent genome-wide screens on related but distinct cell lines.

```
Chen 2019 (MOLM-13)             Sharon 2019 (MOLM-13-R1)
~19,115 genes                   ~17,237 genes
       │                              │
  Sample 2,000                   Ground truth
       │                              │
       ▼                              │
  Training data                       │
       └──────── Model ───────────────┘
                   │
                   ▼
       Evaluate cross-study transferability
```

**What this tests**: Can results from one venetoclax screen generalize to predict another independent screen across two related but distinct cell line contexts?

**Strengths**:
- Tests real-world scenario (use published data to predict new experiments)
- Accounts for biological reproducibility

**Limitations**:
- Confounded by technical and biological differences between datasets
- May underestimate true predictive power due to batch effects

### Data Sources

#### Primary Datasets for Venetoclax CRISPR Screens

| Study | Cell Lines | Library Size | Condition | Data Availability |
|-------|-----------|--------------|-----------|-------------------|
| Chen et al. 2019 | MOLM-13 | Genome-wide (~19,115 genes) | Venetoclax (mitochondrial targeting) | BioGRID ORCS |
| Sharon et al. 2019 | MOLM-13-R1 | Genome-wide (~17,237 genes) | Venetoclax (mitochondrial translation) | BioGRID ORCS |

#### Supporting Datasets

| Dataset | Description | Use |
|---------|-------------|-----|
| DepMap CRISPR | Baseline gene essentiality (~1,000 cell lines) | Feature engineering, co-essentiality |
| PRISM | Venetoclax drug sensitivity across cell lines | Validation, cell line selection |
| BeatAML 2.0 | Venetoclax response in primary AML | Clinical relevance |
| CCLE Expression | Gene expression across cell lines | Predictive features |

### Gene Selection Strategy

**Primary approach**: Random sampling (baseline)

For each iteration:
1. Randomly sample 2,000 genes from the genome-wide screen
2. Use these as the "focused library" training set
3. Predict remaining ~16,000 genes
4. Repeat 100 times to generate confidence intervals

This establishes the baseline expectation for any focused library and allows comparison to more sophisticated selection strategies in future work.

### Predictive Features (for extrapolation)

When predicting holdout genes, we will use:

1. **Gene expression** (CCLE): Expression level in the cell line
2. **Pathway membership**: GO terms, KEGG, Reactome
3. **Co-essentiality**: Correlation with training genes in DepMap
4. **Gene properties**: Length, GC content, number of isoforms
5. **Functional annotations**: Essential gene status, druggability

### Evaluation Metrics

#### Correlation Metrics
- **Pearson correlation**: Linear relationship between predicted and actual scores
- **Spearman correlation**: Rank-based relationship (robust to outliers)
- **R²**: Variance explained by the model

#### Hit Identification Metrics
- **Precision@K**: Of top K predicted hits, how many are true hits?
- **Recall@K**: Of true top K hits, how many were predicted?
- **AUROC**: Area under ROC curve for classifying sensitizers vs resistors
- **AUPRC**: Area under precision-recall curve (better for imbalanced data)

#### Thresholds for "Hits"
- **Sensitizers**: Genes where knockout increases venetoclax sensitivity (negative CRISPR score under drug)
- **Resistors**: Genes where knockout decreases venetoclax sensitivity (positive CRISPR score under drug)
- **Threshold**: Top/bottom 5% of genes, or FDR < 0.05

### Analysis Pipeline

```
Step 1: Data Acquisition
├── Download genome-wide venetoclax CRISPR screen data
├── Download supporting datasets (DepMap, CCLE, PRISM)
└── Harmonize gene identifiers

Step 2: Data Preprocessing
├── Calculate gene-level scores from sgRNA data (if needed)
├── Normalize scores (z-score within screen)
├── Filter low-quality genes (low read counts, high variance)
└── Merge with feature datasets

Step 3: Design A - Within-Screen Holdout
├── For each of 100 iterations:
│   ├── Randomly sample 2,000 genes (training)
│   ├── Remaining genes = holdout (test)
│   ├── Train model (Ridge Regression, Random Forest)
│   ├── Predict holdout gene scores
│   └── Calculate metrics (correlation, precision-recall)
├── Aggregate results across iterations
└── Generate confidence intervals

Step 4: Design B - Cross-Screen Transfer
├── Sample 2,000 genes from Chen 2019 (training screen)
├── Define test screen (Sharon 2019, genome-wide)
├── For overlapping genes:
│   └── Calculate direct correlation (no model needed)
├── For non-overlapping genes:
│   ├── Train model on Chen 2019 training genes
│   ├── Predict Sharon 2019 scores
│   └── Evaluate predictions
├── Repeat with Chen↔Sharon swapped
└── Compare to random baseline

Step 5: Visualization & Interpretation
├── Scatter plots: predicted vs actual scores
├── Precision-recall curves
├── ROC curves
├── Distribution of correlations across iterations
└── Identify systematic prediction failures
```

### Expected Outputs

1. **Correlation estimates**: Mean ± SD correlation between focused and genome-wide screens
2. **Precision-recall curves**: For identifying top sensitizers/resistors
3. **Feature importance**: Which features best predict venetoclax response?
4. **Failure analysis**: Which genes are systematically mispredicted?

---

## Aim 2: In Vitro vs In Vivo CRISPR Screen Prediction

### Scientific Question

**How well do in vitro genome-wide CRISPR screens predict in vivo dependencies?**

This addresses the translational gap between cell culture and animal models, with implications for:
- Target validation strategies
- Prioritizing hits for in vivo follow-up
- Understanding context-dependent gene function

### Data Sources

We will use published studies with **paired in vitro and in vivo screens** from the same cell line:

| Study | Cancer Type | Cell Line | Library Size | In Vivo Model | Data Availability |
|-------|-------------|-----------|--------------|---------------|-------------------|
| Lin et al. 2022 | AML | MV4-11, U937, PDX16-01 | 200 genes (1,320 sgRNAs) | Orthotopic xenograft | Supplementary Tables |
| CRISPR-StAR (Elling et al. 2024) | Melanoma | Yumm1.7 450R | ~22,000 genes | Subcutaneous tumor | GEO: GSE262309 |
| TNBC Paclitaxel (Yan et al. 2023) | Breast | SUM159 | ~19,000 genes | Subcutaneous xenograft | Supplementary Files |
| MEN1 Study (He et al. 2024) | Lung | A549 | ~1,000 genes | Xenograft | GEO: GSE194349 |

### Analysis Approach

#### Step 1: Data Harmonization
- Download paired in vitro/in vivo screen data
- Extract gene-level scores (log fold change or MAGeCK scores)
- Normalize within each screen (z-score or rank-based)
- Align gene identifiers across studies

#### Step 2: Agreement Metrics

**Global correlation**:
- Pearson and Spearman correlation between in vitro and in vivo scores
- Calculated per study and aggregated across studies

**Concordance at the top (CAT)**:
- For top N hits in vitro, what fraction are also top N in vivo?
- Plot CAT curves for N = 10, 50, 100, 500

**Hit overlap**:
- Jaccard similarity for significant hits (FDR < 0.05)
- Separate analysis for sensitizers and resistors

#### Step 3: Identify Discordant Genes

**Categories**:
- **In vivo-specific**: Essential in vivo but NOT in vitro
- **In vitro-specific**: Essential in vitro but NOT in vivo
- **Concordant**: Essential (or non-essential) in both

**Threshold**: |Δscore| > 1 SD AND opposite direction OR significant in one but not other

#### Step 4: Pathway Enrichment of Discordant Genes

Hypothesized enrichments based on literature:
- **In vivo-specific**: Immune-related, hypoxia, nutrient stress, ECM interactions
- **In vitro-specific**: Cell cycle (faster proliferation in vitro), DNA damage response

Tools: clusterProfiler, Enrichr, GSEA

#### Step 5: Predictive Modeling

**Goal**: Predict which in vitro hits will validate in vivo

**Features**:
- Gene expression level
- Pathway membership (immune, metabolism, TME-related)
- Protein localization (membrane, secreted, nuclear)
- DepMap essentiality (common essential vs selective)
- In vitro effect size

**Model**: Logistic regression or Random Forest

**Evaluation**: Cross-validation across studies (train on 3 studies, test on 1)

### Expected Outputs

1. **Correlation summary table**: In vitro vs in vivo correlation per study
2. **Scatter plots**: In vitro vs in vivo scores with discordant genes highlighted
3. **Discordant gene lists**: Annotated with pathways and functions
4. **Pathway enrichment**: For in vivo-specific and in vitro-specific genes
5. **Predictive model**: ROC curve for predicting in vivo validation

---

## Summary Table

| Aspect | Aim 1 | Aim 2 |
|--------|-------|-------|
| **Question** | Focused → Genome-wide prediction | In vitro → In vivo prediction |
| **Context** | Venetoclax CRISPR screens | General CRISPR screens |
| **Design A** | Within-screen holdout | - |
| **Design B** | Cross-screen transfer | Cross-condition comparison |
| **Gene selection** | Random sampling (baseline) | All genes |
| **Primary metrics** | Correlation, Precision-Recall | Correlation, Discordance analysis |
| **Key output** | Predictive power of focused screens | Context-dependent dependencies |

---

## Timeline Estimate

| Step | Aim 1 | Aim 2 |
|------|-------|-------|
| Data download | 1-2 hours | 2-3 hours |
| Preprocessing | 2-3 hours | 2-3 hours |
| Main analysis | 4-5 hours | 4-5 hours |
| Visualization | 2-3 hours | 2-3 hours |
| **Total** | **~10-13 hours** | **~10-14 hours** |

---

## References

1. Chen X, et al. (2019). Targeting Mitochondrial Structure Sensitizes Acute Myeloid Leukemia to Venetoclax Treatment. Cancer Discovery. BioGRID ORCS.
2. Sharon D, et al. (2019). Inhibition of mitochondrial translation overcomes venetoclax resistance in AML through activation of the integrated stress response. Science Translational Medicine. BioGRID ORCS.
3. Lin S, et al. (2022). An In Vivo CRISPR Screening Platform for Prioritizing Therapeutic Targets in AML. Cancer Discovery.
4. Uijttewaal ECH, et al. (2024). CRISPR-StAR enables high-resolution genetic screening in complex in vivo models. Nature Biotechnology. GEO: GSE262309
5. Yan G, et al. (2023). Combined in vitro/in vivo genome-wide CRISPR screens in triple negative breast cancer identify cancer stemness regulators in paclitaxel resistance. Oncogenesis.
6. He S, et al. (2024). In vivo CRISPR screens identify a dual function of MEN1 in regulating tumor–microenvironment interactions. Nature Genetics. GEO: GSE194349

---

## Next Steps

Upon approval of this plan:
1. Download all required datasets
2. Execute Aim 1 analysis (Design A and B)
3. Execute Aim 2 analysis
4. Generate summary figures and conclusions
5. Compile final report with recommendations
