# PRD: Active Learning Tutorial for Bulk CRISPR Epistasis with Overparameterized Additive Baseline (K562)

- **Status:** Draft
- **Updated:** 2026-02-07
- **Owner:** OpenCode
- **Location:** `notebooks/bulk_single_dual/`

---

## 1. Problem Statement

We need a lightweight, reproducible tutorial that demonstrates how **active learning (AL)** can be applied to prioritize **gene pairs** for CRISPR epistasis screening, while **explicitly conditioning predictions on known single-gene effects**.

The tutorial should illustrate how an **overparameterized additive baseline**, inspired by the **NAIAD method**, provides a strong inductive bias for modeling pairwise CRISPR screens before learning nonlinear genetic interactions.

---

## 2. Goals

- Teach a complete AL workflow using real bulk CRISPR data
- Assume undergraduate level knowledge
- Use **single-gene knockout effects** to construct an additive baseline
- Demonstrate how **overparameterized encodings** capture additive structure
- Learn **nonlinear epistatic deviations** on top of this baseline
- Keep runtime feasible on a laptop
- Separate heavy compute (script) from explanation and visualization (notebook)

---

## 3. Non-goals

- Production-grade deep learning systems
- Genome-wide pair enumeration
- Reproducing the full NAIAD architecture
- Exhaustive hyperparameter tuning

---

## 4. Target Audience

- Unsergraduate level maths
- Computational biologists and ML practitioners
- Familiar with Python, basic neural networks, and CRISPR screens
- Interested in inductive biases for biological modeling

---

## 5. Datasets

### 5.1 Single-Gene CRISPR Data (Conditioning Signal)

- **Dataset:** DepMap CRISPR Gene Effect (Chronos)
- **Cell line:** K562
- **Genes loaded:** genome-wide (~17k)
- **Usage:** provides scalar perturbation effects \( Y_i \)

### 5.2 Pairwise Epistasis Oracle

- **Dataset:** Horlbeck et al. 2018 CRISPRi GI Map
- **Cell line:** K562
- **Genes covered:** approximately 472
- **Usage:** oracle for querying pairwise GI labels during AL

### 5.3 Candidate Gene Set

```
candidate_genes =
    genes_with_non_missing_DepMap_effect
    ∩ genes_present_in_Horlbeck_GI_map
```

Resulting in approximately 472 genes and approximately 111k candidate pairs.

---

## 6. Data Layout

Raw datasets are not stored in the repository.

```
data/
  real/
    depmap_crispr_gene_effect/
      vDepMapPublic_XXQY/
    horlbeck_2018_gi_k562/
      v1/
```

Each dataset version includes:

- `metadata.json`
- `checksums.sha256`
- Raw CSV/TSV files

---

## 7. Data Schema

### 7.1 Single-Gene Table (K562)

Required:

- `gene_name`
- `effect` (scalar phenotype \( Y_i \))

### 7.2 Pairwise GI Table

Required:

- `gene_a`
- `gene_b`
- `gi_score`

Pairs are canonicalized as:

```
(gene_a, gene_b) = (min(gene_a, gene_b), max(gene_a, gene_b))
```

---

## 8. Perturbation Encoding: Overparameterized Additive Baseline (NAIAD-style)

### 8.1 Motivation

Pairwise CRISPR phenotypes are dominated by **additive effects of individual gene knockouts**.
Before modeling epistasis, the model should **explicitly condition on this additive baseline**.

Inspired by **NAIAD**, we introduce an **overparameterized encoder** that:

- Projects low-dimensional single-gene effects into a high-dimensional latent space
- Learns a flexible additive mapping
- Correlates strongly with linear additive models
- Provides a stable baseline for learning nonlinear interactions

### 8.2 Input Representation

For a gene pair \( (i, j) \):

- \( Y_i \): single-gene effect of gene \( i \)
- \( Y_j \): single-gene effect of gene \( j \)

\( x_{ij} = \text{concat}(Y_i, Y_j) \in \mathbb{R}^2 \)

### 8.3 Additive Encoder Module

The additive baseline is defined as:

$$
Y_{\text{additive}} = \phi(x_{ij} W_1) A_1^\top
$$

#### Components

- **\( W_1 \in \mathbb{R}^{2 \times m} \)**
  Overparameterized encoder matrix, where \( m \gg 2 \) (for example, 64 to 512)
- **\( \phi \)**
  Nonlinear activation function (ReLU or GeLU)
- **\( A_1^T \in \mathbb{R}^{m \times 1} \)**
  Projection back to a scalar additive prediction

### 8.4 Properties

- Learns a **nonlinear but additive baseline**
- Strongly correlated with standard linear additive models
- Prevents the model from rediscovering trivial additive structure
- Improves stability and sample efficiency in AL settings

---

## 9. Full Surrogate Model Structure

The surrogate model predicts GI score as:

$$
\hat{y}_{ij} = Y_{\text{additive}}(i, j) + f_{\text{epistasis}}(i, j)
$$

Where:

- \( Y_{\text{additive}} \): NAIAD-style additive baseline
- \( f_{\text{epistasis}} \): residual model capturing nonlinear interactions

### Residual Model (default)

- Input: latent additive embedding or concatenated pair features
- Model: shallow MLP or tree ensemble
- Output: epistatic deviation

---

## 10. Uncertainty Estimation

- Use ensemble models or MC dropout
- Uncertainty is computed on the **full prediction**
- Used directly in acquisition functions

---

## 11. Acquisition Function

Target: **strong interactions regardless of sign**

$$
\text{acquisition}(i, j) = |\mu_{ij}| + \beta \cdot \sigma_{ij}
$$

Where:

- \( \mu_{ij} \) = predicted GI
- \( \sigma_{ij} \) = uncertainty
- \( \beta \) = exploration parameter

---

## 12. Diversity-Aware Batch Selection

- **Method:** greedy k-center (max-min)
- **Embedding space:** latent additive or residual representation
- Ensures coverage of perturbation space

---

## 13. Active Learning Loop

1. Load single-gene effects
2. Build candidate gene pairs
3. Encode additive baseline
4. Seed initial labeled batch
5. Train surrogate model
6. Score unlabeled pairs
7. Apply diversity selection
8. Query GI oracle
9. Update training set and repeat

---

## 14. Evaluation Metrics

- **Hit rate:** fraction with \( |GI| \ge 3 \)
- **Top-k enrichment**
- **Learning curves**
- **Correlation with oracle GI**

---

## 15. Deliverables

### 15.1 Compute Script

`notebooks/bulk_single_dual/scripts/al_bulk_epistasis.py`

### 15.2 Marimo Notebook

`notebooks/bulk_single_dual/active_learning_bulk_epistasis_tutorial.py`

---

## 16. Outputs

```
outputs/
  rounds.csv
  metrics.csv
  selected_pairs.csv
```

---

## 17. Constraints and Risks

- DepMap licensing (no redistribution)
- Pair explosion mitigated by oracle alignment
- Overparameterization requires careful regularization

---

## 18. Acceptance Criteria

- Additive encoder reproduces linear baseline behavior
- Residual model learns non-additive GI
- AL improves sample efficiency vs random
- End-to-end run completes locally

---

## 19. References

- Horlbeck et al. 2018 CRISPRi GI Map
- DepMap CRISPR Gene Effect (Chronos)
- NAIAD: Neural Additive Interaction Decomposition
- Active Learning (UCB, batch AL)
