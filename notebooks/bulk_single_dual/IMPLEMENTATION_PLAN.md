# Implementation Plan: Active Learning for Bulk CRISPR Epistasis (IterPert Strategy Reuse)

This plan outlines how to implement the active learning tutorial using reusable components and ideas from the IterPert codebase in `external/iterative-perturb-seq`. The focus is on adapting the AL strategies (kernel-based selection, k-center, k-means, TypiClust) to bulk pairwise epistasis with a NAIAD-style additive baseline.

## 1. Scope and Outputs

- **Goal:** A lightweight, reproducible tutorial that performs active learning over gene pairs with an additive baseline + epistasis residual model.
- **Outputs:**
  - `notebooks/bulk_single_dual/scripts/al_bulk_epistasis.py` (compute loop)
  - `notebooks/bulk_single_dual/active_learning_bulk_epistasis_tutorial.py` (marimo narrative)
  - `outputs/rounds.csv`, `outputs/metrics.csv`, `outputs/selected_pairs.csv`

---

## 2. Reusable IterPert Strategy Components

### 2.1 Strategy Interface (API contract)

**Source:** `external/iterative-perturb-seq/reproduce_repo/query_strategies/strategy.py`

Reuse the minimal strategy structure:

- `query(n)` returns indices of selected candidates
- `train()` updates the surrogate model
- `predict()` returns mean predictions
- `get_embeddings()` returns embeddings for diversity selection

Adopt the same pattern for a `BulkEpistasisStrategy` wrapper so AL logic is separable from model code.

### 2.2 Kernel-based Selection (uncertainty/diversity)

**Source:** `external/iterative-perturb-seq/reproduce_repo/query_strategies/kernel_based_active_learning.py`

Key ideas to reuse:

- Build embeddings for candidates (here: additive encoder latent or residual features).
- Optional PCA reduction if embedding dimension is large.
- Use a kernel (linear on embeddings) and select using a batch method.

Adaptation:

- Replace perturbation embeddings with pair embeddings.
- Use kernel on pair embeddings for diversity or acquisition scoring.
- Skip prior kernels unless explicitly provided (keep minimal).

### 2.3 k-means Batch Selection

**Source:** `external/iterative-perturb-seq/reproduce_repo/query_strategies/kmeans_sampling.py`

Use the k-means batch idea as a diversity baseline:

- Cluster pair embeddings into `n` clusters.
- Select nearest-to-centroid per cluster.
- Optionally apply PCA first, mirroring IterPert’s fallback.

### 2.4 TypiClust (density-aware selection)

**Source:** `external/iterative-perturb-seq/reproduce_repo/query_strategies/typiclust.py`

Use TypiClust as an optional diversity strategy:

- Cluster into `n + labeled` clusters.
- Select most typical (highest density) point in least-covered clusters.
- Use kNN distance as typicality.

### 2.5 Max-distance (k-center) Greedy Selection

**Source (in-repo tutorial):** `notebooks/iterpert/iterpert_tutorial_k562.py` (`farthest_point_batch`)

Use this as the default diversity method for simplicity:

- Compute embeddings for all unlabeled pairs.
- Seed with random labeled set.
- Iteratively pick the farthest point (max-min) from selected set.

---

## 3. Data and Candidate Pair Construction

1. **Load DepMap single-gene effects** (K562, Chronos).
2. **Load Horlbeck GI map** (K562 pairs).
3. **Intersect genes** to define candidate pool.
4. **Canonicalize pairs** as `(min(gene_a, gene_b), max(gene_a, gene_b))`.
5. **Create index mapping** from pair to row for fast lookup.

---

## 4. Model Architecture

### 4.1 Additive Encoder (NAIAD-style)

- Input: `x_ij = concat(Y_i, Y_j)`
- Encode with overparameterized MLP layer: `phi(x_ij W1)`
- Project to scalar `Y_additive`

### 4.2 Residual Epistasis Model

- Use a shallow MLP or tree ensemble.
- Input: additive embedding or concatenated pair features.
- Output: epistatic residual.

### 4.3 Prediction

- `y_hat = Y_additive + f_epistasis`

---

## 5. Uncertainty + Acquisition

- Default: ensemble of lightweight models or MC dropout for MLP.
- Compute `mu` and `sigma` on full prediction.
- Acquisition: `|mu_ij| + beta * sigma_ij`.

---

## 6. Active Learning Loop

1. Seed labeled pairs.
2. Train surrogate model on labeled set.
3. Compute embeddings and predictions for unlabeled set.
4. Rank by acquisition.
5. Apply diversity selection (k-center or k-means/TypiClust).
6. Query GI oracle and append to labeled set.
7. Log metrics and repeat.

---

## 7. Implementation Milestones

1. **Data loaders + candidate pairs**
   - Implement loading + intersection logic.
2. **Additive baseline module**
   - Overparameterized encoder + scalar projection.
3. **Residual model + training loop**
   - Fit residuals; compute predictions.
4. **AL strategy wrappers**
   - Implement k-center, k-means, TypiClust selectors.
5. **Full AL loop + logging**
   - Save outputs per round.
6. **Notebook narrative + plots**
   - Visualize hits, learning curves, and selection diversity.

---

## 8. Minimal Strategy Selection Defaults

- **Primary:** k-center greedy on pair embeddings.
- **Optional:** k-means or TypiClust (plug-in selection).
- **Fallback:** random sampling for baseline comparison.

---

## 9. Notes and Constraints

- Keep CPU-only and laptop-feasible by limiting candidate pairs.
- Avoid pulling heavy dependencies from IterPert; reuse only algorithm ideas.
- Maintain data confidentiality (DepMap licensing).
