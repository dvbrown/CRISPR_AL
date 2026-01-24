# Active learning for CRISPR screens — high-level summary

## What “active learning” means here
Active learning (AL) in CRISPR screening is an **iterative design loop**:
1) run an initial (often smaller) screen  
2) train a model mapping perturbations → phenotype  
3) select the next perturbations using an acquisition rule  
4) repeat to maximize discoveries per experiment

AL is most valuable when screens are **expensive per perturbation** (single-cell, imaging, in vivo) or **combinatorial** (gene pairs/tuples).

---

## How the initial guides / genes are chosen (iteration 1)
**Cold start problem:** the model needs initial labeled data.

Typical round-0 strategies:
- **Random / stratified sampling** to avoid bias
- **Diversity-first sampling** (cover embedding/feature space)
- **Prior-informed seeding** (pathways, expression, known regulators, prior screens)
---

## Using other CRISPR screens to inform initial gRNA selection
Prior screen results can guide round-0 library design via:
1) **empirical guide priors** (guides that consistently work across screens)
2) **trained guide-activity models** (transfer learning from prior datasets)

---

## Surrogate models used
Surrogates predict phenotype from perturbation(s). Common types:
- **deep models + gene graphs** for transcriptomic/single-cell readouts
- **embedding-based models** for gene-pair / interaction prediction
- **probabilistic/Bayesian surrogates** for uncertainty-aware selection

---

## Acquisition strategies used
How the next perturbations are selected:
- **exploitation:** pick highest predicted effects
- **exploration:** pick highest uncertainty / disagreement
- **Bayesian optimization style:** mean + uncertainty tradeoffs (e.g. UCB/Thompson)
- **diversity-aware batch selection:** avoid redundant picks in pooled batches
- **discovery-set objectives:** maximize number of strong, diverse hits (not just the top one)

---

## Balancing exploration vs exploitation
Common practical patterns:
- explicit mean/uncertainty mixing in acquisition functions
- **split-batch** selection (e.g. X% exploit + (100–X)% explore)
- enforce **diversity constraints** to prevent model lock-in
- in low-round wet-lab settings, priors often push toward more exploitation early

---

## From single-gene screens → selecting combinations to test
Goal: use single-gene data to prioritize **gene pairs / guide pairs** for experimental validation.

Workflow:
1) train a model on **single-gene perturbation outcomes**
2) define objective (synergy, max effect, target program shift, diversity)
3) score candidate gene pairs and select with AL (effect + uncertainty + diversity)
4) map gene pairs → **high-quality guide pairs** (activity + specificity + compatibility)

Key idea: single-gene data provides the base signal; AL prioritizes which combinations are worth spending budget on.
