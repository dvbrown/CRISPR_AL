# Instructions

This tutorial outlines a Python AI-in-the-loop framework for bulk CRISPR screens.
It follows NAIAD and BioBO-style methodology for phenotype prediction (for example,
cell viability), rather than full transcriptomic profile prediction.

## Tutorial Goal

### Cross-Dataset Bayesian Optimization

Train a surrogate model on Dataset A (for example, Norman et al. 2019 CRISPRa), then
run active learning to efficiently discover top-performing perturbations in Dataset B
(for example, Simpson et al. 2023 CRISPRi).

## 1. Model Architecture

### NAIAD-Style Surrogate

Implement a neural network with separate additive and interaction pathways.

Additive component (overparameterized baseline):

$$
\hat{y}^{\mathrm{add}}_{i,j}
=
\phi\!\left(\left[y_i, y_j\right] W_1\right) A_1^\top
$$

Genetic interaction component (adaptive gene embeddings):

$$
\hat{y}^{\mathrm{int}}_{i,j}
=
f\!\left(\phi\!\left(W_2 x_i\right) + \phi\!\left(W_2 x_j\right)\right) A_2^\top
$$

where \(x_i, x_j \in \mathbb{R}^{p}\) are learnable gene embeddings and \(p\) scales
with available training data to reduce overfitting.

Fusion layer:

$$
\hat{y}_{i,j} = \hat{y}^{\mathrm{add}}_{i,j} + \hat{y}^{\mathrm{int}}_{i,j}
$$

## 2. Active Learning Loop

Run the following stages iteratively:

1. Initial training:
   Train the surrogate on Dataset A and initialize embedding dimension \(p\) to a
   small value (for example, \(p=2\)).
2. Recommendation (acquisition):
   Select the next batch of \(N\) gene pairs from Dataset B.
   Maximum Predicted Effect (MPE):
   $$
   a_{\mathrm{MPE}}(i,j) = \hat{y}_{i,j}
   $$
   \(\pi\)-BO (EA-informed):
   $$
   a_{\pi\text{-}\mathrm{BO}}(x)
   =
   \hat{y}(x)\cdot\left(\pi_n(x)\right)^{\frac{\beta}{L_n}}
   $$
3. Simulated wet-lab query:
   Label selected pairs using ground-truth phenotype values from Dataset B.
4. Model update:
   Append new labeled pairs, increase \(p\) according to the NAIAD schedule (Table 5),
   and retrain.

## 3. Evaluation Metrics

Use cross-dataset efficiency metrics:

- Cumulative Top-\(k\) Recall:
  $$
  \mathrm{Recall@}k(t)
  =
  \frac{\left|S_t \cap T_k\right|}{\left|T_k\right|}
  $$
  where \(S_t\) is the set discovered up to round \(t\), and \(T_k\) is the true top
  \(k\%\) set in Dataset B.
- Marginal Gain (top-200 discovery per round):
  $$
  \mathrm{MG}(t) = \left| \left(S_t \setminus S_{t-1}\right) \cap T_{200} \right|
  $$
- RMSE near optimum (top \(q\%\), e.g. \(q \in [1,10]\)):
  $$
  \mathrm{RMSE}_{\mathrm{top}\ q\%}
  =
  \sqrt{
    \frac{1}{\left|\Omega_q\right|}
    \sum_{(i,j)\in\Omega_q}
    \left(\hat{y}_{i,j} - y_{i,j}\right)^2
  }
  $$
  where \(\Omega_q\) indexes top-performing perturbations by ground truth.

## 4. GEARS-Adaptable Functionality

Add the following GEARS-inspired components to improve robustness and transfer:

1. Split taxonomy for extrapolation stress tests:
   Report performance by perturbation visibility class (analogous to GEARS
   simulation/custom split modes):
   - `seen-2`: both genes in a queried pair were seen as singles during training.
   - `seen-1`: exactly one gene was seen as a single.
   - `seen-0`: neither gene was seen as a single.
   This should be tracked both within Dataset A validation and during Dataset B active
   learning rounds.
2. Graph priors from biological networks:
   Build a gene graph from GO similarity and/or co-expression, then regularize
   embeddings with a graph smoothness penalty:
   $$
   \mathcal{L}_{\mathrm{graph}} = \mathrm{tr}(X^\top L X)
   $$
   where \(X\) is the gene embedding matrix and \(L\) is the graph Laplacian.
3. Uncertainty-aware surrogate mode:
   Add a heteroscedastic output head predicting \(\mu(x)\) and \(\log \sigma^2(x)\),
   optimized with Gaussian NLL:
   $$
   \mathcal{L}_{\mathrm{NLL}}
   =
   \frac{1}{2}\left(
   \log \sigma^2(x) + \frac{(y-\mu(x))^2}{\sigma^2(x)}
   \right)
   $$
   Then support uncertainty-aware acquisition, for example:
   $$
   a_{\mathrm{UCB}}(x) = \mu(x) + \kappa \sigma(x)
   $$
4. Genetic interaction diagnostics:
   In addition to prediction error, compute interaction residuals:
   $$
   \epsilon_{i,j} = y_{i,j} - (y_i + y_j)
   $$
   and evaluate whether selected hits are enriched for strong \(\lvert\epsilon_{i,j}\rvert\).
5. Reproducible data/model interface:
   Separate `PertData`-like data handling from model/trainer logic, and persist:
   split definitions, graph construction artifacts, and round-wise checkpoints.
   This is required for restartable active learning experiments.

## 5. NAIAD-Adaptable Functionality

Add the following NAIAD-specific behaviors to make the plan operationally aligned
with the reference implementation:

1. Data contract and preprocessing pipeline:
   Standardize data loading through a `load_naiad_data`-style pipeline with:
   `load_phenotype_df -> reorganize_single_treatment_effects -> shuffle_ids`.
   Ensure the processed schema is:
   `id1, id2, comb_score, id1_score, id2_score`.
   Keep row-wise treatment shuffling enabled by default to prevent positional bias.
2. Explicit seed and reshuffle guard:
   Mirror NAIAD's reproducibility safeguard:
   - `set_seed` marks the random state as changed.
   - `shuffle_data` must be called before running a new AL cycle.
   - raise a warning/error if seed changed without reshuffling.
3. Ensemble-based acquisition statistics:
   For ensemble predictions \(\{\hat{y}_m(x)\}_{m=1}^{M}\), compute:
   $$
   \mu(x) = \frac{1}{M}\sum_{m=1}^{M}\hat{y}_m(x),
   \quad
   \sigma(x) = \sqrt{\frac{1}{M-1}\sum_{m=1}^{M}\left(\hat{y}_m(x)-\mu(x)\right)^2}
   $$
   Fit a linear baseline from single-treatment effects:
   $$
   \hat{y}_{\mathrm{lin}}(x) = w^\top s(x) + b,
   \quad
   s(x)=[y_i, y_j]^\top
   $$
   and support NAIAD-style ranking scores:
   $$
   a_{\mathrm{mean}}(x)=\mu(x),\;
   a_{\mathrm{std}}(x)=\sigma(x),\;
   a_{\mathrm{mean+std}}(x)=|\mu(x)|+\sigma(x)
   $$
   $$
   a_{\mathrm{residual}}(x)=\left|\hat{y}_{\mathrm{lin}}(x)-\mu(x)\right|,
   \;
   a_{\mathrm{residual+std}}(x)=\sigma(x)+\left|\hat{y}_{\mathrm{lin}}(x)-\mu(x)\right|
   $$
4. Round-wise sampling budget update:
   Use cumulative sample schedule \(n_{\mathrm{sample}}[t]\), with incremental draw:
   $$
   n_{\mathrm{select}}(t)=n_{\mathrm{sample}}[t]-n_{\mathrm{sample}}[t-1]
   $$
   At each round, move the top-ranked \(n_{\mathrm{select}}(t)\) candidates from the
   unlabeled/validation pool into the training pool.
5. Aggregate metrics aligned to NAIAD utilities:
   Track:
   $$
   \mathrm{MSE}(t,S)=\frac{1}{|S|}\sum_{x\in S}\left(\mu_t(x)-y(x)\right)^2
   $$
   and top-\(N\) discovery curves (TPR/recall) via a
   `find_top_n_perturbations`-style computation across rounds.
   For "overall" reporting, overwrite predictions with measured values for already
   assayed pairs before computing top-\(N\) recovery.
6. Replicate manager for uncertainty over runs:
   Add an `ActiveLearnerReplicates`-style wrapper over multiple seeds that stores,
   per replicate and round:
   - predictions,
   - training metrics,
   - aggregate metrics.
   Support serial and parallel replicate execution and produce method-comparison
   plots for MSE and top-\(N\) recovery.
7. Adaptive embedding dimension schedule:
   Keep embedding size \(p\) adaptive to treatment coverage in training data
   (`n_train / n_treatments`), using an explicit lookup schedule (the same design as
   NAIAD's `n_treatment_seen` table) instead of a fixed embedding size.

### References

- [NAIAD](https://deepwiki.com/NeptuneBio/NAIAD)
  - Active learning system that models combinatorial perturbation outcomes and recommends gene pairs for follow-up screens.
- [NAIAD: Overview](https://deepwiki.com/NeptuneBio/NAIAD/1-overview)
- [NAIAD: The NAIAD Class](https://deepwiki.com/NeptuneBio/NAIAD/2.1-the-naiad-class)
- [NAIAD: Model Architecture](https://deepwiki.com/NeptuneBio/NAIAD/2.2-model-architecture)
- [NAIAD: Active Learning](https://deepwiki.com/NeptuneBio/NAIAD/2.3-active-learning)
- [NAIAD: Data Utilities](https://deepwiki.com/NeptuneBio/NAIAD/2.4-data-utilities)
- [NAIAD: Active Learning Tutorial](https://deepwiki.com/NeptuneBio/NAIAD/3.3-active-learning-tutorial)
- [GEARS](https://deepwiki.com/snap-stanford/GEARS)
  - Geometric deep learning model that predicts transcriptional outcomes for single and multi-gene perturbations in single-cell screens.
- [GEARS: Data Splitting and Simulation](https://deepwiki.com/snap-stanford/GEARS/2.2-data-splitting-and-simulation)
- [GEARS: Network Construction](https://deepwiki.com/snap-stanford/GEARS/3.2-network-construction)
- [GEARS: Uncertainty Prediction](https://deepwiki.com/snap-stanford/GEARS/4.2-uncertainty-prediction)
- [GEARS: Genetic Interaction Analysis](https://deepwiki.com/snap-stanford/GEARS/4.3-genetic-interaction-analysis)
