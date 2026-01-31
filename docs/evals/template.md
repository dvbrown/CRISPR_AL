# Repo Evaluations

## Contents
- [Active Learning Focused](#active-learning-focused)
- [GEARS](#gears)
- [Iterative Perturb-Seq (IterPert)](#iterative-perturb-seq-iterpert)
- [Combinatorial / Dual-Guide / Interaction-Focused](#combinatorial--dual-guide--interaction-focused)
- [gimap](#gimap)
- [NAIAD](#naiad)
- [DiscoBAX](#discobax)
- [Perturbation Prediction](#perturbation-prediction)
- [GenePert](#genepert)
- [scGenePT](#scgenept)
- [state](#state)
- [AttentionPert](#attentionpert)
- [scPRAM](#scpram)
- [GPerturb](#gperturb)
- [PerturbNet](#perturbnet)
- [scLAMBDA](#sclambda)
- [Benchmarking Frameworks](#benchmarking-frameworks)
- [perturbench](#perturbench)
- [scPerturb](#scperturb)
- [scPerturBench](#scperturbench)


## Active Learning Focused

### GEARS
#### Repo
- URL: https://github.com/snap-stanford/GEARS
- Primary language: Python
- Last update: 2025-02-01

#### Problem Statement
- What the repository is trying to solve: Predict transcriptional outcomes of single and multi-gene perturbations in scRNA-seq screens.
- Target data type (single/dual/combinatorial): single + combinatorial (multi-gene).
- Output type (gene scores, interactions, hits, etc.): predicted gene expression profiles; GI predictions.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): graph neural network with perturbation-aware modeling.
- Key assumptions: training data must include some combinatorial perturbations; not designed for cross-cell-type transfer.
- Training objective: supervised prediction of perturbation-induced gene expression.
- Inference outputs: predicted expression changes for single or multi-gene perturbations.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none (modeling only).
- Uncertainty estimation: optional uncertainty-aware tutorial notebook.
- Batch strategy: not described.

#### Reproducibility
- Install steps: install PyG, then `pip install cell-gears`.
- Minimal run command: use the API snippet in README to load data, train, and predict.
- Required datasets: GEARS datasets (Norman/Adamson/Dixit) or custom AnnData.
- Expected outputs: trained model artifacts, predicted expression matrices.

#### Evaluation
- Metrics used: not specified in README (see paper).
- Baselines: not specified in README.
- Strengths: clear API, tutorials, and packaged install; strong model for combinatorial prediction.
- Limitations: needs combinatorial training data; not cross-cell-type; heavy compute.
- Documentation quality: high for API usage; demos and Colab provided.
- Code quality: packaged library, structured API; details of experiments external.
- Math quality/robustness: high (peer-reviewed Nature Biotech); modeling assumptions explicit.

#### Reusable Components
- Code modules worth extracting: `PertData` loader, data split utilities, uncertainty tutorial patterns.
- Ideas to integrate into custom workflow: use `PertData` and uncertainty heads for acquisition scoring.

### Iterative Perturb-Seq (IterPert)
#### Repo
- URL: https://github.com/Genentech/iterative-perturb-seq
- Primary language: Python
- Last update: 2024-05-24

#### Problem Statement
- What the repository is trying to solve: sequential experimental design for Perturb-seq using multimodal priors.
- Target data type (single/dual/combinatorial): single-gene perturbations (Perturb-seq).
- Output type (gene scores, interactions, hits, etc.): ranked perturbations to query next; predicted expression.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): GEARS-based neural model with active learning policy.
- Key assumptions: active learning under budget; priors improve early rounds.
- Training objective: predict perturbation outcomes while optimizing selection.
- Inference outputs: selected perturbation sets and model predictions.

#### Active Learning / Selection Strategy
- Acquisition function (if any): IterPert plus baselines (Random, BALD, BatchBALD, BAIT, Core-Set, BADGE, LCMD).
- Uncertainty estimation: implicit via BALD/BatchBALD-type methods.
- Batch strategy: explicit batch selection per round.

#### Reproducibility
- Install steps: conda env, install PyG, `pip install iterpert` or `pip install -r requirements.txt`.
- Minimal run command: initialize IterPert, data, model, strategy, then `interface.start(...)`.
- Required datasets: Replogle K562/RPE1 or other datasets via API.
- Expected outputs: selected perturbations per round, trained model, metrics.

#### Evaluation
- Metrics used: described in paper; not detailed in README.
- Baselines: multiple AL baselines listed.
- Strengths: explicit AL loop; supports priors and multiple acquisition functions.
- Limitations: relies on GEARS; requires GPU; dataset prep may be heavy.
- Documentation quality: good API overview and tutorials; reproducibility details separated.
- Code quality: organized API wrapper around AL loop.
- Math quality/robustness: strong conceptual framing; relies on standard AL methods.

#### Reusable Components
- Code modules worth extracting: acquisition strategy implementations; IterPert orchestration.
- Ideas to integrate into custom workflow: reuse AL baselines and batch selection loop.

## Combinatorial / Dual-Guide / Interaction-Focused

### gimap
#### Repo
- URL: https://github.com/FredHutch/gimap
- Primary language: R
- Last update: 2026-01-14

#### Problem Statement
- What the repository is trying to solve: genetic interaction scoring for dual-guide CRISPR screens.
- Target data type (single/dual/combinatorial): dual-guide paired CRISPR counts.
- Output type (gene scores, interactions, hits, etc.): GI scores, p-values, FDR.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): linear model + t-tests.
- Key assumptions: expected effects are additive; control normalization required.
- Training objective: estimate deviations from expected additive effects.
- Inference outputs: GI scores and significance per gene pair.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none (analysis only).
- Uncertainty estimation: statistical tests and FDR.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: install R, `remotes::install_github("FredHutch/gimap")`.
- Minimal run command: follow quick-start tutorial on hosted docs.
- Required datasets: paired gRNA counts (pgmap or similar).
- Expected outputs: GI score tables, volcano plots.

#### Evaluation
- Metrics used: GI scores, t-tests, FDR.
- Baselines: implicit additive expectation model.
- Strengths: clear statistical definitions, strong tutorial docs, Docker option.
- Limitations: limited to paired-guide paradigm; assumes additive expectation.
- Documentation quality: high (long-form docs, tutorials, figures).
- Code quality: mature R package, documented workflows.
- Math quality/robustness: solid classical stats; robust for pairwise GI but not for AL.

#### Reusable Components
- Code modules worth extracting: normalization and GI score calculation.
- Ideas to integrate into custom workflow: use GI scores as labels/priors for combinatorial AL.

### NAIAD
#### Repo
- URL: https://github.com/NeptuneBio/NAIAD
- Primary language: Python
- Last update: 2025-07-09

#### Problem Statement
- What the repository is trying to solve: active learning for combinatorial perturbation outcome prediction and gene-pair recommendation.
- Target data type (single/dual/combinatorial): combinatorial (gene pairs).
- Output type (gene scores, interactions, hits, etc.): recommended gene pairs; predicted phenotypes.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): not specified in README (see paper).
- Key assumptions: small-sample learning with iterative AL.
- Training objective: predict phenotypes and maximize information gain.
- Inference outputs: ranked gene pairs for follow-up.

#### Active Learning / Selection Strategy
- Acquisition function (if any): information gain-driven recommendations (details in paper).
- Uncertainty estimation: implied; not detailed in README.
- Batch strategy: iterative rounds recommended.

#### Reproducibility
- Install steps: Python >=3.8, `pip install -e .`.
- Minimal run command: use tutorial notebooks for data prep, training, AL.
- Required datasets: combinatorial perturbation datasets (prepared via tutorial).
- Expected outputs: trained models and recommended gene pairs.

#### Evaluation
- Metrics used: not specified in README.
- Baselines: not specified in README.
- Strengths: explicitly focuses on combinatorial AL; concise tutorials.
- Limitations: sparse install and method details in README; rely on paper for math.
- Documentation quality: minimal but focused (3 notebooks).
- Code quality: unknown without deeper inspection; structure implied by tutorials.
- Math quality/robustness: likely solid (arXiv), but needs deeper review.

#### Reusable Components
- Code modules worth extracting: AL recommendation pipeline in tutorial.
- Ideas to integrate into custom workflow: combine with GEARS-like predictors for pair ranking.

### DiscoBAX
#### Repo
- URL: https://github.com/amehrjou/DiscoBAX
- Primary language: Python
- Last update: 2023-12-07

#### Problem Statement
- What the repository is trying to solve: discovery of diverse, high-impact intervention sets under limited experiments.
- Target data type (single/dual/combinatorial): intervention sets; gene-level screening.
- Output type (gene scores, interactions, hits, etc.): ranked batches of interventions.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): Bayesian optimization with diversity-aware acquisition.
- Key assumptions: phenotypic proxy and diversity coverage improve downstream success.
- Training objective: maximize phenotype movement + diversity.
- Inference outputs: diverse intervention batches.

#### Active Learning / Selection Strategy
- Acquisition function (if any): BAX variants (topk_bax, levelset_bax, subsetmax_bax_additive).
- Uncertainty estimation: GP/BO uncertainty.
- Batch strategy: explicit batch acquisition cycles.

#### Reproducibility
- Install steps: `pip install -r requirements.txt`, `pip install discobax`.
- Minimal run command: `python discobax/apps/toy_experiment.py ...`.
- Required datasets: synthetic or GeneDisco benchmark; external download required.
- Expected outputs: performance metrics, plots, output directories.

#### Evaluation
- Metrics used: described in paper; not detailed in README.
- Baselines: multiple BO/AL baselines.
- Strengths: rigorous AL framing with theoretical guarantees; clear CLI examples.
- Limitations: dataset setup nontrivial; heavy dependency on GeneDisco benchmark.
- Documentation quality: moderate (examples provided, deeper details in paper).
- Code quality: research code with apps/ scripts; reproducibility notes.
- Math quality/robustness: strong theoretical guarantees (ICML 2023).

#### Reusable Components
- Code modules worth extracting: diversity-aware acquisition functions.
- Ideas to integrate into custom workflow: use BAX acquisition for batch diversity in AL.

## Perturbation Prediction

### GenePert
#### Repo
- URL: https://github.com/zou-group/GenePert
- Primary language: Python
- Last update: 2024-10-29

#### Problem Statement
- What the repository is trying to solve: predict gene expression changes using GenePT embeddings.
- Target data type (single/dual/combinatorial): single-gene perturbations.
- Output type (gene scores, interactions, hits, etc.): predicted expression changes.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): regularized linear regression.
- Key assumptions: linear mapping from gene embeddings to perturbation effects.
- Training objective: minimize regression error on expression profiles.
- Inference outputs: predicted expression for held-out perturbations.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: none specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: not specified in README.
- Minimal run command: `genepert-k562-demo.ipynb` notebook.
- Required datasets: Perturb-seq datasets listed in README.
- Expected outputs: cross-validated prediction metrics.

#### Evaluation
- Metrics used: Pearson correlation, MSE (per README).
- Baselines: compared in paper, not in README.
- Strengths: simple, interpretable model with strong reported performance.
- Limitations: missing install/setup docs; reliant on external embeddings.
- Documentation quality: minimal; notebook-focused.
- Code quality: unknown without deeper inspection.
- Math quality/robustness: solid baseline with clear linear assumptions.

#### Reusable Components
- Code modules worth extracting: embedding-to-expression regression pipeline.
- Ideas to integrate into custom workflow: use GenePT embeddings as priors/features for AL surrogate.

### scGenePT
#### Repo
- URL: https://github.com/czi-ai/scGenePT
- Primary language: Python
- Last update: 2025-01-22

#### Problem Statement
- What the repository is trying to solve: inject language-derived gene embeddings into scGPT for perturbation prediction.
- Target data type (single/dual/combinatorial): single and two-gene perturbations.
- Output type (gene scores, interactions, hits, etc.): predicted expression responses.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): transformer-based scGPT with embedding injection.
- Key assumptions: LLM-derived gene embeddings improve generalization.
- Training objective: predict perturbation expression responses.
- Inference outputs: predicted expression matrices.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: conda env, `pip install -r requirements.txt`, optional flash-attn, scGPT.
- Minimal run command: `python train.py --model-type=... --dataset=norman ...`.
- Required datasets: GEARS datasets plus large embeddings from S3.
- Expected outputs: trained models, inference results.

#### Evaluation
- Metrics used: not detailed in README.
- Baselines: scGPT and genePT variants.
- Strengths: detailed data download instructions; explicit training steps.
- Limitations: heavy data downloads; GPU required.
- Documentation quality: high for setup, moderate for evaluation.
- Code quality: research-grade with scripts and tutorials.
- Math quality/robustness: strong foundation model approach; relies on external embeddings.

#### Reusable Components
- Code modules worth extracting: embedding injection pipeline, data loading utilities.
- Ideas to integrate into custom workflow: use embedding conditioning for surrogate models.

### state
#### Repo
- URL: https://github.com/ArcInstitute/state
- Primary language: Python
- Last update: 2026-01-23

#### Problem Statement
- What the repository is trying to solve: train state transition/embedding models for perturbation prediction across contexts.
- Target data type (single/dual/combinatorial): single-gene perturbations (scRNA-seq).
- Output type (gene scores, interactions, hits, etc.): predicted expression; embeddings.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): deep state transition models; embedding models.
- Key assumptions: shared latent structure across contexts can be captured with embeddings.
- Training objective: predict perturbation responses and learn embeddings.
- Inference outputs: predicted expression, transformed embeddings.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: `uv tool install arc-state` or source install via `uv`.
- Minimal run command: `state tx train ...` or provided Colabs.
- Required datasets: h5ad datasets with TOML config.
- Expected outputs: model checkpoints, predictions, embeddings.

#### Evaluation
- Metrics used: via `cell-eval` (external) for ST evaluation.
- Baselines: not listed in README.
- Strengths: detailed CLI, reproducibility notes, containerization.
- Limitations: higher setup complexity; multiple repos needed (cell-load/eval).
- Documentation quality: high, comprehensive CLI docs.
- Code quality: mature tooling with configs and CLI.
- Math quality/robustness: strong modeling framework; details in paper.

#### Reusable Components
- Code modules worth extracting: CLI pipelines, preprocess/train/predict workflow.
- Ideas to integrate into custom workflow: use ST model for surrogate prediction; reuse Hydra-style configs.

### AttentionPert
#### Repo
- URL: https://github.com/BaiDing1234/AttentionPert
- Primary language: Python
- Last update: 2024-07-24

#### Problem Statement
- What the repository is trying to solve: predict multiplexed perturbation effects with attention and multi-scale modeling.
- Target data type (single/dual/combinatorial): multi-gene perturbations.
- Output type (gene scores, interactions, hits, etc.): predicted expression responses.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): attention-based deep model with multi-scale effects.
- Key assumptions: multi-scale gene relationships improve prediction.
- Training objective: supervised prediction of perturbation expression.
- Inference outputs: predicted expression; logs/metrics.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: `conda env create -f environment.yml` (or GEARS deps).
- Minimal run command: `python run_attnpert.py --dataset_name norman ...`.
- Required datasets: GEARS-processed datasets + external GO/gene2vec files.
- Expected outputs: logs and prediction results.

#### Evaluation
- Metrics used: not specified in README.
- Baselines: not specified in README.
- Strengths: clear data prep steps; published paper.
- Limitations: dataset preparation is multi-step; external data required.
- Documentation quality: moderate; setup documented but verbose.
- Code quality: research scripts; likely not modular.
- Math quality/robustness: peer-reviewed; likely solid attention mechanism.

#### Reusable Components
- Code modules worth extracting: multi-scale attention architecture.
- Ideas to integrate into custom workflow: use model as surrogate for combinatorial ranking.

### scPRAM
#### Repo
- URL: https://github.com/jiang-q19/scPRAM
- Primary language: Python
- Last update: 2024-10-17

#### Problem Statement
- What the repository is trying to solve: predict single-cell perturbation response with attention model.
- Target data type (single/dual/combinatorial): single-gene perturbations.
- Output type (gene scores, interactions, hits, etc.): predicted expression and evaluation metrics.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): attention-based neural model.
- Key assumptions: attention captures gene interactions for perturbation response.
- Training objective: supervised prediction.
- Inference outputs: predicted responses and evaluation scores.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: conda env + pip install scpram (CUDA/CPU wheel).
- Minimal run command: use tutorial notebook.
- Required datasets: user-provided AnnData.
- Expected outputs: predicted perturbation response and evaluation results.

#### Evaluation
- Metrics used: described in tutorial notebook.
- Baselines: not specified in README.
- Strengths: quick-start tutorial, PyPI package.
- Limitations: requires CUDA-specific wheels for GPU; docs light.
- Documentation quality: moderate.
- Code quality: packaged, but limited docs.
- Math quality/robustness: standard attention model; details in paper.

#### Reusable Components
- Code modules worth extracting: training/prediction API in `scpram` package.
- Ideas to integrate into custom workflow: lightweight surrogate for single-gene prediction.

### GPerturb
#### Repo
- URL: https://github.com/hwxing3259/GPerturb
- Primary language: Python
- Last update: unknown

#### Problem Statement
- What the repository is trying to solve: estimate interpretable, sparse gene-level perturbation effects from single-cell data.
- Target data type (single/dual/combinatorial): single-gene perturbations (single-cell expression).
- Output type (gene scores, interactions, hits, etc.): predicted expression and sparse perturbation effect matrix.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): Bayesian sparse distributional regression.
- Key assumptions: perturbation effects are sparse and additive in the model space.
- Training objective: fit perturbation effects to match observed expression distributions.
- Inference outputs: estimated expression and perturbation effect estimates per gene.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: implicit via Bayesian model; not exposed as an AL policy.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: `pip install git+https://github.com/hwxing3259/GPerturb.git`.
- Minimal run command: use numerical example notebooks under `numerical_examples/`.
- Required datasets: SciPlex2 and other single-cell perturbation datasets (examples linked).
- Expected outputs: fitted expression estimates and sparse perturbation effect matrices.

#### Evaluation
- Metrics used: not specified in README (see paper).
- Baselines: not specified in README.
- Strengths: interpretable sparse effects; outputs explicit perturbation effect matrix.
- Limitations: relies on provided covariates and careful preprocessing.
- Documentation quality: concise; relies on notebooks for usage.
- Code quality: single-model implementation, notebook-driven workflows.
- Math quality/robustness: Bayesian modeling with explicit sparsity assumptions.

#### Reusable Components
- Code modules worth extracting: perturbation effect estimation utilities.
- Ideas to integrate into custom workflow: use sparse effect matrix as priors or labels for AL ranking.

### PerturbNet
#### Repo
- URL: https://github.com/welch-lab/PerturbNet/tree/main
- Primary language: Python
- Last update: unknown

#### Problem Statement
- What the repository is trying to solve: predict distributions of single-cell states for unseen chemical or genetic perturbations.
- Target data type (single/dual/combinatorial): single-gene or chemical perturbations (single-cell expression).
- Output type (gene scores, interactions, hits, etc.): predicted cell state distributions and expression profiles.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): deep generative model with cINN and scVI-based decoders.
- Key assumptions: latent representations capture perturbation response distributions.
- Training objective: model perturbation-conditioned single-cell distributions.
- Inference outputs: predicted expression distributions and generated cells.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: conda env + `pip install PerturbNet`.
- Minimal run command: use tutorial notebooks in `notebooks/`.
- Required datasets: tutorial datasets from Hugging Face.
- Expected outputs: trained model checkpoints, predicted cell states, evaluation metrics.

#### Evaluation
- Metrics used: not specified in README (see paper).
- Baselines: not specified in README.
- Strengths: handles unseen perturbations; supports chemical and genetic data.
- Limitations: older dependency stack; GPU recommended.
- Documentation quality: good notebook coverage with benchmarks.
- Code quality: modular package with adapted scVI components.
- Math quality/robustness: strong deep generative modeling; details in paper.

#### Reusable Components
- Code modules worth extracting: conditional generative modeling pipeline.
- Ideas to integrate into custom workflow: use generated distributional predictions as surrogate outputs.

### scLAMBDA
#### Repo
- URL: https://github.com/gefeiwang/scLAMBDA
- Primary language: Python
- Last update: unknown

#### Problem Statement
- What the repository is trying to solve: predict single-cell responses for single and multi-gene perturbations using gene embeddings.
- Target data type (single/dual/combinatorial): single and multi-gene perturbations (single-cell expression).
- Output type (gene scores, interactions, hits, etc.): predicted expression profiles and generated perturbed cells.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): deep generative model with disentangled representations.
- Key assumptions: gene embeddings capture perturbation semantics; disentangling basal vs. perturbation effects improves generalization.
- Training objective: reconstruct and generate perturbation-conditioned single-cell expression.
- Inference outputs: predicted expression means or generated cell distributions.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not specified.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: `conda env update -f environment.yml` then `conda activate sclambda`.
- Minimal run command: use demo notebooks in `demos/`.
- Required datasets: AnnData with perturbation conditions and gene embeddings.
- Expected outputs: trained models, predicted expression matrices.

#### Evaluation
- Metrics used: not specified in README (see paper).
- Baselines: not specified in README.
- Strengths: supports multi-gene perturbations; leverages external embeddings.
- Limitations: requires embedding inputs; setup relies on conda env file.
- Documentation quality: good quick-start and demos.
- Code quality: research code with clear API entry points.
- Math quality/robustness: strong generative modeling framing; details in paper.

#### Reusable Components
- Code modules worth extracting: data split helpers and embedding-conditioned model wrapper.
- Ideas to integrate into custom workflow: use gene embedding conditioning for AL surrogate modeling.

## Benchmarking Frameworks

### perturbench
#### Repo
- URL: https://github.com/altoslabs/perturbench
- Primary language: Python
- Last update: 2026-01-08

#### Problem Statement
- What the repository is trying to solve: standardized benchmarking of perturbation prediction models.
- Target data type (single/dual/combinatorial): single-cell perturbation datasets.
- Output type (gene scores, interactions, hits, etc.): evaluation metrics and model comparisons.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): benchmarking framework (multiple models).
- Key assumptions: standardized preprocessing and metrics enable fair comparison.
- Training objective: configurable via Hydra; evaluates predictions.
- Inference outputs: metrics, rankings, prediction artifacts.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none (evaluation only).
- Uncertainty estimation: not central; depends on model.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: `pip install -e .` with conda env.
- Minimal run command: `train <config-options>` or use Evaluator class.
- Required datasets: HuggingFace-hosted h5ad or curated datasets.
- Expected outputs: metrics tables, evaluation reports.

#### Evaluation
- Metrics used: RMSE, cosine, R2, MMD, top-k recall, etc.
- Baselines: multiple configured models.
- Strengths: extensive metric suite, dataset accessors.
- Limitations: setup can be large; configs complex.
- Documentation quality: high; detailed usage.
- Code quality: well-structured Hydra configs.
- Math quality/robustness: strong evaluation methodology; not an AL method.

#### Reusable Components
- Code modules worth extracting: Evaluator class, metric pipelines, data accessors.
- Ideas to integrate into custom workflow: use metrics and data splits for AL evaluation harness.

### scPerturb
#### Repo
- URL: https://github.com/sanderlab/scPerturb
- Primary language: Python (plus R package)
- Last update: 2025-02-25

#### Problem Statement
- What the repository is trying to solve: resource and statistical tests for perturbation datasets.
- Target data type (single/dual/combinatorial): single-cell perturbation datasets.
- Output type (gene scores, interactions, hits, etc.): E-distance statistics and E-tests.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): statistical distance tests.
- Key assumptions: E-distance captures perturbation differences.
- Training objective: none; compute statistics.
- Inference outputs: distance metrics and test results.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not applicable.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: `pip install scperturb` (or conda env for paper reproduction).
- Minimal run command: use `edist`/`etest` on AnnData.
- Required datasets: scperturb.org datasets or user AnnData.
- Expected outputs: E-distance metrics tables.

#### Evaluation
- Metrics used: E-distance, E-test.
- Baselines: not applicable.
- Strengths: simple install for metrics; curated datasets.
- Limitations: package maintenance slower; notebook-driven reproduction.
- Documentation quality: good for basic usage, limited for full reproducibility.
- Code quality: simple package; more robust tooling is in pertpy.
- Math quality/robustness: solid statistical test foundation.

#### Reusable Components
- Code modules worth extracting: E-distance metric implementation.
- Ideas to integrate into custom workflow: use E-distance for AL evaluation of diversity.

### scPerturBench
#### Repo
- URL: https://github.com/bm2-lab/scPerturBench
- Primary language: Python (scripts) + containerized environments
- Last update: 2026-01-06

#### Problem Statement
- What the repository is trying to solve: comprehensive benchmark of perturbation response prediction methods.
- Target data type (single/dual/combinatorial): genetic and chemical perturbation datasets.
- Output type (gene scores, interactions, hits, etc.): benchmark performance results.

#### Algorithm Overview
- Model family (Bayesian, GP, NN, linear, etc.): benchmarking of 27 methods; includes bioLord-emCell framework.
- Key assumptions: generalization should be tested across contexts and perturbations.
- Training objective: benchmark scripts per method.
- Inference outputs: performance metrics, comparison tables.

#### Active Learning / Selection Strategy
- Acquisition function (if any): none.
- Uncertainty estimation: not applicable.
- Batch strategy: not applicable.

#### Reproducibility
- Install steps: conda env or large Podman image.
- Minimal run command: `python biolord-emCell.py` (in repo) or Podman workflows.
- Required datasets: datasets from Figshare/Zenodo; large downloads.
- Expected outputs: metrics files in `Results` and scripts outputs.

#### Evaluation
- Metrics used: MSE, PCC-delta, E-distance, Wasserstein, KL, Common-DEGs.
- Baselines: 27 methods + baselines.
- Strengths: comprehensive benchmarking and reproducibility container.
- Limitations: heavy setup (large image, many envs).
- Documentation quality: detailed but long; strong reproducibility guidance.
- Code quality: script-heavy; relies on containers for stability.
- Math quality/robustness: strong benchmarking rigor; methodology in paper.

#### Reusable Components
- Code modules worth extracting: performance calculators, benchmarking scripts.
- Ideas to integrate into custom workflow: reuse metrics and generalization splits for AL evaluation.
