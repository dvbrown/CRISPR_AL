# Candidate Repositories

## Active learning Focused
- [GEARS](https://github.com/snap-stanford/GEARS)
  - Geometric deep learning model that predicts transcriptional outcomes for single and multi-gene perturbations in single-cell screens.
- [Iterative Perturb-Seq](https://github.com/Genentech/iterative-perturb-seq)
  - Sequential experimental design framework for Perturb-seq that uses multimodal priors to select informative perturbations.

## Combinatorial / Dual-Guide / Interaction-Focused
- [gimap](https://github.com/FredHutch/gimap)
  - R package for analyzing paired-guide CRISPR screens and estimating genetic interaction scores such as synthetic lethality.
- [NAIAD](https://github.com/NeptuneBio/NAIAD)
  - Active learning system that models combinatorial perturbation outcomes and recommends gene pairs for follow-up screens.
- [DiscoBAX](https://github.com/amehrjou/DiscoBAX)
  - Bayesian optimization approach for discovering diverse, high-impact genomic intervention sets with minimal experiments.

## Perturbation prediction

- [GPerturb](https://github.com/hwxing3259/GPerturb)
- [GenePert](https://github.com/zou-group/GenePert)
  - Regression-based model using GenePT embeddings to predict gene expression changes from genetic perturbations.
- [scGenePT](https://github.com/czi-ai/scGenePT)
  - Single-cell perturbation prediction suite that injects language-derived gene embeddings into scGPT models.
- [state](https://github.com/ArcInstitute/state)
  - State transition and embedding models with a CLI for predicting cellular responses to perturbations across contexts.
- [AttentionPert](https://github.com/BaiDing1234/AttentionPert)
  - Attention-based model for predicting multiplexed genetic perturbation effects with multi-scale representations.
- [scPRAM](https://github.com/jiang-q19/scPRAM)
  - Attention-driven model for predicting single-cell gene expression responses to perturbations.

## Benchmarking Frameworks
- [perturbench](https://github.com/altoslabs/perturbench/)
  - Benchmarking framework with datasets, metrics, and evaluation pipelines for single-cell perturbation prediction models.
- [scPerturb](https://github.com/sanderlab/scPerturb)
  - Resource and Python/R toolkit for curating and analyzing single-cell perturbation datasets.
- [scPerturBench](https://github.com/bm2-lab/scPerturBench)
  - Benchmark suite for evaluating single-cell perturbation response prediction methods across contexts and perturbations.

## Notes
- Active learning specific CRISPR repos are sparse; we will adapt acquisition logic
  from these analysis pipelines when possible.
- Add new repos here as they are discovered and evaluated.
