# CRISPR Active Learning Exploration Plan

## Goals
- Evaluate multiple GitHub codebases for active learning and combinatorial CRISPR screens.
- Extract reusable target-selection, acquisition, and evaluation components.
- Build a custom workflow that integrates Open Targets evidence for gene choice.

## Scope
- Focus on gene selection for CRISPR perturbation campaigns with emphasis on combinatorial
  or dual-guide screens.
- Avoid data-processing pipelines unless required to support selection decisions.
- Use Python modules for reusable compute; notebooks for inspection and reporting.
- Integrate Open Targets MCP via the remote hosted endpoint.

## Milestones

### 1. Curate candidate repositories and capture evaluation notes — COMPLETE
- Candidate list maintained in `docs/repo_list/repo_list.md`.
- Evaluation notes in `docs/evals/` (template, summary, Olivieri 2020).

### 2. Build a minimal selection evaluation harness with shared metrics — COMPLETE
- Core package at `src/crispr_al/` (screen, splits, features, models, metrics, io, plotting).
- JSON schema validation for all metric outputs (`metrics.schema.json`).
- Full pytest suite in `tests/`.
- Design A (within-screen holdout) and Design B (cross-screen transfer) benchmarks
  implemented and running via Nextflow pipelines (SLURM + local profiles).
- Results archived in `notebooks/crispr_screen_transfer_hold/results/`.

### 3. Prototype Open Targets data access and feature integration — IN PROGRESS
- Remote MCP endpoint active (`https://mcp.platform.opentargets.org/mcp`).
- Feature matrix currently uses DepMap, CCLE, and pathway annotations.
- Open Targets evidence not yet incorporated as model features.

### 4. Extend benchmarks to additional screens and contexts — IN PROGRESS
- Olivieri 2020 genotoxic benchmark (RPE1-hTERT) underway in
  `notebooks/RPE1-hTERT_genotoxic/`; LODO splits and 6-feature variant implemented.
- Cas12a EuMyc genotoxic screens underway in `notebooks/Cas12a_EuMyc/`; stratified
  and seeded split strategies implemented in `splits.py`.
- Elling 2024 screens available; cross-screen transfer benchmarks pending.

### 5. Combine the strongest elements into a unified active learning workflow — PENDING
- Acquisition strategy design (uncertainty sampling, expected improvement, etc.).
- Integration of Open Targets evidence as features or priors.
- Iterative experiment design loop.

## Conventions
- Code in `src/` with clear module boundaries.
- Repo evaluations in `docs/evals/` following the template.
- Store raw data in `data/` and outputs in `results/` under the relevant notebook workstream.
- Record architectural decisions in `docs/decision-log/`.

## Open Targets MCP
- Remote MCP endpoint: `https://mcp.platform.opentargets.org/mcp`.
- MCP docs endpoint: `https://platform-docs.opentargets.org/~gitbook/mcp`.
- Keep credentials outside the repo, use environment variables.
