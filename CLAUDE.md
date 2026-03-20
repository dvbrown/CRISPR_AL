# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Agents Guide

This project uses agent-assisted workflows focused on selecting which genes to
perturb with CRISPR in experimental campaigns.
Multiple small CRISPR experiments will be performed in an active learning workflow.

## Commands

Activate the environment before running any Python commands or tests:
```bash
source scripts/activate_env.sh
```

Install the package in development mode (run once after cloning):
```bash
pip install -e .
```

Run the full test suite:
```bash
pytest tests/
```

Run a single test file or test:
```bash
pytest tests/test_metrics.py
pytest tests/test_splits.py::test_no_leakage
```

Launch a Marimo notebook for interactive editing:
```bash
micromamba activate .micromamba/envs/crispr-al
marimo edit --watch notebooks/crispr_screen_transfer/design_a_analysis.py
```

## Code Architecture

Core logic lives in `src/crispr_al/` as six modules:

| Module | Role |
|--------|------|
| `screen.py` | Load and normalise Chen 2019 / Sharon 2019 venetoclax screens; z-score normalisation; hit-label assignment |
| `splits.py` | Generate reproducible gene-holdout splits (Design A seeds 11001–; Design B seeds 21001– / 22001–); SHA-256 split hashing |
| `features.py` | Build 9-feature gene matrix: MOLM-13 expression, DepMap co-essentiality, and five pathway-membership columns |
| `models.py` | Train Ridge (RidgeCV) and Random Forest regressors; StandardScaler fit on train only |
| `metrics.py` | Compute regression, ranking (Precision/Recall@K), and classification (AUROC/AUPRC) metrics; BCa bootstrap CIs; JSON schema validation |
| `io.py` | Parquet, JSON, and split-manifest I/O; `get_code_commit()` for reproducibility tagging |

Metric outputs must conform to `notebooks/crispr_screen_transfer/metrics.schema.json`.
Split configuration is declared in `notebooks/crispr_screen_transfer/splits.yaml`.

## Experimental Designs

Two benchmark designs drive all active development:

**Design A — within-screen holdout** (`splits.SEED_START = 11001`)
- Train on 2,000 Chen 2019 genes; predict remaining ~17,000 held-out genes from the same screen.
- 25 repeats → 50 metric JSON files (25 × 2 models).
- Nextflow pipeline: `notebooks/crispr_screen_transfer/nextflow/pipeline_a/`
- Results: `notebooks/crispr_screen_transfer/results/design_a/`

**Design B — cross-screen transfer** (`splits.XFER_SEED_START = 21001` / `XFER_SEED_START_REVERSE = 22001`)
- Train on 2,000 genes from one screen; predict all genes in the other screen.
- Chen → Sharon: 30 repeats; Sharon → Chen: 30 repeats; plus 2 overlap-only baselines.
- 122 total output files (60 splits × 2 models + 2 baselines + 6 aggregated CSVs).
- Sharon-only genes not in the feature matrix are zero-imputed (not a leakage issue).
- Nextflow pipeline: `notebooks/crispr_screen_transfer/nextflow/pipeline_b/`
- Results: `notebooks/crispr_screen_transfer/results/design_b/`

## Nextflow Pipeline Structure

Each design has its own self-contained pipeline directory:

```
notebooks/crispr_screen_transfer/nextflow/
├── pipeline_a/          # Design A pipeline (within-screen holdout)
│   ├── main.nf
│   ├── nextflow.config  # results_dir → results/design_a/
│   ├── conf/
│   └── scripts/
└── pipeline_b/          # Design B pipeline (cross-screen transfer)
    ├── main.nf
    ├── nextflow.config  # results_dir → results/design_b/
    ├── conf/
    └── scripts/
```

Run pipelines from their respective subdirectory:
```bash
cd notebooks/crispr_screen_transfer/nextflow/pipeline_a && nextflow run main.nf -profile slurm
cd notebooks/crispr_screen_transfer/nextflow/pipeline_b && nextflow run main.nf -profile slurm
```

## Default Behaviors
- Prioritize target selection logic and ranking for perturbation campaigns.
- Avoid data-processing pipelines unless they are required for selection.
- Use one micromamba environment per external repo; do not share envs.
- Use Python modules in `src/` folder for reusable computation.
- Place notebook-specific CLI scripts in `notebooks/<subfolder>/scripts` while keeping core logic in `src/`.
- Use notebooks for exploration and result visualization.
- Save tutorial analysis outputs under `notebooks/<subfolder>/outputs`.
- Keep input datasets in the top-level `data/` directory.
- Include a module-level docstring in Python scripts and provide help text for all `argparse` arguments.
- Record decisions in `docs/decision-log/`.
- Keep external credentials out of the repo.

## Visualisation Conventions
- **plotnine plots:** apply `theme_publication()` from `src/crispr_al/plotting.py` to every figure.
- **matplotlib plots:** apply the matching rcParams at notebook startup (see `notebooks/Cas12a_EuMyc/00_explore_menuetto.py` for the canonical block), including `"axes.labelweight": "bold"`.
- **Save all figures** as 300 dpi PNG to `notebooks/<notebook_folder>/figures/<notebook_name>/`, using `fig.savefig(..., dpi=300, bbox_inches="tight")` (matplotlib) or `fig.save(...)` (plotnine).

## Open Targets MCP
- Remote endpoint: `https://mcp.platform.opentargets.org/mcp`
- Docs MCP: `https://platform-docs.opentargets.org/~gitbook/mcp`

## Evaluation Flow
1. Add a repo to `docs/repo_list.md` if missing.
2. Create an eval note in `docs/evals/` using `docs/evals/template.md`.
3. Update `docs/evals/summary.md` with comparison details.
