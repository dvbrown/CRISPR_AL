# CRISPR Active Learning Lab

This workspace is for evaluating CRISPR screening codebases with a focus on
gene selection for perturbation campaigns, especially combinatorial and
dual-guide contexts. Heavy computation lives in Python modules, while notebooks
are used to run experiments and inspect results.

## Repository Layout

```
CRISPR_AL/
├── src/crispr_al/          # Core Python package (screen, splits, features, models, metrics, io)
├── notebooks/
│   ├── crispr_screen_transfer/   # Design A & B benchmarks (active development)
│   ├── bulk_single_dual/         # Bulk epistasis tutorial
│   ├── iterpert/                 # IterPert tutorial (K562)
│   └── scanpy/                   # scRNA-seq / AnnData tutorial
├── pipelines/
│   └── design_a/                 # Nextflow pipeline for Design A runs
├── tests/                        # pytest suite for all src/ modules
├── data/                         # Input datasets (see data/README.md)
├── external/                     # Vendored external repos (iterative-perturb-seq)
├── scripts/                      # Utility scripts (env activation, data fetch, checksums)
├── config/                       # Apptainer container definition
├── manifests/                    # Dataset registry (registry.yaml)
└── docs/                         # Planning, decision log, repo evaluations
```

## Getting Started

### 1. Set up the environment

```bash
micromamba create -f environment.yml -p .micromamba/envs/crispr-al
source scripts/activate_env.sh
```

### 2. Install the package

```bash
pip install -e .
```

### 3. Run the tests

```bash
pytest tests/
```

## Active Development: CRISPR Screen Transfer

The primary workstream is `notebooks/crispr_screen_transfer/`, which benchmarks
gene-essentiality transfer between two venetoclax CRISPR screens (Chen 2019 and Sharon 2019).

**Design A — within-screen holdout**
- Train on 2,000 Chen 2019 genes; predict the remaining ~17,000 from the same screen.
- 25 repeats × 2 models = 50 metric JSON files.

**Design B — cross-screen transfer**
- Train on one screen, predict all genes in the other (Chen → Sharon and reverse).
- 30 repeats × 2 directions × 2 models + 2 baselines + 6 aggregated CSVs = 122 output files.

Key artifacts in `notebooks/crispr_screen_transfer/`:

| File | Purpose |
|------|---------|
| `design_a_analysis.py` | Marimo notebook — Design A runs and visualisation |
| `splits.yaml` | Canonical split configuration (seeds, n_train, repeats) |
| `metrics.schema.json` | JSON schema that all metric output files must pass |
| `design_a_implementation_handoff.md` | Design A implementation notes |
| `design_b_implementation_handoff.md` | Design B implementation notes |

## Launching Marimo Notebooks

From a terminal:
```bash
micromamba activate .micromamba/envs/crispr-al
marimo edit --watch notebooks/crispr_screen_transfer/design_a_analysis.py
```

Notes:
- Do not pass `--headless`; marimo will open in your system browser.
- If the browser does not auto-open, use the `http://127.0.0.1:<port>` URL printed in the terminal.

From VS Code (task):
1. Open the notebook file you want to run.
2. Run `Tasks: Run Task`.
3. Choose `Marimo: Edit Active Notebook (Watch, micromamba)`.

## Data Layout

See `data/README.md` for the local data layout, registry usage, and scripts.
Small, tracked test fixtures live under `tests/data/`.

Key datasets in `data/bulk/`:
- `chen2019_venetoclax/` — Chen 2019 CRISPR screen
- `sharon2019_venetoclax/` — Sharon 2019 CRISPR screen
- `ccle_expression/` — MOLM-13 gene expression (CCLE)
- `depmap_crispr_gene_effect/` — DepMap co-essentiality scores
- `pathway_annotations/` — Pathway membership features

## Nextflow Pipeline

A Nextflow pipeline for Design A batch runs lives in `pipelines/design_a/`:
```bash
nextflow run pipelines/design_a/main.nf -c pipelines/design_a/nextflow.config
```

## Micromamba Environments

This project evaluates multiple external repositories with conflicting
dependencies, so use one micromamba environment per repo rather than a single
shared environment.

- `repos/` for cloned repositories.
- `repos/.envs/<repo>/` for micromamba env prefixes.
- Set `MAMBA_ROOT_PREFIX` and `CONDA_PKGS_DIRS` to a writable path inside the
  bound workspace (for Apptainer usage).

Record the environment prefix and any install notes in the repo's evaluation
note in `docs/evals/`.

## Open Targets MCP

The remote Open Targets MCP endpoint is used for external evidence:
- Endpoint: `https://mcp.platform.opentargets.org/mcp`
- Docs MCP: `https://platform-docs.opentargets.org/~gitbook/mcp`

Keep any credentials out of the repo and use environment variables instead.

## Evaluation Flow

1. Review `docs/plan.md` for goals and milestones.
2. Check `docs/repo_list/repo_list.md` for candidate repositories.
3. Use `docs/evals/template.md` to evaluate a repo and update `docs/evals/summary.md`.
4. Record architectural decisions in `docs/decision-log/`.
