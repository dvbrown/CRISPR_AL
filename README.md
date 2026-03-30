# CRISPR Active Learning Lab

This workspace benchmarks gene-essentiality prediction for CRISPR perturbation
campaign design, with a focus on gene selection, cross-screen transfer, and
active learning strategies. Heavy computation lives in Python modules under
`src/crispr_al/`; Marimo notebooks run experiments and inspect results.

## Repository Layout

```
CRISPR_AL/
├── src/crispr_al/                     # Core Python package (7 modules)
├── notebooks/
│   ├── crispr_screen_transfer_hold/   # Design A & B benchmarks (active development)
│   ├── Cas12a_EuMyc/                  # Cas12a genotoxic screens (EuMyc mouse model)
│   ├── RPE1-hTERT_genotoxic/          # Olivieri 2020 genotoxic benchmark (RPE1-hTERT)
│   ├── crispr_star/                   # CRISPR-STAR data exploration
│   ├── bulk_single_dual/              # Bulk epistasis tutorial
│   ├── iterpert/                      # IterPert tutorial (K562)
│   └── scanpy/                        # scRNA-seq / AnnData tutorial
├── tests/                             # pytest suite for all src/ modules
├── data/                              # Input datasets (see data/README.md)
├── external/                          # Vendored external repos
├── scripts/                           # Utility scripts (env activation, data fetch, checksums)
│   ├── activate_env.sh
│   ├── fetch_dataset.py
│   ├── verify_checksums.py
│   ├── elling2024/                    # Scripts for Elling 2024 dataset processing
│   └── olivieri2020/                  # Scripts for Olivieri 2020 dataset processing
├── config/                            # Apptainer container definition
├── manifests/                         # Dataset registry (registry.yaml)
└── docs/                              # Planning, decision log, repo evaluations
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

The primary workstream is `notebooks/crispr_screen_transfer_hold/`, which
benchmarks gene-essentiality transfer between two venetoclax CRISPR screens
(Chen 2019 and Sharon 2019).

**Design A — within-screen holdout**
- Train on 2,000 Chen 2019 genes; predict the remaining ~17,000 from the same screen.
- 25 repeats × 2 models = 50 metric JSON files.
- Pipeline: `notebooks/crispr_screen_transfer_hold/nextflow/pipeline_a/`

**Design B — cross-screen transfer**
- Train on one screen, predict all genes in the other (Chen → Sharon and reverse).
- 30 repeats × 2 directions × 2 models + 2 baselines + 6 aggregated CSVs = 122 output files.
- Pipeline: `notebooks/crispr_screen_transfer_hold/nextflow/pipeline_b/`

Key artifacts in `notebooks/crispr_screen_transfer_hold/`:

| File | Purpose |
|------|---------|
| `design_a_analysis.py` | Marimo notebook — Design A runs and visualisation |
| `splits.yaml` | Canonical split configuration (seeds, n_train, repeats) |
| `metrics.schema.json` | JSON schema that all metric output files must pass |

## Other Notebook Workstreams

| Notebook | Purpose |
|----------|---------|
| `Cas12a_EuMyc/` | Cas12a genotoxic screens in an EuMyc mouse model |
| `RPE1-hTERT_genotoxic/` | Olivieri 2020 genotoxic screen benchmark (RPE1-hTERT, Cas12a) |
| `crispr_star/` | CRISPR-STAR data exploration |
| `bulk_single_dual/` | Active learning tutorial on bulk epistasis data |
| `iterpert/` | IterPert tutorial (K562 perturbation-seq) |
| `scanpy/` | scRNA-seq / AnnData tutorial |

## Launching Marimo Notebooks

From a terminal:
```bash
micromamba activate .micromamba/envs/crispr-al
marimo edit --watch notebooks/crispr_screen_transfer_hold/design_a_analysis.py
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
- `chen2019_venetoclax/` — Chen 2019 venetoclax CRISPR screen (MOLM-13, Brunello library)
- `sharon2019_venetoclax/` — Sharon 2019 venetoclax CRISPR screen (MOLM-13-R1, TKOv1)
- `ccle_expression/` — MOLM-13 gene expression (CCLE, log2(TPM+1))
- `depmap_crispr_gene_effect/` — DepMap co-essentiality (Chronos scores)
- `pathway_annotations/` — Reactome, GO, Hallmarks, KEGG pathway membership
- `menuetto_scherzo_2025/` — Cas12a EuMyc genotoxic screens (processed)
- `olivieri2020/` — Olivieri 2020 RPE1-hTERT genotoxic screen (NormZ format)
- `elling2024/` — Elling 2024 screens (GEO supplementary files)

## Nextflow Pipelines

Pipelines are self-contained within their notebook workstream directory.

**Design A** (within-screen holdout):
```bash
cd notebooks/crispr_screen_transfer_hold/nextflow/pipeline_a
nextflow run main.nf -profile slurm
```

**Design B** (cross-screen transfer):
```bash
cd notebooks/crispr_screen_transfer_hold/nextflow/pipeline_b
nextflow run main.nf -profile slurm
```

Both pipelines support `slurm` and `local` profiles. Results land in
`notebooks/crispr_screen_transfer_hold/results/design_a/` and `design_b/`.

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
