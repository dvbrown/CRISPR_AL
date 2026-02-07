# CRISPR Active Learning Lab

This workspace is for evaluating CRISPR screening codebases with a focus on
gene selection for perturbation campaigns, especially combinatorial and
dual-guide contexts. Heavy computation lives in Python modules, while notebooks
are used to run experiments and inspect results.

## Repository Layout
- `docs/` planning, decisions, and repo evaluations
- `src/` reusable Python modules and pipelines
- `notebooks/` analysis and visualization notebooks
- `data/` input datasets (not tracked unless small)
- `results/` outputs from runs and evaluations

## Data Layout
See `data/README.md` for the local data layout, registry usage, and scripts.
Small, tracked test fixtures live under `tests/data/`.

## Open Targets MCP
The remote Open Targets MCP endpoint is used for external evidence:
- Endpoint: `https://mcp.platform.opentargets.org/mcp`
- Docs MCP: `https://platform-docs.opentargets.org/~gitbook/mcp`

Keep any credentials out of the repo and use environment variables instead.

## Getting Started
1. Review `docs/plan.md` for goals and milestones.
2. Check `docs/repo_list.md` for candidate repositories.
3. Use `docs/evals/template.md` to evaluate a repo and update
   `docs/evals/summary.md`.

## Micromamba Environments
This project evaluates multiple external repositories with conflicting
dependencies, so use one micromamba environment per repo rather than a single
shared environment.

Recommended layout:
- `repos/` for cloned repositories.
- `repos/.envs/<repo>/` for micromamba env prefixes.
- Set `MAMBA_ROOT_PREFIX` and `CONDA_PKGS_DIRS` to a writable path inside the
  bound workspace (for Apptainer usage).

Record the environment prefix and any install notes in the repo's evaluation
note in `docs/evals/`.

### Global micromamba env

There is a global micromamba env for running generic notebooks
```bash
micromamba create -f environment.yml -p .micromamba.envs/crispr-al
micromamba activate .micromamba.envs/crispr-al
python -m ipykernel install --user --name crispr-al --display-name "Python (crispr-al)"
```