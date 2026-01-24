# CRISPR Active Learning Lab

This workspace is for evaluating CRISPR screening codebases with a focus on
combinatorial and dual-guide screens, then combining the strongest ideas into
one custom workflow. Heavy computation lives in Python modules, while notebooks
are used to run experiments and inspect results.

## Repository Layout
- `docs/` planning, decisions, and repo evaluations
- `src/` reusable Python modules and pipelines
- `notebooks/` analysis and visualization notebooks
- `data/` input datasets (not tracked unless small)
- `results/` outputs from runs and evaluations

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
