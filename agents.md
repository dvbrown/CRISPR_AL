# Agents Guide

This project uses agent-assisted workflows focused on selecting which genes to
perturb with CRISPR in experimental campaigns.
Multiple small CRISPR experiments will be performed in an active learning workflow.

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

## Open Targets MCP
- Remote endpoint: `https://mcp.platform.opentargets.org/mcp`
- Docs MCP: `https://platform-docs.opentargets.org/~gitbook/mcp`

## Evaluation Flow
1. Add a repo to `docs/repo_list.md` if missing.
2. Create an eval note in `docs/evals/` using `docs/evals/template.md`.
3. Update `docs/evals/summary.md` with comparison details.
