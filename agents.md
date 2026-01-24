# Agents Guide

This project uses agent-assisted workflows for repeatable analysis.

## Default Behaviors
- Use Python modules in `src/` folder for reusable computation.
- Use notebooks for exploration and result visualization.
- Record decisions in `docs/decision-log/`.
- Keep external credentials out of the repo.

## Open Targets MCP
- Remote endpoint: `https://mcp.platform.opentargets.org/mcp`
- Docs MCP: `https://platform-docs.opentargets.org/~gitbook/mcp`

## Evaluation Flow
1. Add a repo to `docs/repo_list.md` if missing.
2. Create an eval note in `docs/evals/` using `docs/evals/template.md`.
3. Update `docs/evals/summary.md` with comparison details.
