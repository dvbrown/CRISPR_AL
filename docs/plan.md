# CRISPR Active Learning Exploration Plan

## Goals
- Evaluate multiple GitHub codebases for active learning and combinatorial CRISPR screens.
- Extract reusable modeling, acquisition, and evaluation components.
- Build a custom workflow that integrates Open Targets evidence.

## Scope
- Focus on CRISPR screen analysis with emphasis on combinatorial or dual-guide screens.
- Use Python modules for reusable compute; notebooks for inspection and reporting.
- Integrate Open Targets MCP via the remote hosted endpoint.

## Milestones
1. Curate candidate repositories and capture evaluation notes.
2. Build a minimal evaluation harness with shared metrics.
3. Prototype Open Targets data access and feature integration.
4. Combine the strongest elements into a unified workflow.

## Conventions
- Code in `src/` with clear module boundaries.
- Repo evaluations in `docs/evals/` following the template.
- Store raw data in `data/` and outputs in `results/`.

## Open Targets MCP
- Remote MCP endpoint: `https://mcp.platform.opentargets.org/mcp`.
- MCP docs endpoint: `https://platform-docs.opentargets.org/~gitbook/mcp`.
- Keep credentials outside the repo, use environment variables.
