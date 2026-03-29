# Eval Summary

## Benchmark Datasets

| Dataset | Task | Models | Best AUROC | Notes |
|---|---|---|---|---|
| [Olivieri 2020](olivieri2020.md) | Genotoxic screen transfer (RPE1-hTERT, 30 screens, 27 agents) | Ridge, RF | RF ≈ 0.82 (cross-library + LODO); ≈ 0.58 within-screen | Pathway-only features (no DepMap/CCLE); RF jumps to 0.82 with paired screen; Ridge plateaus at ≈ 0.64–0.68 across all aims |

## External Repo Evals

See [template.md](template.md) for full evaluations of external repos (GEARS, IterPert, gimap, NAIAD, DiscoBAX, GenePert, scGenePT, state, AttentionPert, scPRAM, GPerturb, PerturbNet, scLAMBDA, perturbench, scPerturb, scPerturBench).
