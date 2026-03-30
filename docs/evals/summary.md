# Eval Summary

## Benchmark Datasets

| Dataset | Task | Models | Best AUROC | Notes |
|---|---|---|---|---|
| [Olivieri 2020](olivieri2020.md) | Genotoxic screen transfer (RPE1-hTERT, 30 screens, 27 agents) | Ridge, RF | RF 0.82 (Aims 2–3), 0.55 (Aim 1); Ridge 0.60–0.68 across all aims | Pathway-only features (no DepMap/CCLE); RF P@50=0.37 cross-library, 0.31 LODO; Ridge near-chance at P@50 (≈0.03) |

## External Repo Evals

See [template.md](template.md) for full evaluations of external repos (GEARS, IterPert, gimap, NAIAD, DiscoBAX, GenePert, scGenePT, state, AttentionPert, scPRAM, GPerturb, PerturbNet, scLAMBDA, perturbench, scPerturb, scPerturBench).
