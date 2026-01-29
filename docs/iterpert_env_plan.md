# IterPert micromamba install plan

## Prereqs
- Ensure micromamba is installed in the container.
- Decide env root path. Recommended: `.micromamba/envs` under the repo.
- Decide CPU vs CUDA. Default: CPU unless GPU is available.

## Steps
1) Verify micromamba:
   - `micromamba --version`
2) Set the env root so the env lives in the repo:
   - `export MAMBA_ROOT_PREFIX=/home/dandroid/Code/CRISPR_AL/.micromamba`
3) Create the env (per IterPert README):
   - `micromamba create -y -n iterpert_env python=3.8`
4) Install PyG:
   - CPU default: `micromamba install -y -n iterpert_env -c pyg -c pytorch pyg`
   - If CUDA is needed, select the correct PyTorch/CUDA combo first, then PyG.
5) Install Python deps:
   - `micromamba run -n iterpert_env pip install -r /home/dandroid/Code/CRISPR_AL/external/iterative-perturb-seq/requirements.txt`
6) Install IterPert from source:
   - `micromamba run -n iterpert_env pip install -e /home/dandroid/Code/CRISPR_AL/external/iterative-perturb-seq`
7) Test loading:
   - `micromamba run -n iterpert_env python -c "import iterpert; import torch_geometric; print('ok')"`

## References
- `external/iterative-perturb-seq/README.md`
- `external/iterative-perturb-seq/requirements.txt`
