#!/bin/bash
# Activate the MLDE environment
# Usage: source scripts/activate_env.sh

set -euo pipefail

# Load micromamba module
module load micromamba

# Initialize micromamba shell hook
eval "$(micromamba shell hook --shell=bash)"

# Use absolute path for SLURM compatibility
micromamba activate /home/users/allstaff/brown.d/vast_project_Protein/Repos/CRISPR_AL/.micromamba/envs/crispr-al
