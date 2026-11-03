#!/bin/bash
#SBATCH --job-name=CA-V2-prepare
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ca_prepare_%j.out
#SBATCH --error=slurm_logs/ca_prepare_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p data/ca_seed_trials
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
flock -n data/ca_seed_trials/.prepare.lock .venv/bin/python -u -m scripts.ca_seed_trials prepare
