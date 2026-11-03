#!/bin/bash
#SBATCH --job-name=CA-V2-trials
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/ca_trial_%A_%a.out
#SBATCH --error=slurm_logs/ca_trial_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# PCATT's Linux extension dynamically links oneTBB. GPU nodes may lack
# the system library available on the CPU preparation node.
export LD_LIBRARY_PATH="$PWD/.ca-runtime/lib:$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
.venv/bin/python -u -m scripts.ca_seed_trials run "${SLURM_ARRAY_TASK_ID:?}"
