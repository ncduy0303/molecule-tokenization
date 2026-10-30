#!/bin/bash
#SBATCH --job-name=TOX21-SR-p53
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=03:00:00
#SBATCH --array=0-8%3
#SBATCH --output=slurm_logs/tox21_finetune_%A_%a.out
#SBATCH --error=slurm_logs/tox21_finetune_%A_%a.err

# Indices 0-2: enhanced, 3-5: TSDY, 6-8: fragsmiles; seeds 42,43,44.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
export TOKENIZERS_PARALLELISM=false
export WANDB_LOG_MODEL=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python -u -m scripts.run_tox21_array_task --index "${SLURM_ARRAY_TASK_ID}"
