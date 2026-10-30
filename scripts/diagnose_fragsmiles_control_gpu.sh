#!/bin/bash
#SBATCH --job-name=DIAG-FRAGSMILES-FULL-HIV
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/out_%j.out
#SBATCH --error=slurm_logs/err_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
.venv/bin/python -m scripts.diagnose_fragsmiles_control legacy_loader
.venv/bin/python -m scripts.diagnose_fragsmiles_control prepared_loader
.venv/bin/python -m scripts.diagnose_fragsmiles_control summarize
