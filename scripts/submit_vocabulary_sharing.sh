#!/bin/bash
#SBATCH --job-name=VOCABULARY-SHARING
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/out_%j.out
#SBATCH --error=slurm_logs/err_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -u -m scripts.analyze_vocabulary_sharing \
  --limit 10000 \
  --output "${1:-outputs/vocabulary_sharing/full_${SLURM_JOB_ID}}"
