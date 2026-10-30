#!/bin/bash
#SBATCH --job-name=INTRINSIC-FROZEN-VOCABS
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/out_%j.out
#SBATCH --error=slurm_logs/err_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fyp-intrinsic-mpl-${SLURM_JOB_ID}"
.venv/bin/python -m unittest tests.test_intrinsic_evaluation
.venv/bin/python -u -m scripts.evaluate_intrinsic \
  --output outputs/intrinsic_study/full \
  --pubchem-limit 10000 --workers 4
