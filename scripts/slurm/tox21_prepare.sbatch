#!/bin/bash
#SBATCH --job-name=TOX21-SR-p53-PREPARE
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/tox21_prepare_%j.out
#SBATCH --error=slurm_logs/tox21_prepare_%j.err

# Submit from the repository root, after creating slurm_logs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
test -x .venv/bin/python
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python -m unittest tests.test_tox21_extension
.venv/bin/python -u -m scripts.prepare_tox21_extension --workers "${SLURM_CPUS_PER_TASK}" --resume
