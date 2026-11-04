#!/usr/bin/env bash
# Pass resource options to sbatch and Hydra overrides after this file's name.
#SBATCH --ntasks=1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec .venv/bin/python -u -m main "$@"
