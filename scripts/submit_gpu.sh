#!/bin/bash
# Usage: ./scripts/submit_gpu.sh <run_name> [hydra overrides...]
# Example: ./scripts/submit_gpu.sh my_run experiment=example_classification dataset=example_cifar10 algorithm=example_classifier
set -euo pipefail

# --- Configuration (edit these) ---
PARTITION="gpu"
GRES="gpu:h100-96:1" 
# GRES="gpu:a100-40:1" # For testing when h100s are not available
CPUS=8
MEM="32G"
TIME="03:00:00"
# TIME="00:15:00" # For higher priority testing jobs
# ----------------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [hydra overrides...]"
    echo "Example: $0 my_run experiment=example_classification dataset=example_cifar10 algorithm=example_classifier"
    exit 1
fi

NAME="$1"
shift
HYDRA_ARGS="$*"

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create log directory
mkdir -p "$PROJECT_ROOT/slurm_logs"

# Write a proper job script (avoids --wrap quoting issues)
JOB_SCRIPT="$PROJECT_ROOT/slurm_logs/.job_${NAME}.sh"
cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${PROJECT_ROOT}/slurm_logs/out_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm_logs/err_%j.err

cd ${PROJECT_ROOT}
source .venv/bin/activate
nvidia-smi
python -m main +name=${NAME} ${HYDRA_ARGS}
JOBEOF

chmod +x "$JOB_SCRIPT"
echo "Submitting: $JOB_SCRIPT"
echo "Command:    python -m main +name=${NAME} ${HYDRA_ARGS}"
sbatch "$JOB_SCRIPT"
