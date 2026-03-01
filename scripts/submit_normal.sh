#!/bin/bash
# Usage: ./scripts/submit_normal.sh <run_name> [hydra overrides...]
set -euo pipefail

# --- Configuration (Optimized for AMD EPYC Nodes) ---
PARTITION="normal"
# Targeting the specific xcnf node range for AMD EPYC 7763
NODELIST="xcnf[0-25]" 
CPUS=16               # Increased slightly as EPYC has 64 cores per node
MEM="512G"            # Increased; these nodes have 1TB available
TIME="03:00:00"       # +200 for jobs 15 mins or less
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
#SBATCH --nodelist=${NODELIST}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${PROJECT_ROOT}/slurm_logs/out_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm_logs/err_%j.err

cd ${PROJECT_ROOT}
source .venv/bin/activate

python -m main +name=${NAME} ${HYDRA_ARGS}
JOBEOF

chmod +x "$JOB_SCRIPT"
echo "Submitting: $JOB_SCRIPT"
echo "Command:    python -m main +name=${NAME} ${HYDRA_ARGS}"
sbatch "$JOB_SCRIPT"
