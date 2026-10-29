#!/usr/bin/env bash
# Run from repository root: bash scripts/submit_external_tsdy_preflight.sh [--resume]
set -euo pipefail
cd "$(dirname "$0")/.."
args=()
if [[ $# -gt 0 ]]; then
  [[ $# -eq 1 && "$1" == --resume ]] || { echo "Usage: bash $0 [--resume]" >&2; exit 1; }
  args=(--resume)
fi
[[ -x .venv/bin/python ]] || { echo 'Missing .venv/bin/python' >&2; exit 1; }
[[ -s data/shared_study_2m_v2/miner/corpus.txt ]] || { echo 'Missing original 2M corpus' >&2; exit 1; }
mkdir -p slurm_logs
script=slurm_logs/.job_ENHANCED-TSDY-V3-PREFLIGHT-2M.sh
cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name=ENHANCED-TSDY-V3-PREFLIGHT-2M
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=$PWD/slurm_logs/out_%j.out
#SBATCH --error=$PWD/slurm_logs/err_%j.err
set -euo pipefail
EOF
printf 'cd %q\n' "$PWD" >> "$script"
printf '.venv/bin/python -u -m scripts.prepare_external_tsdy --workers 8 %s\n' "${args[*]}" >> "$script"
sbatch "$script"
