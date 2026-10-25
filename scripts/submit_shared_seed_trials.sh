#!/usr/bin/env bash
# Preview by default. Run with --submit to prepare caches and submit all trials.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=--dry-run
preview=false
for arg in "$@"; do
  case "$arg" in
    --submit|--resume) mode=$arg ;;
    --dry-run) preview=true ;;
    *) echo "Usage: bash $0 [--submit|--resume] [--dry-run]" >&2; exit 1 ;;
  esac
done
study=outputs/shared_study_2m_v2
cache_root=data/shared_study_2m_v2/seed_trials
output_root=$study/seed_trials
receipt=$output_root/submissions.log
if [[ "$mode" == --submit && -e "$receipt" ]]; then
  echo "Receipt exists. Use --resume to skip recorded submissions." >&2
  exit 1
fi
if [[ "$mode" == --resume && ! -s "$receipt" ]]; then
  echo "Cannot resume without the original receipt: $receipt" >&2
  exit 1
fi
if [[ "$mode" == --dry-run ]]; then preview=true; fi
lock=
submitted=$(mktemp)
cleanup() {
  rm -f "$submitted"
  if [[ -n "$lock" ]]; then rmdir "$lock"; fi
}
trap cleanup EXIT
if [[ "$preview" == false ]]; then
  command -v sbatch >/dev/null
  mkdir -p "$output_root"
  if ! mkdir "$output_root/.submission-lock"; then
    echo "Another submitter may be active. Inspect .submission-lock before retrying." >&2
    exit 1
  fi
  lock=$output_root/.submission-lock
fi

# A command line is not evidence of acceptance. Only an acknowledged Slurm ID
# marks a trial submitted. An interrupted/ambiguous attempt requires inspection.
python3 - "$receipt" > "$submitted" <<'PYLOG'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
accepted = {}
name = None
resolved = True
for line in path.read_text().splitlines() if path.exists() else []:
    match = re.match(r"Submitting: .*[/]\.job_(SHARED-V2-[^/]+)\.sh$", line)
    if match:
        if name and not resolved:
            sys.exit(f"Ambiguous previous submission for {name}; inspect Slurm before retrying.")
        name, resolved = match[1], False
    match = re.fullmatch(r"Submitted batch job (\d+)", line)
    if match and name:
        if name in accepted and accepted[name] != match[1]:
            sys.exit(f"Multiple job IDs recorded for {name}; inspect receipt.")
        accepted[name] = match[1]
        resolved = True
    if line.startswith('sbatch: error: Batch job submission failed:'):
        resolved = True
if name and not resolved:
    sys.exit(f"Submission outcome unknown for {name}; check squeue/sacct before retrying. Keep the receipt.")
for name, job in accepted.items():
    print(name, job)
PYLOG
submitted_job() {
  awk -v name="$1" '$1 == name {print $2}' "$submitted"
}

# Check every input before submitting any jobs. Copy only completed downstream
# caches, including their provenance sidecars, before concurrent readers start.
for condition in base ps; do
  for rep in fragsmiles TSID TSDY TSIS; do
    key=${condition}_${rep}
    for file in "$study/$key/tokenizer/vocab.json" "$study/${key}_mlm/final_checkpoint/model.safetensors"; do
      [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
    done
    for task in hiv bbbp; do
      source=data/shared_study_2m_v2/${key}_${task}
      if [[ "$rep" == fragsmiles && "$task" == hiv ]]; then
        source=${source}_repair2
      fi
      for file in "$task.csv" "${task}_split_indices.json" "${task}_split_indices.json.meta.json"; do
        [[ -s "$source/$file" ]] || { echo "Missing completed cache: $source/$file" >&2; exit 1; }
      done
      for split in train validation test; do
        if [[ "$rep" == fragsmiles ]]; then
          pattern="${task}_*fragsmiles*_${split}_seed42.txt"
        else
          pattern="${task}_*${rep}*_${split}_seed42.txt"
        fi
        # Require strings and provenance; do not queue three cache-building jobs.
        found=false
        for file in "$source"/$pattern; do
          if [[ -s "$file" && -s "$file.meta.json" ]]; then found=true; fi
        done
        "$found" || { echo "Missing strings/metadata: $source/$pattern" >&2; exit 1; }
      done
      for seed in 42 43 44; do
        destination=$output_root/${key}_${task}/seed${seed}
        name=SHARED-V2-${condition}-${rep}-${task}-SEED${seed}
        if [[ -n "$(submitted_job "$name")" ]]; then continue; fi
        [[ ! -e "$destination" ]] || { echo "Output already exists: $destination" >&2; exit 1; }
      done
    done
  done
done
[[ -s "$study/miner/ps_merges.json" ]] || { echo 'Missing PS merge model' >&2; exit 1; }

# Existing copied caches must match their original files. Never overwrite them.
for condition in base ps; do
  for rep in fragsmiles TSID TSDY TSIS; do
    for task in hiv bbbp; do
      key=${condition}_${rep}_${task}
      source=data/shared_study_2m_v2/$key
      if [[ "$rep" == fragsmiles && "$task" == hiv ]]; then source=${source}_repair2; fi
      if [[ -e "$cache_root/$key" ]]; then
        for file in "$source"/*; do
          [[ -f "$file" ]] || continue
          cmp -s "$file" "$cache_root/$key/$(basename "$file")" || {
            echo "Existing cache differs or is incomplete: $cache_root/$key" >&2; exit 1;
          }
        done
      fi
    done
  done
done
if [[ "$preview" == false ]]; then
  mkdir -p "$cache_root"
  for condition in base ps; do
    for rep in fragsmiles TSID TSDY TSIS; do
      for task in hiv bbbp; do
        key=${condition}_${rep}_${task}
        source=data/shared_study_2m_v2/$key
        if [[ "$rep" == fragsmiles && "$task" == hiv ]]; then source=${source}_repair2; fi
        if [[ ! -e "$cache_root/$key" ]]; then cp -R "$source" "$cache_root/$key"; fi
      done
    done
  done
fi

for condition in base ps; do
  for rep in fragsmiles TSID TSDY TSIS; do
    key=${condition}_${rep}
    if [[ "$rep" == fragsmiles ]]; then
      representation=(algorithm=fragsmiles_classifier dataset.use_fragsmiles=true dataset.use_tsmiles=false)
    else
      representation=(algorithm=tsmiles_classifier dataset.use_fragsmiles=false dataset.use_tsmiles=true "dataset.tsmiles_variant=$rep")
    fi
    merges=null
    if [[ "$condition" == ps ]]; then merges=$study/miner/ps_merges.json; fi
    for task in hiv bbbp; do
      for seed in 42 43 44; do
        name=SHARED-V2-${condition}-${rep}-${task}-SEED${seed}
        job=$(submitted_job "$name")
        if [[ -n "$job" ]]; then
          echo "Skipping $name: already submitted as $job"
          continue
        fi
        command=(bash scripts/submit_gpu.sh "$name"
          experiment=downstream_classification "dataset=molnet_$task"
          "${representation[@]}" dataset.fragmentation=exocyclic
          "dataset.ps_merges_path=$merges" dataset.seed=42
          "dataset.data_dir=$cache_root/${key}_${task}"
          "algorithm.tokenizer.pretrained=$study/$key/tokenizer"
          "algorithm.model.pretrained=$study/${key}_mlm/final_checkpoint"
          "experiment.training.seed=$seed" "experiment.training.data_seed=$seed"
          experiment.training.num_train_epochs=5 experiment.training.early_stopping_patience=3
          wandb.mode=online wandb.entity=ncduy0303 wandb.project=molecule-tokenization
          "hydra.run.dir=$output_root/${key}_${task}/seed${seed}")
        if [[ "$preview" == false ]]; then
          if ! WANDB_TAGS="shared_study_2m_v2,$condition,$rep,$task,seed$seed" "${command[@]}" 2>&1 | tee -a "$receipt"; then
            echo "Submission stopped. Keep submissions.log; use --resume after resolving the Slurm error." >&2
            exit 1
          fi
        else
          printf '%q ' "${command[@]}"
          printf '\n'
        fi
      done
    done
  done
done
