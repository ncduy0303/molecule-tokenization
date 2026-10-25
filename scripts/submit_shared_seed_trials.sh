#!/usr/bin/env bash
# Preview by default. Run with --submit to prepare caches and submit all trials.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:---dry-run}
if [[ "$mode" != --dry-run && "$mode" != --submit ]]; then
  echo "Usage: bash scripts/submit_shared_seed_trials.sh [--dry-run|--submit]" >&2
  exit 1
fi
study=outputs/shared_study_2m_v2
cache_root=data/shared_study_2m_v2/seed_trials
output_root=$study/seed_trials
receipt=$output_root/submissions.log
if [[ -e "$receipt" ]]; then
  echo "Submission receipt already exists: $receipt. Inspect it before submitting again." >&2
  exit 1
fi

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
        [[ ! -e "$destination" ]] || { echo "Output already exists: $destination" >&2; exit 1; }
      done
    done
  done
done
[[ -s "$study/miner/ps_merges.json" ]] || { echo 'Missing PS merge model' >&2; exit 1; }

if [[ "$mode" == --submit ]]; then
  command -v sbatch >/dev/null
  mkdir -p "$cache_root" "$output_root"
  for condition in base ps; do
    for rep in fragsmiles TSID TSDY TSIS; do
      for task in hiv bbbp; do
        key=${condition}_${rep}_${task}
        source=data/shared_study_2m_v2/$key
        if [[ "$rep" == fragsmiles && "$task" == hiv ]]; then source=${source}_repair2; fi
        # Never merge a fresh source into a possibly partial existing copy.
        [[ ! -e "$cache_root/$key" ]] || { echo "Cache already exists: $cache_root/$key" >&2; exit 1; }
      done
    done
  done
  for condition in base ps; do
    for rep in fragsmiles TSID TSDY TSIS; do
      for task in hiv bbbp; do
        key=${condition}_${rep}_${task}
        source=data/shared_study_2m_v2/$key
        if [[ "$rep" == fragsmiles && "$task" == hiv ]]; then source=${source}_repair2; fi
        cp -R "$source" "$cache_root/$key"
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
        if [[ "$mode" == --submit ]]; then
          WANDB_TAGS="shared_study_2m_v2,$condition,$rep,$task,seed$seed" "${command[@]}" 2>&1 | tee -a "$receipt"
        else
          printf '%q ' "${command[@]}"
          printf '\n'
        fi
      done
    done
  done
done
