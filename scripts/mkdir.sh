#!/bin/bash
set -euo pipefail

BASE_DIR="data/pretrained_models"

DATASETS=("HIV" "BBBP")
VARIANTS=("TSSA" "TSID" "TSDY" "TSIS" "TSISD" "TSISO" "TSISR")
VOCAB_SIZES=("5K" "50K")

for dataset in "${DATASETS[@]}"; do
  dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
  
  for variant in "${VARIANTS[@]}"; do
    for vocab_size in "${VOCAB_SIZES[@]}"; do
      dir="${BASE_DIR}/${dataset_lower}/CANONICAL-TSMILES-${variant}-${vocab_size}-1M-20Epoch-${dataset}"
      
      echo "Creating $dir"
      mkdir -p "$dir"
    done
  done
done