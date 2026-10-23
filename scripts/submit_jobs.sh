#!/bin/bash
set -euo pipefail

REPEATS=3
SLEEP_TIME=45

DATASETS=("HIV" "BBBP")
VARIANTS=("TSSA" "TSID" "TSDY" "TSIS" "TSISD" "TSISO" "TSISR")
VOCAB_SIZES=("5K" "50K")

for dataset in "${DATASETS[@]}"; do
  dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
  
  for variant in "${VARIANTS[@]}"; do
    for vocab_size in "${VOCAB_SIZES[@]}"; do
        run_name="CANONICAL-TSMILES-${variant}-${vocab_size}-1M-20Epoch-${dataset}"
        
        cmd="./scripts/submit_gpu.sh ${run_name} \
            experiment=downstream_classification \
            dataset=molnet_${dataset_lower} \
            algorithm=tsmiles_classifier \
            dataset.use_tsmiles=true \
            dataset.tsmiles_variant=\"${variant}\" \
            algorithm.tokenizer.pretrained=\"data/pretrained_tokenizers/CANONICAL_TSMILES_${variant}_${vocab_size}\" \
            algorithm.model.pretrained=\"data/pretrained_models/mlm/CANONICAL-TSMILES-${variant}-${vocab_size}-1M-20Epoch\""

        echo "=== Running command: $cmd ==="

        for ((i=1; i<=REPEATS; i++)); do
        echo "Run $i / $REPEATS"
        
        eval $cmd
        
        if [ "$i" -lt "$REPEATS" ]; then
            echo "Sleeping for ${SLEEP_TIME}s..."
            sleep $SLEEP_TIME
        fi
        done

        echo "Finished command"
        echo "-----------------------------"
    done
  done
done