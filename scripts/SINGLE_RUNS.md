# Running one experiment

Run these commands from the repository root after installing the locked environment. The examples use Bash. Set each path to your own files. W&B is off by default. For the existing SMILES BPE tokenizer, MLM and HIV/BBBP commands, see the [baseline walkthrough](../README.md#run-a-baseline).

## Enhanced TSDY from local files

Fit the vocabulary from a text file containing one SMILES per line. Choose a new output directory. The fit checks reconstruction on every row before exporting the tokenizer. The vocabulary size is an upper bound, including the reserved grammar and model specials.

```bash
RUN=outputs/my-enhanced-run
python -m scripts.prepare_external_tsdy \
  --input inputs/tokenizer-training.smi --all-rows --vocab-size 5000 \
  --output "$RUN/fit" --tokenizer-output "$RUN/tokenizer" --workers 4
```

Supply your own train, validation and test CSVs. The MLM needs a SMILES column. The classifier also needs a binary label column. Prepare the split membership before running these commands; this interface consumes it as supplied. Keep the tokenizer-fitting corpus separate from the MLM splits. The default length policy rejects overlength rows. Use `--length-policy truncate` for the report's MLM input policy.

```bash
python -m scripts.prepare_tokenizer_inputs \
  --tokenizer "$RUN/tokenizer" --output "$RUN/mlm-data" --workers 4 \
  --train inputs/mlm/train.csv --validation inputs/mlm/validation.csv --test inputs/mlm/test.csv \
  --smiles-column smiles --length-policy truncate

python -m main +name=my-enhanced-mlm experiment=tokenizer_benchmark \
  dataset=prepared_mlm dataset.prepared_path="$RUN/mlm-data" \
  algorithm=external_tsdy_roberta algorithm.tokenizer.pretrained="$RUN/tokenizer" \
  experiment.training.seed=42 hydra.run.dir="$RUN/mlm"

python -m scripts.prepare_tokenizer_inputs \
  --tokenizer "$RUN/tokenizer" --output "$RUN/classifier-data" --workers 4 \
  --train inputs/task/train.csv --validation inputs/task/validation.csv --test inputs/task/test.csv \
  --smiles-column smiles --label-column active

python -m main +name=my-enhanced-classifier experiment=downstream_classification \
  dataset=prepared_classifier dataset.prepared_path="$RUN/classifier-data" \
  algorithm=external_tsdy_classifier algorithm.tokenizer.pretrained="$RUN/tokenizer" \
  algorithm.model.pretrained="$RUN/mlm/final_checkpoint" \
  experiment.training.seed=42 hydra.run.dir="$RUN/classifier"
```

The preparation directories contain the split hashes, row counts, failures and length audit. Reuse a completed export across seeds. Use a new classifier output directory for each run. Validation and test splits need both classes for a meaningful ROC-AUC.

## Shared PS mining and fragment vocabulary

The miner takes raw canonical SMILES. Its output is an ordered rule file. Here is one shared PS-fragSMILES tokenizer run using the PubChem loader. Set `REV` to your chosen Hugging Face dataset commit; use the same source, seed and corpus size for fitting the rules and the tokenizer.

```bash
RUN=outputs/my-ps-run
CACHE=data/my-ps-run
: "${REV:?Set REV to the PubChem dataset revision}"
sampling=(dataset=pubchem10m_tokenizer_train "dataset.hf_revision=$REV" dataset.seed=42 dataset.corpus_size=20000)

python -m main +name=my-ps-rules experiment=tokenizer_training algorithm=train_shared_ps \
  "${sampling[@]}" dataset.data_dir="$CACHE/miner" \
  algorithm.tokenizer.num_merges=100 algorithm.tokenizer.min_frequency=10 hydra.run.dir="$RUN/miner"

representation=(dataset.fragmentation=exocyclic dataset.use_fragsmiles=true "dataset.ps_merges_path=$RUN/miner/ps_merges.json")
python -m main +name=my-ps-fragments experiment=tokenizer_training algorithm=train_fragsmiles \
  "${sampling[@]}" "${representation[@]}" dataset.data_dir="$CACHE/fragments" \
  algorithm.tokenizer.vocab_size=5000 hydra.run.dir="$RUN/fragments"

python -m main +name=my-ps-mlm experiment=tokenizer_benchmark algorithm=fragsmiles_roberta \
  dataset=pubchem10m_mlm_train "dataset.hf_revision=$REV" dataset.seed=42 \
  dataset.data_dir="$CACHE/mlm" "${representation[@]}" \
  dataset.tokenizer_indices_path="$CACHE/fragments/subset_indices.json" \
  algorithm.tokenizer.pretrained="$RUN/fragments/tokenizer" hydra.run.dir="$RUN/mlm"

python -m main +name=my-ps-hiv experiment=downstream_classification algorithm=fragsmiles_classifier \
  dataset=molnet_hiv dataset.data_dir="$CACHE/hiv" "${representation[@]}" \
  algorithm.tokenizer.pretrained="$RUN/fragments/tokenizer" algorithm.model.pretrained="$RUN/mlm/final_checkpoint" \
  experiment.training.seed=42 hydra.run.dir="$RUN/hiv"
```

For base fragmentation, omit the merge-file override across all stages. For shared t-SMILES, replace the fragSMILES flag with `dataset.use_tsmiles=true dataset.tsmiles_variant=TSID` (or TSDY or TSIS) and use the corresponding `train_tsmiles`, `tsmiles_roberta` and `tsmiles_classifier` presets.

## Intrinsic evaluation

Score one saved tokenizer on a SMILES text file without training a model:

```bash
python -m scripts.evaluate_tokenizer --tokenizer outputs/my-enhanced-run/tokenizer \
  --representation enhanced --input inputs/evaluation.smi --output outputs/my-enhanced-eval

python -m scripts.evaluate_tokenizer --tokenizer outputs/my-ps-run/fragments/tokenizer \
  --representation shared --variant fragsmiles --ps-merges outputs/my-ps-run/miner/ps_merges.json \
  --input inputs/evaluation.smi --output outputs/my-ps-eval
```

Use a fresh evaluation output directory. Inspect `summary.json` for conversion coverage, lengths, unknown rates and reconstruction checks, and `records.jsonl` for individual successes and failures. For SMILES/SAFE baselines select the representation used when fitting; some historical artifacts do not record this choice. Full results retain per-row token-use information in memory, so start with a fixed sample for large datasets.

For one Slurm job, put any one training command above in a batch file after your site's resource directives, activate the environment and run from the repository root. Set optional tracking overrides on that command: `wandb.mode=online wandb.entity=YOUR_ENTITY wandb.project=YOUR_PROJECT`.

The supplied [single-run batch file](slurm/single_run.sbatch) forwards all Hydra arguments unchanged. For example, with the paths above and site-appropriate resources:

```bash
sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 --cpus-per-task=8 --mem=32G \
  scripts/slurm/single_run.sbatch +name=my-classifier experiment=downstream_classification \
  dataset=prepared_classifier dataset.prepared_path="$RUN/classifier-data" \
  algorithm=external_tsdy_classifier algorithm.tokenizer.pretrained="$RUN/tokenizer" \
  algorithm.model.pretrained="$RUN/mlm/final_checkpoint" hydra.run.dir="$RUN/slurm-classifier"
```

Here `RUN` refers to the enhanced-TSDY example. Slurm writes the job log to its default location unless you supply `--output` or `--error`.
