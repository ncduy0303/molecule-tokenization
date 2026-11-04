# Advanced Tokenization Methods for Molecular Foundation Models

Nguyen Cao Duy · NUS School of Computing · B.Comp. Final Year Project H437020 · Academic Year 2026/2027 · Supervisor: Dr. Bian Yatao

This project compares ways to turn molecules into token sequences for property prediction. It tests how the choice of fragment boundaries affects prediction on HIV and BBBP, and whether different attachment patterns can share a vocabulary entry for the same chemical fragment. The [final report](reports/final/) describes the methods and experiments.

## Setup

Run the following from the repository root with Git and [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --locked
source .venv/bin/activate
python -m main --cfg job --resolve
```

The last command prints the selected configuration. The training commands below download data on first use and write caches and checkpoints locally. Use a GPU for model training. For an example that uses your own SMILES files and CSV splits, see [enhanced TSDY from local files](scripts/SINGLE_RUNS.md#enhanced-tsdy-from-local-files).

## Run a baseline

This example fits one SMILES BPE tokenizer, pretrains one RoBERTa masked language model (MLM), and finetunes it for blood-brain barrier penetration (BBBP). The example uses reduced PubChem sample sizes and one MLM epoch. The [experimental methodology](reports/final/sections/40-methodology.typ) gives the settings used for the reported scores.

Choose fresh cache and output directories. Run the commands in the same shell:

```bash
CACHE=data/my-bpe
RUN=outputs/my-bpe
REV=802e0824f41f8bade43fa95fd7d02fe80703a1d5
SEED=42

# Fit the tokenizer and record which PubChem rows were used.
python -m main +name=bpe-fit experiment=tokenizer_training \
  dataset=pubchem10m_tokenizer_train algorithm=train_bpe \
  dataset.hf_revision="$REV" dataset.seed="$SEED" dataset.corpus_size=10000 \
  algorithm.tokenizer.vocab_size=5000 dataset.data_dir="$CACHE/fit" hydra.run.dir="$RUN/fit"

# Pretrain on separate rows from the same dataset revision.
python -m main +name=bpe-mlm experiment=tokenizer_benchmark \
  dataset=pubchem10m_mlm_train algorithm=bpe_roberta \
  dataset.hf_revision="$REV" dataset.seed="$SEED" dataset.data_dir="$CACHE/mlm" \
  dataset.tokenizer_indices_path="$CACHE/fit/subset_indices.json" \
  dataset.train_size=20000 dataset.val_size=2000 dataset.test_size=2000 \
  algorithm.tokenizer.pretrained="$RUN/fit/tokenizer" experiment.training.seed="$SEED" \
  experiment.training.num_train_epochs=1 hydra.run.dir="$RUN/mlm"

# Finetune from the MLM weights using the same tokenizer.
python -m main +name=bpe-bbbp experiment=downstream_classification \
  dataset=molnet_bbbp algorithm=bpe_classifier dataset.data_dir="$CACHE/bbbp" \
  algorithm.tokenizer.pretrained="$RUN/fit/tokenizer" \
  algorithm.model.pretrained="$RUN/mlm/final_checkpoint" \
  experiment.training.seed="$SEED" hydra.run.dir="$RUN/bbbp"
```

Read `$RUN/bbbp/test_results.json` for test ROC-AUC. The classifier selected by validation ROC-AUC is in `$RUN/bbbp/best_model/`. To evaluate HIV, change the final command to `dataset=molnet_hiv` and give it a separate name, cache, and output directory.

## Choose a method

| Report comparison                            | What to run                                                                                                                                                                                                                                                                                |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Initial benchmark (§5.1, Appendix D)         | The baseline above uses BPE on SMILES. The [algorithm presets](configurations/algorithm/) also cover SMIRK, PCATT, SPE, APE, fragSMILES, and t-SMILES. [Dataset settings](datamodules/README.md) control conversion to SAFE or fragment strings.                                           |
| Shared fragmentation and PS merging (§5.2)   | Use an exocyclic partition with fragSMILES, TSID, TSDY, or TSIS. For the PS variant, fit one rule file on the tokenizer corpus and pass it to each later stage. [Mining, tokenizer fitting, MLM, and finetuning commands](scripts/SINGLE_RUNS.md#shared-ps-mining-and-fragment-vocabulary) |
| Enhanced TSDY (§5.3)                         | Fit a vocabulary of clean fragment cores and attachment tokens, then prepare MLM or binary classification splits. [Local-file workflow](scripts/SINGLE_RUNS.md#enhanced-tsdy-from-local-files)                                                                                             |
| Intrinsic evaluation (§§5.4–5.5, Appendix C) | Supply a saved tokenizer, its representation settings, and a SMILES file to measure token length, unknown-token frequency, and reconstruction. [Evaluation command and output files](scripts/SINGLE_RUNS.md#intrinsic-evaluation)                                                          |

## Configure a run

- **Inputs and outputs.** Supply your own tokenizer and model paths when selecting a preset. Keep the representation and merge rules consistent across stages. Use a fresh cache when changing the data revision or encoding settings.
- **Seeds.** `dataset.seed` controls data selection. `experiment.training.seed` controls training. To repeat finetuning, reuse the MLM, tokenizer, and split, change the training seed, and choose a new output directory.
- **Execution.** Each command runs one experiment. To submit it to Slurm, pass the same overrides to the [single-job batch file](scripts/slurm/single_run.sbatch). Optional W&B logging uses `wandb.mode=online wandb.entity=YOUR_ENTITY wandb.project=YOUR_PROJECT` after `wandb login`. Local metrics are written with the default disabled logging mode.
- **Checkpoints.** To resume an interrupted training run from a full Trainer checkpoint, see [checkpoint settings](experiments/README.md#resuming-training).

## Report and attribution

See the [final report directory](reports/final/) for the Typst source and [reports/CA](reports/CA/) for the earlier midterm report.

## Template attribution

This repo is using [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its license, you must keep the above sentence and this setence in `README.md` and the `LICENSE` file to credit the author.

See [LICENSE](LICENSE). The molecular experiments retain the template's Hydra entry point and base task dispatch, but use Hugging Face Trainer rather than the removed example Lightning training stack.
