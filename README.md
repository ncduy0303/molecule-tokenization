# Advanced Tokenization Methods for Molecular Foundation Models

NUS School of Computing Final Year Project by Nguyen Cao Duy, supervised by Dr. Bian Yatao. Project ID: H437020, academic year 2025-2026.

This project studies how molecular tokenization affects masked language modeling and molecular property prediction. It compares SMILES-based tokenizers, fragSMILES and t-SMILES, including shared exocyclic fragmentation, learned principal-subgraph merges, and an enhanced TSDY representation that separates fragment cores from attachment information.

The pipeline is **tokenizer construction -> RoBERTa MLM pretraining -> property classification on HIV or BBBP**. Datasets, trained models and generated results are not included in Git.

## Setup

Run commands from the repository root:

```bash
uv sync --locked
source .venv/bin/activate
python -m main --cfg job --resolve
```

The last command displays the default configuration without starting training. Installation requires Git and network access. Dataset loaders download their source data when local caches are absent. Use a GPU for model training.

## Run a baseline

These BPE examples connect each stage using explicit output paths. Use fresh output and cache directories for a new experiment. The defaults use a 5,000-token vocabulary, 2,000,000 tokenizer-training molecules and 20 MLM epochs.

### 1. Train the tokenizer

```bash
python -m main +name=bpe_tokenizer \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train \
  algorithm=train_bpe \
  dataset.data_dir=data/bpe_baseline/tokenizer \
  hydra.run.dir=outputs/bpe_baseline/tokenizer
```

### 2. Pretrain the MLM

```bash
python -m main +name=bpe_mlm \
  experiment=tokenizer_benchmark dataset=pubchem10m_mlm_train \
  algorithm=bpe_roberta \
  dataset.data_dir=data/bpe_baseline/mlm \
  dataset.tokenizer_indices_path=data/bpe_baseline/tokenizer/subset_indices.json \
  algorithm.tokenizer.pretrained=outputs/bpe_baseline/tokenizer/tokenizer \
  hydra.run.dir=outputs/bpe_baseline/mlm
```

### 3. Fine-tune and evaluate

```bash
python -m main +name=bpe_bbbp \
  experiment=downstream_classification dataset=molnet_bbbp \
  algorithm=bpe_classifier \
  dataset.data_dir=data/bpe_baseline/bbbp \
  algorithm.tokenizer.pretrained=outputs/bpe_baseline/tokenizer/tokenizer \
  algorithm.model.pretrained=outputs/bpe_baseline/mlm/final_checkpoint \
  hydra.run.dir=outputs/bpe_baseline/bbbp
```

For HIV, change `dataset=molnet_hiv` and use separate HIV cache/output paths and a run name. Classification selects a checkpoint using validation performance, evaluates test ROC-AUC, and exports `best_model/`.

## Reproducibility and logging

- Pin the same `dataset.hf_revision` in both PubChem stages for reproducible source selection. MLM excludes the tokenizer corpus's source-row indices; this does not deduplicate chemically identical molecules in different rows.
- Keep representation settings and tokenizer artifacts matched across all stages. Preset artifact paths refer to local research runs; override them with your own paths.
- Cache provenance is validated. If source or conversion settings change, use a fresh `dataset.data_dir` instead of overwriting old artifacts.
- W&B is disabled by default. To enable it, run `wandb login` and add `wandb.mode=online wandb.entity=YOUR_ENTITY wandb.project=YOUR_PROJECT` to model-training commands.
- Resume training with `resume=/absolute/path/to/checkpoint-N`.

## Where to go next

- [Configuration reference](configurations/README.md): choose tokenizer families, representations and training settings.
- [Cluster scripts](scripts/README.md): submit jobs on Slurm after adapting resource requests to your cluster.
- [Research reports](reports/README.md): project motivation and Continual Assessment report/slides.

For code changes, start at [main.py](main.py). Pipeline stages live in `experiments/`, tokenizer/model construction in `algorithms/`, dataset preparation in `datamodules/`, and molecular encoders and helpers in `utils/`. Each directory has a short README for local details.

## Template attribution

This repo is using [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its license, you must keep the above sentence and this setence in `README.md` and the `LICENSE` file to credit the author.

See [LICENSE](LICENSE). The molecular experiments retain the template's Hydra entry point and base task dispatch, but use Hugging Face Trainer rather than the removed example Lightning training stack.
