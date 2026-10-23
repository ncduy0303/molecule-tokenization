# Advanced Tokenization Methods for Molecular Foundation Models

NUS School of Computing Final Year Project by Nguyen Cao Duy, supervised by Dr. Bian Yatao. Project ID: H437020, academic year 2025-2026.

This repository benchmarks how molecular tokenization affects representation learning and molecular property prediction. It also explores fragment representations, including principal-subgraph fragmentation combined with fragSMILES.

## Project context

- [Continual Assessment report](reports/FYP_CA_Report.pdf): motivation, literature review, methodology, results, and proposed next steps.
- [Continual Assessment slides](reports/FYP_CA_Slides.pdf): presentation overview.

The implemented pipeline is:

1. Construct a tokenizer from a PubChem10M corpus.
2. Pretrain a RoBERTa model with masked language modeling (MLM).
3. Fine-tune and evaluate molecular property classification on MoleculeNet, with BBBP and HIV configuration files.

Supported families include BPE, PCATT, SMIRK and its GPE/PCATT extensions, APE, SPE, fragSMILES, t-SMILES, and PS-fragSMILES. SAFE is a dataset representation option.

## Setup

Run commands from the repository root. Dependencies and Git sources are declared in [pyproject.toml](pyproject.toml), with resolved versions in [uv.lock](uv.lock).

```bash
uv sync --locked
source .venv/bin/activate
```

The project declares Python >=3.10. Some dependencies come from Git repositories, so installation requires Git and network access. Dataset loaders download PubChem or MoleculeNet data when their local caches are absent. Model training requires substantially more resources than inspecting configurations or constructing a small tokenizer.

## Inspect a run without training

```bash
python -m main --cfg job --resolve
```

The default configuration selects BPE tokenizer construction. Actual runs require `+name=...`. W&B logging is disabled by default. For model runs with online logging, run `wandb login` and pass `wandb.mode=online wandb.entity=YOUR_ENTITY`.

## Run the pipeline

These BPE examples explicitly connect each stage's artifacts. Replace the illustrative output paths with the directories printed by your runs. Do not use `outputs/latest-run` for an earlier stage's input, because each new run updates that link.

### Construct a tokenizer

```bash
python -m main +name=bpe_tokenizer \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train \
  algorithm=train_bpe
```

The tokenizer is saved under `outputs/<date>/<time>/tokenizer/`. The default corpus contains 2,000,000 molecules and the target vocabulary size is 5,000. A smaller run should use a separate cache directory, for example `dataset.corpus_size=1000 dataset.data_dir=data/smoke_tokenizer`. A first run can still download the full source dataset.

### Pretrain a molecular language model

```bash
python -m main +name=bpe_mlm \
  experiment=tokenizer_benchmark dataset=pubchem10m_mlm_train \
  algorithm=bpe_roberta \
  algorithm.tokenizer.pretrained=outputs/TOKENIZER_DATE/TOKENIZER_TIME/tokenizer
```

The final model and tokenizer are saved under `outputs/<date>/<time>/final_checkpoint/`. Defaults specify 1,000,000 training, 100,000 validation, and 100,000 test molecules, with 20 training epochs. The model has 6 layers, 12 attention heads, hidden size 768, intermediate size 1536, and 514 position embeddings. Dataset tokenization truncates at 512 tokens.

### Fine-tune on BBBP or HIV

```bash
python -m main +name=bpe_bbbp \
  experiment=downstream_classification dataset=molnet_bbbp \
  algorithm=bpe_classifier \
  algorithm.tokenizer.pretrained=outputs/TOKENIZER_DATE/TOKENIZER_TIME/tokenizer \
  algorithm.model.pretrained=outputs/MLM_DATE/MLM_TIME/final_checkpoint
```

Use `dataset=molnet_hiv` for HIV. The experiment evaluates ROC-AUC, selects the best validation checkpoint, and saves it under `best_model/`. The default training limit is 5 epochs with early stopping.

For evaluation of a saved classifier, use the same tokenizer and dataset settings, set `'experiment.tasks=[evaluation]'`, and add `+algorithm.model.finetuned=outputs/CLASSIFIER_DATE/CLASSIFIER_TIME/best_model`. For MLM evaluation, use `'experiment.tasks=[evaluation]'` with `+algorithm.model.pretrained=outputs/MLM_DATE/MLM_TIME/final_checkpoint` and the matching tokenizer.

## Configuration and reproducibility

- [Configuration guide](configurations/README.md) lists compatible experiment, dataset, and algorithm choices.
- Existing algorithm presets often point to local `data/pretrained_tokenizers/` and `data/pretrained_models/` artifacts. These artifacts are not included in Git. Override paths for your own runs.
- Use matching representation settings and tokenizer vocabulary throughout the pipeline. A SAFE-trained tokenizer should receive SAFE data, for example.
- MLM excludes the source-row indices in `dataset.tokenizer_indices_path`. Prepare the tokenizer corpus first and point this field at its `subset_indices.json` when using a non-default corpus directory. Both stages must use the same source and `dataset.hf_revision`. This guarantees disjoint source rows, not deduplication of chemically identical molecules in different source rows.
- Cached indices, processed strings, and word counts require matching provenance and checksums. Legacy or incompatible caches fail without being overwritten. Use fresh directories for new experiments; see [data migration notes](data/README.md).
- Use `resume=/absolute/path/to/checkpoint-N` to restore a full local Hugging Face Trainer checkpoint. `load` and W&B run IDs are rejected. Final `final_checkpoint/` and `best_model/` exports contain model artifacts, not complete resume state. See [resume instructions](experiments/README.md).

## Repository map

| Location                                    | Purpose                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------ |
| [main.py](main.py)                          | Hydra entry point, output directories, local or Slurm dispatch                 |
| [experiments/](experiments/README.md)       | Pipeline orchestration and Hugging Face training/evaluation                    |
| [algorithms/](algorithms/README.md)         | Tokenizer and RoBERTa model construction                                       |
| [datamodules/](datamodules/README.md)       | Corpus sampling, representation conversion, dataset splits                     |
| [utils/](utils/README.md)                   | Custom tokenizers, molecular encoders, principal-subgraph code, infrastructure |
| [configurations/](configurations/README.md) | Hydra settings for each pipeline stage                                         |
| [scripts/](scripts/README.md)               | Project Slurm wrappers and batch experiment helpers                            |
| [data/](data/README.md)                     | Local caches and saved artifacts, excluded from Git except documentation       |
| [reports/](reports/README.md)               | CA report and slides                                                           |

`archive/` is an ignored local research archive. It is not part of the maintained execution path. Existing local data and archived work are preserved.

## Template attribution

This repo is using [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its license, you must keep the above sentence and this setence in `README.md` and the `LICENSE` file to credit the author.

See [LICENSE](LICENSE). The molecular experiments retain the template's Hydra entry point and base task dispatch, but use Hugging Face Trainer rather than the removed example Lightning training stack.
