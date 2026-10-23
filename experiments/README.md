# Molecular experiments

`main.py` selects an experiment through `exp_registry` in `__init__.py`. Each experiment inherits `BaseExperiment`, which stores the configuration/output directory, builds an algorithm from its registry, and invokes methods listed in `experiment.tasks`.

| Experiment                  | Tasks                    | Inputs                                        | Outputs within the run directory   |
| --------------------------- | ------------------------ | --------------------------------------------- | ---------------------------------- |
| `tokenizer_training`        | `training`               | PubChem tokenizer corpus and `train_*` config | `tokenizer/`                       |
| `tokenizer_benchmark`       | `training`, `evaluation` | PubChem MLM splits and `*_roberta` config     | `final_checkpoint/` after training |
| `downstream_classification` | `training`, `evaluation` | MoleculeNet data and `*_classifier` config    | `best_model/` after training       |

## Execution flow

- `TokenizerTrainingExperiment` dispatches tokenizer construction by `algorithm.tokenizer.type`. It supports BPE, APE, SPE, SMIRK, SMIRK-GPE, PCATT, SMIRK-PCATT, fragSMILES, t-SMILES, and PS-fragSMILES.
- `TokenizerBenchmarkExperiment` constructs the model/tokenizer, loads tokenized PubChem splits, and uses dynamic MLM masking through `DataCollatorForLanguageModeling`. Training includes validation and final test evaluation.
- `DownstreamClassificationExperiment` loads MoleculeNet data, infers the target count, constructs a classification head, and uses ROC-AUC for checkpoint selection and early stopping. It evaluates validation and test splits after training.
- The two model experiments configure W&B through environment variables and Hugging Face Trainer. Tokenizer construction prints progress and saves files locally.

## Loading and evaluation

- `algorithm.model.pretrained` initializes a model from saved weights.
- `algorithm.model.finetuned` loads an already fine-tuned classifier, including its classification head.

Use the complete commands in the [root README](../README.md). Select compatible groups using the [configuration guide](../configurations/README.md).

## Resume training

Both model experiments pass `resume` to Hugging Face Trainer to restore model weights, optimizer, scheduler, RNG, and Trainer state. Use a local `checkpoint-N` directory containing the full state, with the same tokenizer, data, model architecture, and training settings as the original run.

```bash
python -m main +name=bpe_mlm_resumed \
  experiment=tokenizer_benchmark dataset=pubchem10m_mlm_train algorithm=bpe_roberta \
  algorithm.tokenizer.pretrained=PATH_TO_ORIGINAL_TOKENIZER \
  resume=/absolute/path/to/checkpoint-N
```

- Reapply the original dataset/cache overrides as well. The saved `.hydra/config.yaml` is a record, not automatically merged into the new command.
- `experiment.training.num_train_epochs` is the total target across the original and resumed training, not additional epochs.
- MLM defaults to `save_strategy=no`. Set `experiment.training.save_strategy=epoch` in the original run to produce resumable checkpoints. Classification already saves epoch checkpoints.
- `final_checkpoint/` and `best_model/` exports support loading weights but lack optimizer and scheduler state. Use the sibling `checkpoint-N` directories for resume.
- Only the `training` task supports resume. Tokenizer construction and evaluation reject it. The old root `load` option and W&B run IDs produce explicit errors; use model configuration for weight-only loading.
- This path supports standard local Trainer checkpoints. DeepSpeed/FSDP-specific distributed checkpoint formats are not supported by the validator.
