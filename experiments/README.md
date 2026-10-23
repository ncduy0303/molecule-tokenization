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
