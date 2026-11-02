# Pipeline stages

`main.py` selects an experiment, builds its algorithm, and runs the methods in `experiment.tasks`.

| Experiment                  | Purpose                                              | Main training output |
| --------------------------- | ---------------------------------------------------- | -------------------- |
| `tokenizer_training`        | Construct a vocabulary/tokenizer                     | `tokenizer/`         |
| `tokenizer_benchmark`       | MLM pretraining and evaluation                       | `final_checkpoint/`  |
| `downstream_classification` | Property prediction, selecting by validation ROC-AUC | `best_model/`        |

Shared PS mining is a tokenizer-training task that produces `ps_merges.json` instead of a tokenizer. Model training uses Hugging Face Trainer; examples are in the [main README](../README.md).

## Loading and evaluation

- `algorithm.model.pretrained` initializes model weights.
- `algorithm.model.finetuned` loads a trained classifier, including its head.
- `'experiment.tasks=[evaluation]'` evaluates saved weights. Keep the tokenizer and dataset settings matched to the model.

## Resuming training

Use `resume=/absolute/path/to/checkpoint-N` with the original tokenizer, dataset, architecture and training settings. A full checkpoint restores optimizer, scheduler, RNG and Trainer state as well as weights.

- Reapply the original overrides; `.hydra/config.yaml` is not automatically loaded.
- `num_train_epochs` is the total target, not additional epochs.
- Enable checkpoint saving in the original run, for example `experiment.training.save_strategy=epoch`. MLM defaults to no intermediate checkpoints.
- `final_checkpoint/` and `best_model/` are weight exports, not full resume checkpoints.
- Resume supports model training with standard local Trainer checkpoints, not tokenizer construction, evaluation or W&B run IDs.
