# Molecular model construction

The active algorithms are small wrappers around Hugging Face tokenizers and RoBERTa models. Hugging Face Trainer runs the optimization loop in `experiments/`.

| Module                        | Responsibility                                                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `tokenizer_benchmark/algo.py` | `MoleculeMLMAlgo` loads a tokenizer and builds `RobertaForMaskedLM`, either from configuration or pretrained weights.                    |
| `downstream_finetune/algo.py` | `MoleculeClassificationAlgo` loads a tokenizer, then builds `RobertaForSequenceClassification` once the dataset's target count is known. |

Tokenizer construction itself lives in `experiments/tokenizer_training.py`. Its `TokenizerTrainer` holds configuration and does not construct a neural model.

Both model wrappers dispatch on `algorithm.tokenizer.type`: `auto`, `smirk`, `ape`, `pcatt`, `spe`, `smirk_pcatt`, `fragsmiles`, `ps_fragsmiles`, or `tsmiles`. `auto` uses Hugging Face AutoTokenizer, including locally trained BPE tokenizers. Custom tokenizer implementations live in [utils](../utils/README.md).

## Adding a method

- Add its tokenizer loading branch to both wrappers if it needs pretraining and downstream evaluation.
- Add the relevant YAML files under `configurations/algorithm/`.
- Register the YAML names in the appropriate experiment's `compatible_algorithms` mapping.
- For a new tokenizer construction method, also add its training dispatch and implementation to `TokenizerTrainingExperiment`.
- Check special-token IDs, vocabulary size, saved-tokenizer reload behavior, and consistency between training and evaluation tokenization.
