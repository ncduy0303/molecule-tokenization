# Model construction

These wrappers load tokenizers and construct RoBERTa models. Training is handled by [experiments](../experiments/README.md).

| Module                        | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `tokenizer_benchmark/algo.py` | Tokenizer loading and masked language model construction |
| `downstream_finetune/algo.py` | Tokenizer loading and classification model construction  |

Tokenizer implementations live in [utils](../utils/README.md); tokenizer training lives in `experiments/tokenizer_training.py`.

To add a method, add its loading logic to the relevant wrappers, create algorithm YAML presets, and register them in the experiment. If it needs vocabulary training, add that training dispatch too. Check saved-tokenizer reloads, vocabulary size and special-token IDs across both model stages.
