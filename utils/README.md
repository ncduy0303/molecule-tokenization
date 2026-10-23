# Tokenizers, molecular encoders, and infrastructure

## Molecular code

| Files                                                      | Purpose                                                                                   |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `spe_tokenizer.py`                                         | Hugging Face wrapper for SMILES Pair Encoding, with vocabulary and merge-file persistence |
| `smirk_pcatt_adapter.py`, `smirk_pcatt_tokenizer.py`       | Map SMIRK glyphs to bytes for PCATT and expose the result as a Hugging Face tokenizer     |
| `fragsmiles_tokenizer.py`                                  | Split precomputed fragSMILES strings and map fragments to vocabulary IDs                  |
| `tsmiles_tokenizer.py`                                     | Split precomputed t-SMILES into fragment and structural tokens                            |
| `ps_fragsmiles_tokenizer.py`                               | Convert SMILES with a learned principal-subgraph vocabulary, then tokenize fragSMILES     |
| `safe_utils.py`, `fragsmiles_utils.py`, `tsmiles_utils.py` | Representation conversion used by dataset loaders                                         |
| `ps_fragsmiles_utils.py`                                   | Batch helpers for learned principal-subgraph conversion during tokenizer construction     |
| [ps/](ps/README.md)                                        | Ring-aware principal-subgraph mining and conversion to fragSMILES                         |
| `timeout_utils.py`                                         | SIGALRM-based timeout context used by t-SMILES conversion                                 |

Representation converters retain a SMILES fallback when conversion fails. Cached corpora may therefore contain fallback strings. t-SMILES timeout protection is a no-op on systems without SIGALRM or outside the main thread.

## Retained infrastructure

- `print_utils.py`: colored progress messages.
- `distributed_utils.py`: rank-zero detection used by the launcher and base experiment. This still imports Lightning, even though molecular training uses Hugging Face Trainer.
- `cluster_utils.py`: optional generic Slurm job generation used by `main.py`.
- `ckpt_utils.py`: legacy W&B checkpoint download helpers used by the root `load`/`resume` options. These do not implement resumption of the molecular Trainers.
