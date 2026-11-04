# Tokenizers and molecular encoders

This directory contains representation conversion, tokenizer wrappers and shared training helpers.

| Files                                                      | Purpose                                              |
| ---------------------------------------------------------- | ---------------------------------------------------- |
| `*_tokenizer.py`                                           | Tokenizer wrappers and saved-vocabulary handling     |
| `safe_utils.py`, `fragsmiles_utils.py`, `tsmiles_utils.py` | Historical representation conversion                 |
| `shared_fragmentation.py`, `shared_ps.py`                  | Shared exocyclic partition and learned PS merges     |
| `external_tsdy.py`, `external_tsdy_tokenizer.py`           | Enhanced TSDY encoding, decoding and tokenization    |
| `cache_utils.py`                                           | Cache provenance, checksums and corpus exclusion     |
| `ckpt_utils.py`                                            | Full training-checkpoint validation                  |
| `smirk_utils.py`                                           | Shared SMIRK vocabulary loading                      |

The remaining helpers support logging, distributed execution and Slurm submission. Representation behavior differs by path: historical converters can fall back to SMILES, while shared fragmentation raises conversion errors. See [dataset settings](../datamodules/README.md).
