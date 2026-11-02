# Local data and artifacts

This directory is ignored by Git except for this README. Loaders create caches as needed; pretrained artifacts must be supplied locally.

| Location                                       | Contents                                           |
| ---------------------------------------------- | -------------------------------------------------- |
| `pubchem10m_tokenizer_train/`                  | Tokenizer corpus, selected indices and word counts |
| `pubchem10m_mlm_train/`                        | MLM split indices and processed strings            |
| `molnet/`                                      | MoleculeNet downloads and caches                   |
| `pretrained_tokenizers/`, `pretrained_models/` | Saved artifacts referenced by research presets     |
| `ape_tokenizer/`                               | Vocabulary files referenced by APE presets         |

Training saves new artifacts under `outputs/`; it does not copy them here automatically. Pass output paths directly to later stages.

## Cache rules

- Keep artifacts with their `.meta.json` provenance files. Settings and checksums are validated on reuse.
- For changed settings or incompatible old caches, choose a fresh `dataset.data_dir`. Do not fabricate metadata or overwrite old research caches.
- Train the tokenizer first, then set MLM's `dataset.tokenizer_indices_path` to its `subset_indices.json`. Use the same source and `dataset.hf_revision` in both stages.
- Exclusion guarantees disjoint source rows, not chemical deduplication. Cache validation checks provenance, not reconstruction correctness.

See the [main README](../README.md) for commands connecting the stages.
