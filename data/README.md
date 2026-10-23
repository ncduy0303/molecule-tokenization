# Local data and model artifacts

This directory holds generated or downloaded data. Its contents are ignored by Git except this README. Loaders create their cache directories as needed. Existing local files are preserved by the repository cleanup.

| Location                      | Contents                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| `pubchem10m_tokenizer_train/` | Tokenizer corpus indices, corpus text, representation-specific corpora, PCATT word counts |
| `pubchem10m_mlm_train/`       | MLM split indices and processed strings for each split                                    |
| `molnet/`                     | MoleculeNet CSV downloads, split indices, and processed strings                           |
| `pretrained_tokenizers/`      | Manually organized tokenizer artifacts referenced by existing presets                     |
| `pretrained_models/`          | Manually organized model artifacts referenced by existing presets                         |
| `ape_tokenizer/`              | APE vocabulary location referenced by APE presets                                         |

Training initially writes artifacts into the Hydra run directory under `outputs/`. It does not automatically copy them into `data/pretrained_*`. Pass those output paths directly to later stages or organize copies yourself.

## Cache caveats

- Tokenizer corpus text can be reused without checking the requested seed or dataset source. A smaller requested corpus is trimmed from the cached list; a larger request than the cached corpus raises an error.
- MLM index reuse checks RDKit validation and seed, but does not check requested split sizes. Processed-string caches are reused based on file existence.
- MoleculeNet split-index reuse does not validate the requested seed or current source rows.
- SAFE cache filenames do not include the slicer. PCATT word-count caches do not include every sampling setting.
- Use separate `dataset.data_dir` directories for changed sampling settings or SAFE slicers. Keep previous cache directories alongside the configs for runs you want to reproduce.
- Tokenizer and MLM indices are sampled independently without exclusion. Inspect both index files to establish whether the actual corpora overlap.

Conversion failures can fall back to SMILES. The loaders do not produce a conversion-success manifest, so the cache name alone is not evidence that every molecule was converted.
