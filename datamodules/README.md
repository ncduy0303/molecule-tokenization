# Molecular datasets

The loaders in `molecule_datasets/` prepare strings and labels for the three project stages.

| Module                          | Input and behavior                                                                                           | Return value                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| `pubchem10m_tokenizer_train.py` | Samples valid PubChem molecules, caches indices and strings, optionally constructs PCATT word counts         | Corpus list and text-file path                |
| `pubchem10m_mlm_train.py`       | Samples PubChem train/validation/test indices, caches strings, applies the selected tokenizer                | Hugging Face `DatasetDict`                    |
| `molnet.py`                     | Downloads MoleculeNet CSVs, drops rows with missing selected targets, splits data, caches strings, tokenizes | Hugging Face `DatasetDict` with label vectors |

`molecule_datasets/__init__.py` registers `pubchem10m_mlm_train` for the MLM experiment. Tokenizer construction and downstream classification call their loaders directly.

## Representations

- Default input is RDKit-canonicalized SMILES.
- `dataset.use_safe=true` selects SAFE, with `dataset.safe_slicer` controlling fragmentation.
- `dataset.use_fragsmiles=true` selects precomputed fragSMILES strings.
- `dataset.use_tsmiles=true` selects t-SMILES, with `dataset.tsmiles_variant` and `dataset.tsmiles_slicer` controlling encoding.
- Only one of these representation flags may be enabled.
- PS-fragSMILES takes SMILES and performs its own learned decomposition inside its tokenizer. Leave all representation flags false for this method.

## MoleculeNet coverage

- BBBP uses target `p_np`; HIV uses `HIV_active`. Both use DeepChem scaffold splitting with train/validation/test fractions 0.8/0.1/0.1.
- The loader also has ClinTox (`CT_TOX`) and Tox21 (`SR-p53`) branches using random splitting. They do not have dedicated YAML presets and are single-target selections, not full multitask benchmarks.
- Invalid strings can survive canonicalization through fallback behavior. Do not assume every downstream row has been validated or converted successfully.

## Reproducibility

MLM requires `dataset.tokenizer_indices_path` from a verified tokenizer corpus and excludes those source rows from all splits. It validates the exclusion on cache reuse as well. Use the same source and `hf_revision` in both stages; pin the revision when reproducing experiments.

Generated caches require matching provenance and content checksums. Changed settings or legacy caches raise errors instead of silently reusing data. Keep old research caches intact and choose a fresh `dataset.data_dir` for new settings. See [data storage and migration](../data/README.md).
