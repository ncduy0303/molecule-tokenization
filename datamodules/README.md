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

The tokenizer and MLM samplers each use their own seeded permutation of the same PubChem source. Separate cache directories do not guarantee disjoint samples. Cross-stage exclusion is not implemented.

Cache reuse does not check all sampling and representation settings. Preserve existing caches for old runs and use a fresh `dataset.data_dir` when changing corpus size, seed, source, or SAFE slicer. See [data storage](../data/README.md).
