# Dataset preparation

The loaders in `molecule_datasets/` sample corpora, convert molecular representations and prepare tokenized datasets.

| Loader                          | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| `pubchem10m_tokenizer_train.py` | Tokenizer corpus and vocabulary-training inputs |
| `pubchem10m_mlm_train.py`       | MLM train, validation and test splits           |
| `molnet.py`                     | Molecular property datasets and labels          |

## Representations

- Default: canonical SMILES.
- SAFE: `dataset.use_safe=true`, with `dataset.safe_slicer`.
- fragSMILES: `dataset.use_fragsmiles=true`.
- t-SMILES: `dataset.use_tsmiles=true`, with `dataset.tsmiles_variant` and `dataset.tsmiles_slicer`.

Enable only one representation flag. Legacy PS-fragSMILES converts SMILES inside its tokenizer, so leave these flags false for that method.

For a shared exocyclic partition, set `dataset.fragmentation=exocyclic` with fragSMILES or t-SMILES TSID/TSDY/TSIS. Optional `dataset.ps_merges_path` applies frozen learned merges. This path raises conversion errors; historical conversion paths can fall back to SMILES.

Enhanced TSDY uses separate `external_tsdy_*` dataset presets and prepared inputs.

## Splits and caches

- HIV (`HIV_active`) and BBBP (`p_np`) use scaffold splits with train/validation/test fractions of 0.8/0.1/0.1.
- MLM excludes the tokenizer corpus's source indices. Both PubChem stages must use the same source revision.
- Changed sampling or representation settings require fresh cache directories. See [cache rules](../data/README.md).
