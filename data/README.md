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

## Cache validation and migration

- Generated indices, corpus strings, and word counts have adjacent `.meta.json` provenance files containing settings and a SHA-256 checksum. Keep each artifact and its sidecar together.
- Index identities include the source, requested revision, seed, and sizes. MoleculeNet also records a digest of filtered source rows and target columns. Representation caches include their selection and active conversion settings, including the SAFE slicer. Word counts track the verified corpus and pre-tokenization settings.
- A mismatch, missing legacy metadata, or changed artifact raises an actionable error. The loader does not silently overwrite old research data or trim a differently sized corpus.
- For existing caches, keep the old directory and choose a fresh `dataset.data_dir`. Do not fabricate metadata for old caches: the settings and source identity cannot be recovered reliably from filenames alone.
- Prepare the tokenizer corpus first. Set MLM's `dataset.tokenizer_indices_path` to that corpus directory's `subset_indices.json`. All MLM splits exclude those indices, including after cache reload.
- Both PubChem stages must use the same dataset source and requested `hf_revision`. Source fingerprints are also compared when data is loaded. Pin a revision for reproducible source row identities. Disjoint source rows do not guarantee chemical deduplication across duplicate source records.

Example migration for new BPE runs:

```bash
python -m main +name=bpe_tokenizer_v2 \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train algorithm=train_bpe \
  dataset.data_dir=data/pubchem_tokenizer_v2

python -m main +name=bpe_mlm_v2 \
  experiment=tokenizer_benchmark dataset=pubchem10m_mlm_train algorithm=bpe_roberta \
  dataset.data_dir=data/pubchem_mlm_v2 \
  dataset.tokenizer_indices_path=data/pubchem_tokenizer_v2/subset_indices.json \
  algorithm.tokenizer.pretrained=outputs/TOKENIZER_DATE/TOKENIZER_TIME/tokenizer
```

Conversion failures can still fall back to SMILES. Cache validation establishes provenance and integrity, not conversion success or scientific validity. Historical benchmark results are not retroactively validated by these checks.
