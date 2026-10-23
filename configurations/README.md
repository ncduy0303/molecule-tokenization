# Experiment configuration

Hydra composes `config.yaml` with the selected experiment, dataset, algorithm, and optional cluster files. Names on the command line are YAML basenames without `.yaml`.

The default is `experiment=tokenizer_training dataset=pubchem10m_tokenizer_train algorithm=train_bpe`. Logging defaults to `wandb.mode=disabled`.

## Compatible groups

| Stage                  | Experiment                  | Dataset                      | Algorithm choices                                                                                                                                                                                 |
| ---------------------- | --------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tokenizer construction | `tokenizer_training`        | `pubchem10m_tokenizer_train` | `train_bpe`, `train_ape`, `train_spe`, `train_smirk`, `train_smirk_gpe`, `train_pcatt`, `train_smirk_pcatt`, `train_fragsmiles`, `train_tsmiles`, `train_ps_fragsmiles`                           |
| MLM                    | `tokenizer_benchmark`       | `pubchem10m_mlm_train`       | `bpe_roberta`, `ape_roberta`, `spe_roberta`, `smirk_roberta`, `pcatt_roberta`, `smirk_pcatt_roberta`, `fragsmiles_roberta`, `tsmiles_roberta`, `ps_fragsmiles_roberta`                            |
| Classification         | `downstream_classification` | `molnet_bbbp`, `molnet_hiv`  | `bpe_classifier`, `ape_classifier`, `spe_classifier`, `smirk_classifier`, `pcatt_classifier`, `smirk_pcatt_classifier`, `fragsmiles_classifier`, `tsmiles_classifier`, `ps_fragsmiles_classifier` |

MLM also registers `ape_roberta_pretrained` and `bpe_roberta_pretrained`. Classification also registers `ape_classifier_pretrained_hiv` and `bpe_classifier_pretrained_hiv` for published fine-tuned HIV models. Use `experiment.tasks=[evaluation]` when evaluating those models without further training.

The `base_*` YAML files provide inherited settings. They are not standalone experiments or runnable algorithms.

`algorithm=train_shared_ps` with `experiment=tokenizer_training` learns a reusable
exocyclic-block PS merge model from raw SMILES. Its output is `ps_merges.json`,
not a Hugging Face tokenizer. Use the ordinary `train_fragsmiles` or
`train_tsmiles` presets afterward with `dataset.fragmentation=exocyclic` and
optional `dataset.ps_merges_path`. Keep those representation settings for MLM
and classification. See the [shared study runbook](../docs/SHARED_FRAGMENTATION.md).

## Common overrides

```bash
# Inspect only, without dataset downloads or training.
python -m main experiment=tokenizer_benchmark \
  dataset=pubchem10m_mlm_train algorithm=bpe_roberta --cfg job --resolve

# Structural pre-splitting for BPE vocabulary learning.
python -m main +name=sbpe experiment=tokenizer_training \
  dataset=pubchem10m_tokenizer_train algorithm=train_bpe \
  dataset.pretokenizer=structure_split

# t-SMILES vocabulary with matched representation settings.
python -m main +name=tsmiles_tsdym experiment=tokenizer_training \
  dataset=pubchem10m_tokenizer_train algorithm=train_tsmiles \
  dataset.use_tsmiles=true dataset.tsmiles_variant=TSDY dataset.tsmiles_slicer=mmpa
```

- BPE/PCATT pre-splitting uses `dataset.pretokenizer=null`, `atom_split`, or `structure_split`.
- SMIRK-GPE uses `algorithm.tokenizer.split_structure` and `algorithm.tokenizer.merge_brackets` instead.
- Vocabulary targets use `algorithm.tokenizer.vocab_size`; model training parameters live under `experiment.training`.
- Existing fields use `key=value`. Add a missing field with `+key=value`, for example `+algorithm.model.pretrained=PATH` on a scratch MLM preset.
- Quote list overrides in shells: `'experiment.tasks=[evaluation]'`.
- Many presets contain paths from previous runs. Set `algorithm.tokenizer.pretrained` and `algorithm.model.pretrained` to your artifacts instead of assuming those files are distributed with the repository.
- Keep representation settings, tokenizer vocabulary, and model weights matched. Changed sampling or conversion settings require fresh cache directories; incompatible caches raise errors.

## Cluster configuration

The generic `cluster/base_slurm.yaml` and launcher are retained as an optional template-derived path. It requires a configured environment and suitable resources before use.

The project's existing job wrappers are documented in [scripts](../scripts/README.md). Do not combine those wrappers with a `cluster=...` override: the wrapper already submits a job, and the override would trigger another submission.

## Corpus exclusion and checkpoints

- `dataset.hf_revision` selects a PubChem source revision in both stages.
- `dataset.tokenizer_indices_path` selects the verified tokenizer corpus to exclude from MLM. Override it when tokenizer training used a custom cache directory.
- `resume=/path/to/checkpoint-N` restores full local Trainer state for model training. See [experiments](../experiments/README.md) for required files and saving checkpoints.
