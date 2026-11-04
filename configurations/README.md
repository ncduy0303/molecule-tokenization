# Experiment configuration

Hydra combines `config.yaml` with an experiment, dataset and algorithm preset. Use YAML filenames without `.yaml` on the command line. Start with the complete pipeline in the [main README](../README.md).

| Stage              | Experiment                  | Typical dataset               | Example algorithm |
| ------------------ | --------------------------- | ----------------------------- | ----------------- |
| Tokenizer training | `tokenizer_training`        | `pubchem10m_tokenizer_train`  | `train_bpe`       |
| MLM                | `tokenizer_benchmark`       | `pubchem10m_mlm_train`        | `bpe_roberta`     |
| Classification     | `downstream_classification` | `molnet_hiv` or `molnet_bbbp` | `bpe_classifier`  |

Other method presets are in `algorithm/`. `base_*` files provide inherited settings, not standalone runs. Supply the tokenizer directory (or APE vocabulary file) with `algorithm.tokenizer.pretrained`. For finetuning, supply the MLM checkpoint with `algorithm.model.pretrained`.

## Common settings

- `algorithm.tokenizer.vocab_size`: target vocabulary size.
- `experiment.training`: model-training settings, including epochs and checkpoint saving.
- `dataset`: corpus size, seed, representation and cache location. See [dataset settings](../datamodules/README.md).
- `wandb.mode=online`: enable W&B; set your own entity and project.
- `resume=/path/to/checkpoint-N`: restore a full training checkpoint. See [resume details](../experiments/README.md).

Use `key=value` for existing fields and `+key=value` to add a field. Quote list overrides, such as `'experiment.tasks=[evaluation]'`.

Inspect a configuration without training:

```bash
python -m main experiment=tokenizer_benchmark \
  dataset=pubchem10m_mlm_train algorithm=bpe_roberta --cfg job --resolve
```

## Shared fragmentation

`algorithm=train_shared_ps` learns `ps_merges.json` from raw SMILES. Then use `train_fragsmiles` or `train_tsmiles` with `dataset.fragmentation=exocyclic` and, optionally, `dataset.ps_merges_path=PATH`. Keep the same representation settings for MLM and classification.

For Slurm, use the [single-job batch file](../scripts/slurm/single_run.sbatch). Do not combine them with `cluster=...`, which selects a separate submission path.
