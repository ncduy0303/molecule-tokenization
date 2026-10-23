# Cluster job scripts

Run these scripts from the repository root after preparing `.venv` on the cluster. They generate a job script in `slurm_logs/`, activate `.venv`, and submit through `sbatch`.

| Script               | Configured partition | Resource settings in the script                  |
| -------------------- | -------------------- | ------------------------------------------------ |
| `submit_gpu.sh`      | `gpu`                | H100 GPU request, 8 CPUs, 32G memory, 03:00:00   |
| `submit_gpu_long.sh` | `gpu-long`           | H100 GPU request, 8 CPUs, 32G memory, 3-00:00:00 |
| `submit_normal.sh`   | `normal`             | 16 CPUs, 512G memory, `xcnf[0-25]`, 03:00:00     |
| `submit_long.sh`     | `long`               | 16 CPUs, 512G memory, `xcnf[0-25]`, 3-00:00:00   |

These are repository settings, not a verified description of current cluster availability. Edit the resource block for your cluster before submission.

```bash
bash scripts/submit_normal.sh bpe_tokenizer \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train algorithm=train_bpe

bash scripts/submit_gpu.sh bpe_mlm \
  experiment=tokenizer_benchmark dataset=pubchem10m_mlm_train algorithm=bpe_roberta \
  algorithm.tokenizer.pretrained=outputs/TOKENIZER_DATE/TOKENIZER_TIME/tokenizer
```

The first argument is the run name; remaining arguments are Hydra overrides. Use simple run names without spaces or shell metacharacters. The wrappers interpolate overrides into a generated shell script and are intended for trusted command-line input.

Do not pass `cluster=...` to these wrappers, since the wrapper already performs submission. Logs go to `slurm_logs/out_<job-id>.out` and `slurm_logs/err_<job-id>.err`.

## Batch helpers

- `submit_jobs.sh` submits a t-SMILES classification grid over HIV/BBBP, configured variants, and vocabulary sizes. It expects matching local tokenizer and model artifacts. Inspect the grid and paths before running it.
- The repeat loop in `submit_jobs.sh` does not change the training seed. Repeated submissions are not independent-seed trials.
- `mkdir.sh` creates directories for that t-SMILES grid. It does not train or populate models.

The generic Hydra Slurm path is separate and documented in [configurations](../configurations/README.md).
