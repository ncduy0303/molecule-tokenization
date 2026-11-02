# Scripts

Run scripts from the repository root. For Slurm submissions, prepare `.venv` on the cluster and edit the wrapper's resource settings for your environment.

| Wrapper              | Intended job                 |
| -------------------- | ---------------------------- |
| `submit_normal.sh`   | CPU job on `normal`          |
| `submit_long.sh`     | Longer CPU job on `long`     |
| `submit_gpu.sh`      | GPU job on `gpu`             |
| `submit_gpu_long.sh` | Longer GPU job on `gpu-long` |

The first argument is the run name; remaining arguments are Hydra overrides:

```bash
bash scripts/submit_normal.sh bpe_tokenizer \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train algorithm=train_bpe
```

Use the stage settings and artifact paths from the [main README](../README.md). The wrappers activate `.venv`, generate a script, and call `sbatch`. Do not add `cluster=...`, which would trigger a second submission path.

Logs are written to `slurm_logs/out_<job-id>.out` and `slurm_logs/err_<job-id>.err`. Use simple run names and trusted overrides, since arguments are interpolated into shell scripts.

Other scripts provide experiment grids, audits and result analysis. Inspect their arguments and artifact paths before running them. In particular, the legacy `submit_jobs.sh` repeat loop does not change the seed, so its repeats are not independent-seed trials.
