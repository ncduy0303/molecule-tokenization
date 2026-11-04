# Single-run commands

Use the [single-run guide](SINGLE_RUNS.md) for enhanced TSDY vocabulary fitting, CSV preparation, shared PS mining, and intrinsic evaluation. The [baseline walkthrough](../README.md#run-a-baseline) connects tokenizer training, MLM pretraining, and classification.

To run a Hydra training command under Slurm, pass your site's resources to `sbatch`, followed by [the batch file](slurm/single_run.sbatch) and the same Hydra arguments you would pass to `python -m main`. Submit from the repository root with the locked environment installed in `.venv`.
