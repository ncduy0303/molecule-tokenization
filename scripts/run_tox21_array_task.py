"""Run one existing SR-p53 finetune plan entry inside a Slurm GPU allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from scripts.submit_tox21_extension import plan
from scripts.prepare_tox21_extension import DATA, TOKENIZERS
from scripts.prepare_external_tsdy_training import load_tokenizer
from datamodules.molecule_datasets.prepared import load_prepared
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(9), required=True)
    args = parser.parse_args()
    # Validate the entire matched study before running any individual condition.
    sources = []
    for kind, path in TOKENIZERS.items():
        load_prepared(
            OmegaConf.create(dict(prepared_path=str(DATA / kind), prepared_task="tox21_sr_p53", max_length=512)),
            load_tokenizer(path, kind),
        )
        sources.append(json.loads((DATA / kind / "manifest.json").read_text())["source"])
    if any(source != sources[0] for source in sources):
        raise ValueError("Prepared common domains differ")
    job = plan("finetune")[args.index]
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".lock")
    lock.mkdir()
    try:
        if output.exists():
            raise ValueError(f"Output already exists; inspect before retrying: {output}")
        env = dict(os.environ, WANDB_TAGS=job["tags"])
        for command in job["commands"]:
            subprocess.run(command, env=env, check=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
