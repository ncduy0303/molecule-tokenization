"""Preview/submit SR-p53 preparation or nine finetunes; receipt-safe resumption."""

import argparse
import json
from pathlib import Path
import shutil
from scripts import submit_external_tsdy_training as scheduler
from scripts.prepare_tox21_extension import DATA, TOKENIZERS

STUDY = Path("outputs/tox21_sr_p53")
MODELS = {
    k: (
        Path("outputs/enhanced_tsdy_v3_1/mlm/final_checkpoint")
        if k == "enhanced"
        else Path(f"outputs/shared_study_2m_v2/base_{k}_mlm/final_checkpoint")
    )
    for k in TOKENIZERS
}


def plan(stage):
    if stage == "prepare":
        return [
            dict(
                key="prepare",
                name="TOX21-SR-p53-PREPARE",
                kind="cpu",
                tags="tox21,sr_p53,preparation",
                commands=[[".venv/bin/python", "-u", "-m", "scripts.prepare_tox21_extension", "--resume"]],
            )
        ]
    jobs = []
    for kind in TOKENIZERS:
        algorithm = {
            "enhanced": "external_tsdy_classifier",
            "TSDY": "tsmiles_classifier",
            "fragsmiles": "fragsmiles_classifier",
        }[kind]
        for seed in (42, 43, 44):
            key = f"{kind}_seed{seed}"
            name = f"TOX21-SR-p53-{kind}-SEED{seed}"
            output = STUDY / "trials" / key
            jobs.append(
                dict(
                    key=key,
                    name=name,
                    kind="gpu",
                    output=str(output),
                    tags=f"tox21,sr_p53,{kind},seed{seed},common_domain",
                    commands=[
                        [
                            ".venv/bin/python",
                            "-m",
                            "main",
                            f"+name={name}",
                            "experiment=downstream_classification",
                            "dataset=tox21_sr_p53",
                            f"algorithm={algorithm}",
                            f"dataset.prepared_path={DATA}/{kind}",
                            f"algorithm.tokenizer.pretrained={TOKENIZERS[kind]}",
                            f"algorithm.model.pretrained={MODELS[kind]}",
                            f"experiment.training.seed={seed}",
                            f"hydra.run.dir={output}",
                            "wandb.mode=online",
                            "wandb.entity=ncduy0303",
                            "wandb.project=molecule-tokenization",
                        ]
                    ],
                )
            )
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "finetune"])
    p.add_argument("--submit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gpu", default="gpu:h100-96:1")
    a = p.parse_args()
    scheduler.STUDY = STUDY
    jobs = plan(a.stage)
    if not a.submit:
        for job in jobs:
            print(scheduler.render_job(job, Path.cwd(), a.gpu))
        return
    if shutil.which("sbatch") is None:
        raise ValueError("Submit on the Slurm login host")
    if a.stage == "finetune":
        from omegaconf import OmegaConf
        from scripts.prepare_external_tsdy_training import load_tokenizer
        from datamodules.molecule_datasets.prepared import load_prepared

        domains = []
        for kind, path in TOKENIZERS.items():
            load_prepared(
                OmegaConf.create(dict(prepared_path=str(DATA / kind), prepared_task="tox21_sr_p53", max_length=512)),
                load_tokenizer(path, kind),
            )
            manifest = json.loads((DATA / kind / "manifest.json").read_text())
            domains.append(manifest["source"])
            if not (MODELS[kind] / "model.safetensors").is_file():
                raise ValueError(f"Missing model: {MODELS[kind]}")
        if any(d != domains[0] for d in domains):
            raise ValueError("Prepared common domains differ")
    STUDY.mkdir(parents=True, exist_ok=True)
    Path("slurm_logs").mkdir(exist_ok=True)
    ledger = STUDY / "submissions.json"
    lock = STUDY / ".submission-lock"
    lock.mkdir()
    try:
        receipt = json.loads(ledger.read_text()) if ledger.exists() else {}
        scheduler.validate_receipt(receipt)
        if not a.resume and any(j["key"] in receipt for j in jobs):
            raise ValueError("Already submitted; use --resume to skip acknowledged jobs")
        for job in jobs:
            if receipt.get(job["key"], {}).get("state") == "submitted":
                print(f"Skipping {job['key']}: {receipt[job['key']]['job_id']}")
                continue
            if job["kind"] == "gpu" and Path(job["output"]).exists():
                raise ValueError(f"Output already exists: {job['output']}")
            script = Path("slurm_logs") / f".job_{job['name']}.sh"
            script.write_text(scheduler.render_job(job, Path.cwd(), a.gpu))
            job_id = scheduler.dispatch(job, script, [], receipt, ledger)
            print(f"Submitted {job['key']}: {job_id}", flush=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
