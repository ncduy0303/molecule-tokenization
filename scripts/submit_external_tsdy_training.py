"""Preview or submit the enhanced study, with afterok dependencies and receipts.

Default is preview. --submit performs submissions. --resume skips acknowledged
jobs; an unresolved submission intent is never silently retried.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from scripts.prepare_external_tsdy import write_json

STUDY = Path("outputs/enhanced_tsdy_v3_1")
DATA = Path("data/enhanced_tsdy_v3_1")


def job_plan(preflight, resume_preparation=False):
    python = ".venv/bin/python"
    prep = [
        python,
        "-u",
        "-m",
        "scripts.prepare_external_tsdy_training",
        "--preflight",
        str(preflight),
        "--controls",
        "--workers",
        "8",
        "--length-policy",
        "truncate",
    ]
    if resume_preparation:
        prep.append("--resume")
    plan = [
        dict(
            key="prepare",
            name="ENHANCED-TSDY-V3_1-PREPARE",
            kind="cpu",
            deps=[],
            commands=[
                [
                    python,
                    "-m",
                    "unittest",
                    "tests.test_external_tsdy",
                    "tests.test_external_tsdy_preflight",
                    "tests.test_external_tsdy_integration",
                ],
                prep,
                [python, "-u", "-m", "scripts.smoke_external_tsdy_training", "--output", str(STUDY / "cluster_smoke")],
            ],
            tags="enhanced_tsdy_v3_1,preparation",
        )
    ]
    common = ["wandb.mode=online", "wandb.entity=ncduy0303", "wandb.project=molecule-tokenization"]

    def training(key, name, experiment, dataset, algorithm, output, overrides, deps, tags):
        return dict(
            key=key,
            name=name,
            kind="gpu",
            deps=deps,
            output=str(output),
            tags=tags,
            commands=[
                [
                    python,
                    "-m",
                    "main",
                    f"+name={name}",
                    f"experiment={experiment}",
                    f"dataset={dataset}",
                    f"algorithm={algorithm}",
                    f"hydra.run.dir={output}",
                    *overrides,
                    *common,
                ]
            ],
        )

    plan.append(
        training(
            "mlm",
            "SHARED-EXO-ENHANCED-TSDY-V3_1-5K-1M-20Epoch",
            "tokenizer_benchmark",
            "external_tsdy_mlm",
            "external_tsdy_roberta",
            STUDY / "mlm",
            ["experiment.training.seed=42", "experiment.training.save_strategy=no"],
            ["prepare"],
            "canonical,enhanced_tsdy,exocyclic,mlm,v3_1",
        )
    )
    for kind in ("enhanced", "TSDY", "fragsmiles"):
        for task in (("hiv", "bbbp") if kind == "enhanced" else ("hiv",)):
            for seed in (42, 43, 44):
                key = f"{kind}_{task}_seed{seed}"
                name = f"ENHANCED-STUDY-{kind}-{task}-SEED{seed}"
                algorithm = (
                    "external_tsdy_classifier"
                    if kind == "enhanced"
                    else ("tsmiles_classifier" if kind == "TSDY" else "fragsmiles_classifier")
                )
                overrides = [f"dataset.prepared_path={DATA}/{kind}_{task}", f"experiment.training.seed={seed}"]
                if kind != "enhanced":
                    overrides += [
                        f"algorithm.tokenizer.pretrained=outputs/shared_study_2m_v2/base_{kind}/tokenizer",
                        f"algorithm.model.pretrained=outputs/shared_study_2m_v2/base_{kind}_mlm/final_checkpoint",
                    ]
                plan.append(
                    training(
                        key,
                        name,
                        "downstream_classification",
                        f"external_tsdy_{task}",
                        algorithm,
                        STUDY / "trials" / key,
                        overrides,
                        ["mlm"] if kind == "enhanced" else ["prepare"],
                        f"enhanced_study_v3_1,{kind},{task},seed{seed},common_domain",
                    )
                )
    return plan


def render_job(job, root, gpu="gpu:h100-96:1"):
    lines = [
        "#!/bin/bash",
        f'#SBATCH --job-name={job["name"]}',
        f'#SBATCH --partition={"long" if job["kind"]=="cpu" else "gpu"}',
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        f'#SBATCH --time={"3-00:00:00" if job["kind"]=="cpu" else "03:00:00"}',
        f"#SBATCH --output={root}/slurm_logs/out_%j.out",
        f"#SBATCH --error={root}/slurm_logs/err_%j.err",
    ]
    if job["kind"] == "gpu":
        lines.append(f"#SBATCH --gres={gpu}")
    lines += [
        "set -euo pipefail",
        f"cd {shlex.quote(str(root))}",
        "export TOKENIZERS_PARALLELISM=false",
        "export WANDB_LOG_MODEL=false",
        f'export WANDB_TAGS={shlex.quote(job["tags"])}',
    ]
    if job["kind"] == "gpu":
        lines += ["nvidia-smi"]
        # Fail at execution too, including if the submitter predated an output creation.
        lines += [
            f'test ! -e {shlex.quote(job["output"])} || {{ echo "Run output exists; inspect before retrying" >&2; exit 1; }}'
        ]
    for command in job["commands"]:
        if command[3:4] == ["scripts.smoke_external_tsdy_training"]:
            # Each successful/retried CPU allocation gets its own diagnostic model output.
            command = command[:-1] + [command[-1] + "_${SLURM_JOB_ID}"]
            lines.append(shlex.join(command[:-1]) + ' "' + command[-1] + '"')
        else:
            lines.append(shlex.join(command))
    marker = root / STUDY / "completed" / f'{job["key"]}.json'
    lines += [
        f"mkdir -p {shlex.quote(str(marker.parent))}",
        'printf \'{"job_id":"%s","passed":true}\\n\' "$SLURM_JOB_ID" > ' + shlex.quote(str(marker)),
    ]
    return "\n".join(lines) + "\n"


def completed(key, job_id):
    marker = STUDY / "completed" / f"{key}.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    return record.get("passed") is True and str(record.get("job_id")) == str(job_id)


def validate_receipt(receipt):
    for key, entry in receipt.items():
        if entry["state"] not in ("submitted", "rejected"):
            raise ValueError(
                f"Submission outcome unknown for {key}. Inspect squeue/sacct and reconcile the receipt before retrying."
            )
        if entry["state"] == "submitted" and not re.fullmatch(r"\d+", str(entry["job_id"])):
            raise ValueError("Invalid acknowledged job ID")


def dispatch(job, script, dependency_ids, receipt, receipt_path, runner=subprocess.run):
    key = job["key"]
    if key in receipt and receipt[key]["state"] == "submitted":
        return str(receipt[key]["job_id"])
    receipt[key] = dict(state="pending", name=job["name"], script=str(script))
    write_json(receipt_path, receipt)  # Crash before/after acceptance is deliberately ambiguous.
    args = ["sbatch", "--parsable"]
    if dependency_ids:
        args.append("--dependency=afterok:" + ":".join(dependency_ids))
    args.append(str(script))
    result = runner(args, text=True, capture_output=True)
    if result.returncode:
        receipt[key].update(state="rejected", stderr=result.stderr)
        write_json(receipt_path, receipt)
        raise RuntimeError(result.stderr)
    match = re.fullmatch(r"(\d+)(?:;[^\s]+)?", result.stdout.strip())
    if not match:
        raise RuntimeError(
            "Unrecognized sbatch acknowledgement; receipt remains pending. Inspect Slurm before retrying."
        )
    job_id = match[1]
    receipt[key].update(state="submitted", job_id=job_id)
    write_json(receipt_path, receipt)
    return job_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "mlm", "finetune", "controls", "all"])
    p.add_argument("--submit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--resume-preparation", action="store_true", help="CPU worker may reuse identical completed preparation shards"
    )
    p.add_argument("--preflight", type=Path, default=STUDY / "preflight_2m")
    p.add_argument("--gpu", default="gpu:h100-96:1")
    a = p.parse_args()
    root = Path.cwd()
    plan = job_plan(a.preflight, a.resume_preparation)
    selected = [
        j
        for j in plan
        if a.stage == "all"
        or (a.stage in ("prepare", "mlm") and j["key"] == a.stage)
        or (a.stage == "finetune" and j["key"].startswith("enhanced_"))
        or (a.stage == "controls" and j["key"].startswith(("TSDY_", "fragsmiles_")))
    ]
    ledger = STUDY / "submissions.json"
    receipt = json.loads(ledger.read_text()) if ledger.exists() else {}
    validate_receipt(receipt)
    if not a.submit:
        for j in selected:
            print(f'{j["key"]}: dependencies={j["deps"]}; {j["name"]}')
            print(render_job(j, root, a.gpu))
        return
    if a.resume and not receipt:
        raise ValueError("No receipt to resume")
    if not a.resume and any(j["key"] in receipt for j in selected):
        raise ValueError("Selected jobs already have receipts; use --resume, not duplicate submissions")
    import shutil

    if shutil.which("sbatch") is None:
        raise ValueError("Run --submit on the Slurm login host; sbatch is not available here")
    preflight_summary = json.loads((a.preflight / "summary.json").read_text())
    if not preflight_summary.get("vocabulary_gate_passed") or preflight_summary.get("rows") != 2000000:
        raise ValueError("Require the verified full 2M v3.1 preflight")
    if not (root / ".venv/bin/python").is_file():
        raise ValueError("Missing project environment")
    # Dependency receipts may refer to running jobs; Slurm afterok enforces their success.
    known = {j["key"] for j in selected} | {k for k, e in receipt.items() if e["state"] == "submitted"}
    for j in selected:
        if set(j["deps"]) - known:
            raise ValueError(f'{j["key"]}: submit the preceding stage first')
        if (
            j["kind"] == "gpu"
            and (root / j["output"]).exists()
            and receipt.get(j["key"], {}).get("state") != "submitted"
        ):
            raise ValueError(f'Existing run output: {j["output"]}')
    for kind in ("TSDY", "fragsmiles"):
        if any(j["key"].startswith(kind + "_") for j in selected):
            for path in (
                f"outputs/shared_study_2m_v2/base_{kind}/tokenizer/vocab.json",
                f"outputs/shared_study_2m_v2/base_{kind}_mlm/final_checkpoint/config.json",
            ):
                if not Path(path).is_file():
                    raise ValueError(f"Missing frozen control: {path}")
    STUDY.mkdir(parents=True, exist_ok=True)
    Path("slurm_logs").mkdir(exist_ok=True)
    lock = STUDY / ".submission-lock"
    lock.mkdir()
    try:
        # Re-read after acquiring the lock to reject a concurrent stale submission.
        if ledger.exists():
            current = json.loads(ledger.read_text())
            if current != receipt:
                raise ValueError("Receipt changed; rerun with --resume")
        for j in selected:
            previous = receipt.get(j["key"], {})
            if previous.get("state") == "submitted":
                print(f'Skipping {j["key"]}: submitted as {previous["job_id"]}')
                continue
            script = Path("slurm_logs") / f'.job_{j["name"]}.sh'
            script.write_text(render_job(j, root, a.gpu))
            deps = [str(receipt[k]["job_id"]) for k in j["deps"] if not completed(k, receipt[k]["job_id"])]
            job_id = dispatch(j, script, deps, receipt, ledger)
            print(f'Submitted {j["key"]}: {job_id}', flush=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
