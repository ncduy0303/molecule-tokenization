"""Submit small waves and persist Slurm receipts. Run from the repository root."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import time

from scripts.ca_seed_trials import DATA, ROOT, trials


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def queue():
    args = ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%A"]
    return subprocess.check_output(args, text=True).splitlines()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--watch",
        action="store_true",
        help="Keep feeding small batches; otherwise submit one batch",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=8,
        help="Target maximum queued/running tasks for this user",
    )
    p.add_argument(
        "--retry",
        type=int,
        nargs="+",
        default=[],
        help="Explicit failed trial indices to retry",
    )
    p.add_argument(
        "--minutes",
        type=int,
        default=15,
        help="GPU wall time, including checkpoint export",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also select recorded trials with a confirmed failed or cancelled Slurm state",
    )
    a = p.parse_args()
    rows = trials()
    if a.minutes < 1 or a.cap < 1 or any(i not in range(len(rows)) for i in a.retry):
        p.error("Invalid cap or retry index")
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = ROOT / "submissions.json"
    pending = ROOT / "submission_pending.json"
    if pending.exists():
        raise SystemExit(f"Ambiguous previous submission. Reconcile {pending} with squeue/sacct before continuing.")
    history = json.loads(receipt.read_text()) if receipt.exists() else {}
    missing = [
        str(DATA / f"{key}_{task}")
        for (key, task, *_), seed in rows
        if seed == 42 and not (DATA / f"{key}_{task}" / "ready.json").is_file()
    ]
    if missing:
        raise SystemExit(f"Finish ca_prepare.sbatch first. Missing {missing[0]}")
    retry = set(a.retry)
    while True:
        active = queue()
        todo = []
        unfinished = []
        for i, ((key, task, *_), seed) in enumerate(rows):
            if (ROOT / "trials" / f"{key}_{task}" / f"seed{seed}" / "complete.json").is_file():
                continue
            ids = history.get(str(i), [])
            if not ids:
                todo.append(i)
            elif i in retry or a.retry_failed:
                state = (
                    subprocess.check_output(
                        [
                            "sacct",
                            "-X",
                            "-nP",
                            "-j",
                            f"{ids[-1]}_{i}",
                            "--format=State",
                        ],
                        text=True,
                    )
                    .strip()
                    .splitlines()
                )
                if state and state[0].split()[0] in {
                    "FAILED",
                    "TIMEOUT",
                    "CANCELLED",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "PREEMPTED",
                }:
                    todo.append(i)
                elif a.retry_failed and i not in retry:
                    unfinished.append(i)
                else:
                    raise SystemExit(f"Trial {i}: retry requires confirmed terminal failure; sacct returned {state}")
            else:
                unfinished.append(i)
        if not todo:
            print(
                "All selected trials submitted. Outstanding without completion marker:",
                unfinished,
            )
            return
        n = min(len(todo), max(0, a.cap - len(active)))
        while n:
            batch = todo[:n]
            write(pending, batch)
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    f"--time={a.minutes}",
                    "--array=" + ",".join(map(str, batch)),
                    "scripts/slurm/ca_trial.sbatch",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                print(result.stderr.strip(), flush=True)
                if (
                    "Batch job submission failed:" not in result.stderr
                    and "QOSMaxSubmitJobPerUserLimit" not in result.stderr
                ):
                    raise SystemExit(f"Uncertain outcome: inspect {pending}")
                pending.unlink()
                if not re.search(r"limit|quota|policy", result.stderr, re.I):
                    raise SystemExit(result.returncode)
                n //= 2
                continue
            job = result.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise SystemExit(f"Unrecognized sbatch acknowledgment: inspect {pending}")
            for i in batch:
                history.setdefault(str(i), []).append(job)
            write(receipt, history)
            pending.unlink()
            retry.difference_update(batch)
            print(f"Accepted array {job}, trials {batch}", flush=True)
            break
        if not a.watch:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
