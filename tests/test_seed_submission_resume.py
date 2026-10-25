"""Exercise the shell submitter with a fake Slurm, never a live scheduler."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SubmissionResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/submit_shared_seed_trials.sh", self.root / "scripts")
        (self.root / "scripts/submit_gpu.sh").write_text(
            '#!/bin/bash\nset -e\necho "Submitting: $PWD/slurm_logs/.job_$1.sh"\nsbatch "$1"\n'
        )
        bindir = self.root / "bin"
        bindir.mkdir()
        sbatch = bindir / "sbatch"
        sbatch.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
p = Path('attempts')
rows = p.read_text().splitlines() if p.exists() else []
rows.append(sys.argv[1])
p.write_text('\\n'.join(rows)+'\\n')
if len(rows) == int(os.environ.get('FAIL_AT', '-1')):
    print('sbatch: error: AssocMaxSubmitJobLimit', file=sys.stderr)
    print('sbatch: error: Batch job submission failed: Job violates accounting/QOS policy', file=sys.stderr)
    sys.exit(1)
print('Submitted batch job', 1000 + len(rows))
""")
        sbatch.chmod(0o755)
        self.env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"])
        for condition in ("base", "ps"):
            for rep in ("fragsmiles", "TSID", "TSDY", "TSIS"):
                key = f"{condition}_{rep}"
                for file in (
                    f"outputs/shared_study_2m_v2/{key}/tokenizer/vocab.json",
                    f"outputs/shared_study_2m_v2/{key}_mlm/final_checkpoint/model.safetensors",
                ):
                    self.write(file)
                for task in ("hiv", "bbbp"):
                    suffix = "_repair2" if rep == "fragsmiles" and task == "hiv" else ""
                    folder = f"data/shared_study_2m_v2/{key}_{task}{suffix}"
                    for file in (f"{task}.csv", f"{task}_split_indices.json", f"{task}_split_indices.json.meta.json"):
                        self.write(f"{folder}/{file}")
                    for split in ("train", "validation", "test"):
                        file = f"{folder}/{task}_canon_{rep}_{split}_seed42.txt"
                        self.write(file)
                        self.write(file + ".meta.json")
        self.write("outputs/shared_study_2m_v2/miner/ps_merges.json")
        self.receipt = self.root / "outputs/shared_study_2m_v2/seed_trials/submissions.log"

    def write(self, path, text="fixture"):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def run_script(self, *args, fail_at=-1):
        return subprocess.run(
            ["bash", "scripts/submit_shared_seed_trials.sh", *args],
            cwd=self.root,
            env=dict(self.env, FAIL_AT=str(fail_at)),
            capture_output=True,
            text=True,
        )

    def test_limit_then_resume_skips_accepted_jobs_even_with_outputs(self):
        first = self.run_script("--submit", fail_at=33)
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(self.receipt.read_text().count("Submitted batch job"), 32)
        self.write("outputs/shared_study_2m_v2/seed_trials/base_fragsmiles_hiv/seed42/result")
        preview = self.run_script("--resume", "--dry-run")
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertEqual(preview.stdout.count("Skipping "), 32)
        self.assertEqual(preview.stdout.count("bash scripts/submit_gpu.sh"), 16)
        self.assertEqual(len((self.root / "attempts").read_text().splitlines()), 33)
        resumed = self.run_script("--resume")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        accepted = self.receipt.read_text().count("Submitted batch job")
        self.assertEqual(accepted, 48)
        attempts = (self.root / "attempts").read_text().splitlines()
        self.assertEqual(len(set(attempts)), 48)
        self.assertEqual(len(attempts), 49)  # One rejected attempt, then retry.
        self.assertEqual(self.run_script("--resume").returncode, 0)
        self.assertEqual((self.root / "attempts").read_text().splitlines(), attempts)

    def test_ambiguous_submission_is_not_retried(self):
        self.write(
            str(self.receipt.relative_to(self.root)),
            "Submitting: /cluster/slurm_logs/.job_SHARED-V2-base-fragsmiles-hiv-SEED42.sh\n",
        )
        result = self.run_script("--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outcome unknown", result.stderr)
        self.assertFalse((self.root / "attempts").exists())

    def test_unrecorded_output_is_not_overwritten(self):
        self.write("outputs/shared_study_2m_v2/seed_trials/base_fragsmiles_hiv/seed42/result")
        result = self.run_script("--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertFalse((self.root / "attempts").exists())

    def test_partial_cache_is_not_reused(self):
        self.write("data/shared_study_2m_v2/seed_trials/base_fragsmiles_hiv/hiv.csv")
        result = self.run_script("--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs or is incomplete", result.stderr)
        self.assertFalse((self.root / "attempts").exists())
