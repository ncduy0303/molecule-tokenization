import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock
from scripts.submit_external_tsdy_training import job_plan, render_job, dispatch, validate_receipt


class SubmissionTests(unittest.TestCase):
    def test_dependencies_and_shell_syntax(self):
        plan = job_plan(Path("outputs/enhanced_tsdy_v3_1/preflight_2m"))
        self.assertEqual(len(plan), 14)
        self.assertEqual(plan[1]["deps"], ["prepare"])
        self.assertEqual(sum(j["key"].startswith("enhanced_") for j in plan), 6)
        self.assertEqual(sum(j["key"].startswith(("TSDY_", "fragsmiles_")) for j in plan), 6)
        for job in plan:
            script = render_job(job, Path("/tmp/project with spaces"))
            self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode, 0)
        self.assertIn("${SLURM_JOB_ID}", render_job(plan[0], Path("/tmp/project")))
        for job in plan[2:]:
            expected = "mlm" if job["key"].startswith("enhanced_") else "prepare"
            self.assertEqual(job["deps"], [expected])

    def test_receipts_do_not_duplicate_or_retry_ambiguous_submissions(self):
        with tempfile.TemporaryDirectory() as td:
            receipt = {}
            path = Path(td) / "receipt.json"
            job = job_plan(Path("x"))[0]
            runner = Mock(return_value=subprocess.CompletedProcess([], 0, "12345\n", ""))
            self.assertEqual(dispatch(job, "job.sh", [], receipt, path, runner), "12345")
            self.assertEqual(dispatch(job, "job.sh", [], receipt, path, runner), "12345")
            self.assertEqual(runner.call_count, 1)
            validate_receipt(receipt)
            ambiguous = Mock(return_value=subprocess.CompletedProcess([], 0, "unexpected response", ""))
            with self.assertRaises(RuntimeError):
                dispatch(dict(job, key="new"), "job2.sh", ["12345"], receipt, path, ambiguous)
            with self.assertRaises(ValueError):
                validate_receipt(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
