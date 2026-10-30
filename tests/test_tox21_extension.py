import unittest
import subprocess
import sys
from scripts.prepare_tox21_extension import select_rows, audit_row
from scripts.submit_tox21_extension import plan


class Tox21ExtensionTests(unittest.TestCase):
    def test_login_node_imports_do_not_load_torch(self):
        code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise RuntimeError('Training dependency loaded on login node: ' + fullname)
sys.meta_path.insert(0, BlockTorch())
from scripts.submit_tox21_extension import plan, scheduler
from pathlib import Path
for stage in ('prepare', 'finetune'):
    for job in plan(stage):
        scheduler.render_job(job, Path.cwd())
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_missing_labels_preserve_source_ids(self):
        rows = [{"SR-p53": str(i % 2)} for i in range(100)]
        rows[7]["SR-p53"] = ""
        groups, missing = select_rows(rows)
        self.assertEqual(missing, [7])
        self.assertEqual(set(sum(groups.values(), [])), set(range(100)) - {7})
        self.assertEqual(groups, select_rows(rows)[0])
        self.assertEqual(len(sum(groups.values(), [])), 99)

    def test_bad_label_blocks(self):
        with self.assertRaises(ValueError):
            select_rows([{"SR-p53": "2"}])

    def test_shared_encoding_and_failure(self):
        self.assertFalse(audit_row((0, "Cc1ccccc1"))["errors"])
        self.assertTrue(audit_row((1, "not a molecule"))["errors"])

    def test_jobs_use_existing_mlms_and_fixed_dataset(self):
        jobs = plan("finetune")
        self.assertEqual(len(jobs), 9)
        self.assertEqual(len({j["output"] for j in jobs}), 9)
        for job in jobs:
            args = job["commands"][0]
            self.assertIn("dataset=tox21_sr_p53", args)
            self.assertIn("experiment=downstream_classification", args)
            self.assertTrue(any(a.startswith("algorithm.model.pretrained=") for a in args))


if __name__ == "__main__":
    unittest.main()
