"""Bounded subprocess tests of full-corpus preflight and checkpoint integrity."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreflightTests(unittest.TestCase):
    def test_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "input.txt"
            out = Path(td) / "result"
            source.write_text("CCO\nCc1ccccc1\nC[C@H](O)C(=O)O\n")
            cmd = [
                sys.executable,
                "-m",
                "scripts.prepare_external_tsdy",
                "--input",
                str(source),
                "--output",
                str(out),
                "--limit",
                "3",
                "--shard-size",
                "2",
                "--workers",
                "1",
            ]

            def run(extra=()):
                return subprocess.run(cmd + list(extra), cwd=ROOT, capture_output=True, text=True)

            first = run()
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue(json.loads((out / "summary.json").read_text())["gate_passed"])
            shard = out / "shard_000000000.jsonl"
            before = shard.stat().st_mtime_ns
            resumed = run(["--resume"])
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(shard.stat().st_mtime_ns, before)
            # Interrupted audit: keep completed shards and rebuild the missing last one.
            (out / "shard_000000002.jsonl").unlink()
            self.assertEqual(run(["--resume"]).returncode, 0)
            self.assertEqual(shard.stat().st_mtime_ns, before)
            shard.write_text("corrupt\n")
            self.assertNotEqual(run(["--resume"]).returncode, 0)
            source.write_text("C\nC\nC\n")
            self.assertNotEqual(run(["--resume"]).returncode, 0)

    def test_failures_are_reported_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "input.txt"
            out = Path(td) / "result"
            source.write_text("CCO\nN->[Cu+2]\n")
            cmd = [
                sys.executable,
                "-m",
                "scripts.prepare_external_tsdy",
                "--input",
                str(source),
                "--output",
                str(out),
                "--limit",
                "2",
                "--max-content-tokens",
                "3",
                "--workers",
                "1",
            ]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["unsupported"], 1)
            self.assertEqual(summary["overlength"], 1)
            self.assertFalse(summary["gate_passed"])
            self.assertFalse((out / "tokenizer").exists())
