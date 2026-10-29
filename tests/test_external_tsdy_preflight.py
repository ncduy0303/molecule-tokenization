"""Bounded subprocess tests of full-corpus preflight and checkpoint integrity."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreflightTests(unittest.TestCase):
    def test_incomplete_worker_results_are_not_published(self):
        from scripts.prepare_external_tsdy import write_shard

        class IncompletePool:
            def map(self, function, items, chunksize):
                return iter([dict(row=0, ok=True)])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shard.jsonl"
            with self.assertRaisesRegex(ValueError, "not published"):
                write_shard(path, IncompletePool(), ["C", "O"], 0, 510)
            self.assertFalse(path.exists())
            self.assertFalse(Path(str(path) + ".sha256").exists())

    def test_fixed_audit_scope_fails_on_new_error_or_changed_source(self):
        import hashlib
        from scripts.audit_external_tsdy_large import check_scope

        failure = dict(group="hiv_train", row=35278, smiles="N->[Cu+2]", ok=False, error="nonstandard")
        expected = {k: failure[k] for k in ("group", "row", "error")}
        expected["smiles_sha256"] = hashlib.sha256(failure["smiles"].encode()).hexdigest()
        contract = dict(source_hashes={"input": "hash"}, known_unsupported=[expected], max_content_tokens=510)
        self.assertTrue(check_scope([failure], {"input": "hash"}, contract)["passed"])
        self.assertFalse(check_scope([failure], {"input": "different"}, contract)["passed"])
        self.assertFalse(check_scope([dict(failure, row=1)], {"input": "hash"}, contract)["passed"])
        long = dict(group="hiv_test", row=0, ok=True, compact_length=511)
        self.assertFalse(check_scope([failure, long], {"input": "hash"}, contract)["passed"])

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
            self.assertFalse((out / "vocabulary").exists())

    def test_long_reconstructable_rows_contribute_to_vocabulary(self):
        from scripts.prepare_external_tsdy import SPECIAL_TOKENS
        from utils.external_tsdy import COMPACT_GRAMMAR, TOKEN_FORMAT

        with tempfile.TemporaryDirectory() as td:
            source, out = Path(td) / "input.txt", Path(td) / "result"
            source.write_text("CCO\n")
            command = [
                sys.executable,
                "-m",
                "scripts.prepare_external_tsdy",
                "--input",
                str(source),
                "--output",
                str(out),
                "--limit",
                "1",
                "--workers",
                "1",
                "--max-content-tokens",
                "3",
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["vocabulary_gate_passed"])
            self.assertFalse(summary["model_length_gate_passed"])
            self.assertEqual(summary["unsupported"], 0)
            self.assertEqual(json.loads((out / "eligibility.json").read_text())["overlength_rows"], [0])
            self.assertEqual(json.loads((out / "core_counts.json").read_text()), {"C": 2, "O": 1})
            vocab = json.loads((out / "vocabulary/vocab.json").read_text())
            self.assertTrue(set(SPECIAL_TOKENS) | COMPACT_GRAMMAR | {"C", "O"} == set(vocab))
            self.assertEqual(sorted(vocab.values()), list(range(len(vocab))))
            metadata = json.loads((out / "vocabulary/metadata.json").read_text())
            self.assertEqual(metadata["token_format"], TOKEN_FORMAT)
            self.assertTrue(metadata["counts_include_overlength"])

    def test_vocabulary_budget_and_tie_break_are_deterministic(self):
        from scripts.prepare_external_tsdy import vocabulary_artifacts

        counts = {f"core{i:05}": 1 for i in reversed(range(6000))}
        vocab, metadata = vocabulary_artifacts(counts, {})
        self.assertEqual(len(vocab), 5000)
        self.assertIn("core00000", vocab)
        self.assertIn("core04911", vocab)
        self.assertNotIn("core04912", vocab)
        self.assertEqual(metadata["retained_core_occurrences"], 4912)
