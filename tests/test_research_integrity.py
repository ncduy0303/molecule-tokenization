"""Offline regression tests for dataset provenance and molecular training setup."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.cache_utils import CacheMismatchError, cache_matches, record_cache
from utils.ckpt_utils import resolve_resume_checkpoint
from utils.smirk_utils import load_smirk_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def config(**kwargs):
    return OmegaConf.create(kwargs)


class DatasetIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Unique valid molecules; no external downloads are used.
        self.source = Dataset.from_dict({"SMILES": ["C" * n for n in range(1, 31)]})
        self.token_cfg = config(
            hf_dataset="fixture",
            hf_revision="fixed",
            smiles_column="SMILES",
            data_dir=str(self.root / "tokenizer"),
            corpus_size=6,
            seed=42,
            num_proc=None,
            use_safe=False,
            use_fragsmiles=False,
            use_tsmiles=False,
        )
        self.mlm_cfg = config(
            hf_dataset="fixture",
            hf_revision="fixed",
            smiles_column="SMILES",
            data_dir=str(self.root / "mlm"),
            seed=42,
            train_size=8,
            val_size=3,
            test_size=3,
            num_proc=None,
            max_length=32,
            tokenizer_indices_path=str(self.root / "tokenizer/subset_indices.json"),
        )
        self.download = patch("datasets.load_dataset", return_value=self.source)
        self.download.start()
        self.addCleanup(self.download.stop)

    def build_tokenizer_corpus(self):
        from datamodules.molecule_datasets.pubchem10m_tokenizer_train import load_pubchem10m_tokenizer_corpus

        return load_pubchem10m_tokenizer_corpus(self.token_cfg)

    def build_mlm(self):
        from datamodules.molecule_datasets.pubchem10m_mlm_train import load_pubchem10m_mlm_train

        return load_pubchem10m_mlm_train(self.mlm_cfg, load_smirk_tokenizer())

    def test_disjoint_and_deterministic_selection(self):
        self.build_tokenizer_corpus()
        result = self.build_mlm()
        token = json.loads(Path(self.mlm_cfg.tokenizer_indices_path).read_text())["indices"]
        path = Path(self.mlm_cfg.data_dir) / "subset_indices.json"
        first = path.read_bytes()
        splits = json.loads(first)
        sets = [set(token)] + [set(splits[k]) for k in ("train", "validation", "test")]
        for i, left in enumerate(sets):
            for right in sets[i + 1 :]:
                self.assertFalse(left & right)
        self.assertEqual([len(result[k]) for k in ("train", "validation", "test")], [8, 3, 3])
        with patch("datasets.load_dataset", side_effect=AssertionError("cache should avoid downloads")):
            self.build_tokenizer_corpus()
            self.build_mlm()
        self.assertEqual(first, path.read_bytes())

    def test_tokenizer_cache_rejects_settings_changes(self):
        self.build_tokenizer_corpus()
        for key, value in [("seed", 7), ("corpus_size", 4), ("hf_dataset", "different"), ("hf_revision", "new")]:
            original = self.token_cfg[key]
            self.token_cfg[key] = value
            with self.subTest(key=key), self.assertRaises(CacheMismatchError):
                self.build_tokenizer_corpus()
            self.token_cfg[key] = original

    def test_mlm_cache_rejects_split_size_changes(self):
        self.build_tokenizer_corpus()
        self.build_mlm()
        self.mlm_cfg.train_size = 7
        with self.assertRaises(CacheMismatchError):
            self.build_mlm()

    def test_mlm_requires_verified_same_source_tokenizer_selection(self):
        with self.assertRaisesRegex(ValueError, "Prepare the tokenizer corpus first"):
            self.build_mlm()
        self.build_tokenizer_corpus()
        self.mlm_cfg.hf_revision = "other"
        with self.assertRaises(CacheMismatchError):
            self.build_mlm()

    def test_legacy_and_tampered_corpus_are_rejected(self):
        folder = Path(self.token_cfg.data_dir)
        folder.mkdir()
        (folder / "corpus.txt").write_text("CC\n")
        with self.assertRaises(CacheMismatchError):
            self.build_tokenizer_corpus()
        (folder / "corpus.txt").unlink()
        _, path = self.build_tokenizer_corpus()
        path.write_text("CC\n" * 6)
        with self.assertRaises(CacheMismatchError):
            self.build_tokenizer_corpus()

    def test_safe_slicer_and_word_count_cache_identity(self):
        from datamodules.molecule_datasets.pubchem10m_tokenizer_train import build_word_counts

        self.token_cfg.use_safe = True
        self.token_cfg.safe_slicer = "brics"
        module = "datamodules.molecule_datasets.pubchem10m_tokenizer_train"
        with patch(module + ".encode_safe_batch", side_effect=lambda rows, **kw: rows):
            build_word_counts(self.token_cfg)
            self.token_cfg.safe_slicer = "recap"
            with self.assertRaises(CacheMismatchError):
                build_word_counts(self.token_cfg)
        self.token_cfg.safe_slicer = "brics"
        self.token_cfg.seed = 3
        with self.assertRaises(CacheMismatchError):
            build_word_counts(self.token_cfg)

    def test_changed_exclusion_is_rejected_for_existing_mlm(self):
        self.build_tokenizer_corpus()
        self.build_mlm()
        # Create another legitimate selection at a new location.
        self.token_cfg.data_dir = str(self.root / "other")
        self.token_cfg.seed = 13
        self.build_tokenizer_corpus()
        self.mlm_cfg.tokenizer_indices_path = str(self.root / "other/subset_indices.json")
        with self.assertRaises(CacheMismatchError):
            self.build_mlm()

    def test_source_content_change_and_insufficient_remaining_rows(self):
        self.build_tokenizer_corpus()
        changed = Dataset.from_dict({"SMILES": ["N" * n for n in range(1, 31)]})
        with patch("datasets.load_dataset", return_value=changed):
            with self.assertRaises(CacheMismatchError):
                self.build_mlm()
        self.mlm_cfg.train_size = len(self.source)
        with self.assertRaisesRegex(ValueError, "Not enough source rows"):
            self.build_mlm()


class MoleculeNetCacheTests(unittest.TestCase):
    def test_split_and_processed_strings_track_seed_rows_and_slicer(self):
        from datamodules.molecule_datasets.molnet import load_molnet
        import pandas as pd

        module = "datamodules.molecule_datasets.molnet"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(
                molnet_name="bbbp",
                data_dir=tmp,
                seed=42,
                max_length=32,
                num_proc=None,
                use_safe=True,
                safe_slicer="brics",
            )
            rows = pd.DataFrame({"smiles": ["CC", "CCC", "CO", "CN", "CCCC", "CCO"], "p_np": [0, 1, 0, 1, 0, 1]})
            with (
                patch(module + "._download_csv", return_value=rows),
                patch(module + "._deepchem_split", return_value=([0, 1], [2, 3], [4, 5])) as splitter,
                patch(module + ".encode_safe_batch", side_effect=lambda items, **kw: items),
            ):
                load_molnet(cfg, load_smirk_tokenizer())
                load_molnet(cfg, load_smirk_tokenizer())
                self.assertEqual(splitter.call_count, 1)
                cfg.seed = 7
                with self.assertRaises(CacheMismatchError):
                    load_molnet(cfg, load_smirk_tokenizer())
                cfg.seed = 42
                cfg.safe_slicer = "recap"
                with self.assertRaises(CacheMismatchError):
                    load_molnet(cfg, load_smirk_tokenizer())
                cfg.safe_slicer = "brics"
                rows.loc[0, "p_np"] = 1
                with self.assertRaises(CacheMismatchError):
                    load_molnet(cfg, load_smirk_tokenizer())


class SmirkTests(unittest.TestCase):
    def test_saved_vocabulary_is_used_in_both_stages_and_output_is_isolated(self):
        from experiments.tokenizer_training import TokenizerTrainingExperiment
        from algorithms.downstream_finetune.algo import MoleculeClassificationAlgo
        from algorithms.tokenizer_benchmark.algo import MoleculeMLMAlgo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.txt"
            corpus.write_text("CCO\n")
            with initialize_config_dir(config_dir=str(ROOT / "configurations"), version_base=None):
                cfg = compose(config_name="config")
            experiment = TokenizerTrainingExperiment(cfg, root / "run")
            # Verify the old cwd file is neither created nor overwritten.
            import os

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                sentinel = root / "vocab.json"
                sentinel.write_text("do not touch")
                experiment._train_smirk(corpus, cfg.algorithm)
                self.assertEqual(sentinel.read_text(), "do not touch")
            finally:
                os.chdir(old_cwd)
            for path in (root / "run/tokenizer", root / "run/tokenizer/tokenizer.json"):
                expected = load_smirk_tokenizer(path)
                downstream = MoleculeClassificationAlgo(config(tokenizer={"type": "smirk", "pretrained": str(path)}))
                mlm = MoleculeMLMAlgo.__new__(MoleculeMLMAlgo)
                mlm.cfg = config(tokenizer={"type": "smirk", "pretrained": str(path)})
                actual = mlm._build_tokenizer()
                self.assertEqual(expected.get_vocab(), downstream.tokenizer.get_vocab())
                self.assertEqual(actual("CCO")["input_ids"], downstream.tokenizer("CCO")["input_ids"])
                self.assertEqual(expected.pad_token_id, 1)
            self.assertNotEqual(expected.get_vocab(), load_smirk_tokenizer().get_vocab())


class ResumeTests(unittest.TestCase):
    def test_rejects_legacy_and_incomplete_checkpoints(self):
        cfg = config(load="oldrunid", resume=None, algorithm={"model": {}}, experiment={"tasks": ["training"]})
        with self.assertRaisesRegex(ValueError, "legacy load"):
            resolve_resume_checkpoint(cfg)
        cfg.load = None
        cfg.resume = "oldrunid"
        with self.assertRaisesRegex(ValueError, "W&B run IDs"):
            resolve_resume_checkpoint(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.resume = tmp
            with self.assertRaisesRegex(ValueError, "not a full Trainer checkpoint"):
                resolve_resume_checkpoint(cfg)
            cfg.algorithm = {}
            with self.assertRaisesRegex(ValueError, "only supported"):
                resolve_resume_checkpoint(cfg)


if __name__ == "__main__":
    unittest.main()
