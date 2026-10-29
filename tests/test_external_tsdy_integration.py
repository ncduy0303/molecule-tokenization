import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, set_seed
from utils.cache_utils import value_digest
from utils.external_tsdy import COMPACT_GRAMMAR, encode, model_tokens, molecules_equal
from utils.external_tsdy_tokenizer import ExternalTSDYTokenizer, SPECIALS, tokenizer_identity
from scripts.prepare_external_tsdy_training import prepare_task, init_worker, encode_record
from datamodules.molecule_datasets.prepared import load_prepared


def fixture_tokenizer(folder):
    from utils.external_tsdy import TOKEN_FORMAT

    cores = set()
    for smi in ["CCO", "Cc1ccccc1", "C[C@H](O)C(=O)O", "F/C=C/F", "[Na+].[Cl-]"]:
        cores.update(s for c in json.loads(encode(smi))["components"] for s in c["cores"])
    vocab = {s: i for i, s in enumerate(SPECIALS + sorted(COMPACT_GRAMMAR) + sorted(cores))}
    path = Path(folder) / "vocab.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab))
    tok = ExternalTSDYTokenizer(
        str(path), token_format=TOKEN_FORMAT, vocabulary_identity=value_digest(vocab), model_max_length=512
    )
    tok.save_pretrained(str(folder))
    return tok


class IntegrationTests(unittest.TestCase):
    def test_structured_words_reload_masking_and_unknowns(self):
        with tempfile.TemporaryDirectory() as td:
            tok = fixture_tokenizer(Path(td) / "tok")
            reloaded = ExternalTSDYTokenizer.from_pretrained(str(Path(td) / "tok"), local_files_only=True)
            self.assertEqual(tokenizer_identity(tok), tokenizer_identity(reloaded))
            for smi in ["CCO", "C[C@H](O)C(=O)O", "F/C=C/F", "[Na+].[Cl-]"]:
                ids = tok(smi)["input_ids"]
                self.assertEqual(
                    ids, tok.build_inputs_with_special_tokens(tok.convert_tokens_to_ids(model_tokens(encode(smi))))
                )
                self.assertTrue(molecules_equal(smi, reloaded.decode_molecule(ids)))
            batch = tok(["CCO", "Cc1ccccc1"], padding=True, return_tensors="pt")
            self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
            collator = DataCollatorForLanguageModeling(tok, mlm_probability=1.0)
            masked = collator([tok("CCO")])
            self.assertEqual(masked["labels"][0, 0].item(), -100)
            self.assertEqual(masked["labels"][0, -1].item(), -100)
            self.assertIn(tok.convert_tokens_to_ids("<NULL>"), masked["labels"][0].tolist())
            with self.assertRaises(ValueError):
                tok.decode_molecule(tok("[Xe]")["input_ids"])
            with self.assertRaises(ValueError):
                tok.decode_molecule([tok.mask_token_id])
            with self.assertRaises(ValueError):
                ExternalTSDYTokenizer(str(Path(td) / "tok/vocab.json"), token_format="old")

    def test_preparation_resume_guard_and_read_only_loading(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            tok = fixture_tokenizer(folder / "tok")
            groups = {s: [i] for i, s in enumerate(("train", "validation", "test"))}

            def provider(split, start, stop):
                return [dict(source_row=i, text="CCO", labels=[float(i % 2)]) for i in groups[split][start:stop]]

            root = folder / "data"
            prepare_task(
                root, groups, provider, {"fixture": True}, folder / "tok", "enhanced", "hiv", workers=1, smoke=True
            )
            cfg = OmegaConf.create(
                dict(prepared_path=str(root), prepared_task="hiv", max_length=512, allow_smoke=False)
            )
            with self.assertRaises(ValueError):
                load_prepared(cfg, tok)
            cfg.allow_smoke = True
            data = load_prepared(cfg, tok)
            self.assertEqual(data["train"][0]["labels"], [0.0])
            before = (root / "shards/train/000000000.jsonl").stat().st_mtime_ns
            prepare_task(
                root,
                groups,
                provider,
                {"fixture": True},
                folder / "tok",
                "enhanced",
                "hiv",
                workers=1,
                smoke=True,
                resume=True,
            )
            self.assertEqual((root / "shards/train/000000000.jsonl").stat().st_mtime_ns, before)
            bad = fixture_tokenizer(folder / "other")
            bad.add_tokens(["newcore"])
            with self.assertRaises(ValueError):
                load_prepared(cfg, bad)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            with self.assertRaises(ValueError):
                load_prepared(cfg, tok)

    def test_truncation_is_explicit_and_unsupported_never_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            tok = fixture_tokenizer(Path(td) / "tok")
            init_worker(str(Path(td) / "tok"), "enhanced")
            row = dict(source_row=7, text="CCCCCCCC")
            result = encode_record((row, 8, "error"))
            self.assertFalse(result["ok"])
            self.assertTrue(result["overlength"])
            result = encode_record((row, 8, "truncate"))
            self.assertTrue(result["ok"])
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["input_ids"]), 8)
            self.assertEqual(result["input_ids"][-1], tok.sep_token_id)
            self.assertFalse(encode_record((dict(source_row=8, text="N->[Cu+2]"), 512, "truncate"))["ok"])

    def test_prepared_mlm_preserves_frozen_unknown_masking(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            tok = fixture_tokenizer(folder / "tok")
            groups = {s: [i] for i, s in enumerate(("train", "validation", "test"))}

            def provider(split, start, stop):
                return [dict(source_row=i, text="[Xe]") for i in groups[split][start:stop]]

            prepare_task(
                folder / "data",
                groups,
                provider,
                {"fixture": True},
                folder / "tok",
                "enhanced",
                "mlm",
                workers=1,
                smoke=True,
            )
            cfg = OmegaConf.create(
                dict(prepared_path=str(folder / "data"), prepared_task="mlm", max_length=512, allow_smoke=True)
            )
            data = load_prepared(cfg, tok)
            self.assertEqual(data["train"][0]["special_tokens_mask"], [1, 0, 1])
            collator = DataCollatorForLanguageModeling(tok, mlm_probability=1.0)
            batch = collator([data["train"][0]])
            self.assertEqual(batch["labels"][0, 1].item(), tok.unk_token_id)
            self.assertNotIn("special_tokens_mask", batch)

    def test_seeded_model_and_classifier_tokenizer_binding(self):
        from algorithms.tokenizer_benchmark.algo import MoleculeMLMAlgo
        from algorithms.downstream_finetune.algo import MoleculeClassificationAlgo

        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            fixture_tokenizer(folder / "tok")
            cfg = OmegaConf.create(
                dict(
                    tokenizer=dict(type="external_tsdy", pretrained=str(folder / "tok")),
                    model=dict(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        max_position_embeddings=514,
                    ),
                )
            )
            set_seed(42)
            a = MoleculeMLMAlgo(cfg)
            set_seed(42)
            b = MoleculeMLMAlgo(cfg)
            self.assertTrue(
                torch.equal(
                    a.model.roberta.embeddings.word_embeddings.weight, b.model.roberta.embeddings.word_embeddings.weight
                )
            )
            a.model.save_pretrained(folder / "mlm")
            cfg.model.pretrained = str(folder / "mlm")
            classifier = MoleculeClassificationAlgo(cfg)
            classifier.build_model(1)
            model_cfg = json.loads((folder / "mlm/config.json").read_text())
            model_cfg["molecular_tokenizer_identity"]["vocab_sha256"] = "wrong"
            (folder / "mlm/config.json").write_text(json.dumps(model_cfg))
            with self.assertRaises(ValueError):
                classifier.build_model(1)


class FrozenSelectionTests(unittest.TestCase):
    def test_mlm_selection_keeps_frozen_rows_and_checks_source_fingerprint(self):
        import os
        from scripts.prepare_external_tsdy_training import mlm_inputs
        from utils.cache_utils import record_cache, file_digest

        with tempfile.TemporaryDirectory() as td:
            previous = Path.cwd()
            try:
                os.chdir(td)
                root = Path("data/shared_study_2m_v2")
                (root / "miner").mkdir(parents=True)
                (root / "base_TSDY_mlm").mkdir()
                source = dict(dataset="fixture", revision="pinned", split="train", smiles_column="SMILES")
                ti = root / "miner/subset_indices.json"
                ti.write_text(json.dumps(dict(indices=[0, 1], total_source=8, source_fingerprint="frozen")))
                ti_meta = dict(kind="tokenizer_indices", corpus_size=2, seed=42, source=source)
                record_cache(ti, ti_meta)
                corpus = root / "miner/corpus.txt"
                corpus.write_text("C\nO\n")
                record_cache(corpus, dict(indices=file_digest(ti), selection=ti_meta))
                mi = root / "base_TSDY_mlm/subset_indices.json"
                mi.write_text(json.dumps(dict(train=[2, 3], validation=[4], test=[5])))
                record_cache(
                    mi,
                    dict(
                        kind="mlm_indices",
                        seed=42,
                        source=source,
                        tokenizer_indices=file_digest(ti),
                        train_size=2,
                        val_size=1,
                        test_size=1,
                    ),
                )
                preflight = Path("preflight")
                preflight.mkdir()
                (preflight / "manifest.json").write_text(json.dumps(dict(input_sha256=file_digest(corpus))))

                class Selected:
                    def __init__(self, ids):
                        self.ids = ids

                    def __getitem__(self, part):
                        return {"SMILES": ["C" * (i + 1) for i in self.ids[part]]}

                class Raw:
                    _fingerprint = "frozen"

                    def __len__(self):
                        return 8

                    def select(self, ids):
                        return Selected(ids)

                with patch("datasets.load_dataset", return_value=Raw()) as load:
                    groups, provider, identity = mlm_inputs(preflight, 0)
                    self.assertEqual(groups, dict(train=[2, 3], validation=[4], test=[5]))
                    self.assertEqual(
                        provider("train", 0, 2), [dict(source_row=2, text="CCC"), dict(source_row=3, text="CCCC")]
                    )
                    load.assert_called_once_with("fixture", revision="pinned", split="train")
                raw = Raw()
                raw._fingerprint = "changed"
                with patch("datasets.load_dataset", return_value=raw), self.assertRaises(ValueError):
                    mlm_inputs(preflight, 0)
                mi.write_text(json.dumps(dict(train=[0, 3], validation=[4], test=[5])))
                record_cache(
                    mi,
                    dict(
                        kind="mlm_indices",
                        seed=42,
                        source=source,
                        tokenizer_indices=file_digest(ti),
                        train_size=2,
                        val_size=1,
                        test_size=1,
                    ),
                )
                with self.assertRaises(ValueError):
                    mlm_inputs(preflight, 0)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
