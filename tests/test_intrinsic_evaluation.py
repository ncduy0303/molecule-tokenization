import json
import tempfile
import unittest
from pathlib import Path
from scripts.evaluate_intrinsic import percentile, summarize, convert, decode_frag_words
from utils.external_tsdy import molecules_equal


class IntrinsicTests(unittest.TestCase):
    def test_empty_denominators(self):
        a = summarize([dict(ok=False)], 5000)
        self.assertEqual(a["failed"], 1)
        self.assertIsNone(a["mean_length"])
        self.assertIsNone(a["unknown_occurrence_rate"])

    def test_percentile(self):
        self.assertEqual(percentile([1, 3], 0.5), 2)

    def test_native_smirk_ids(self):
        from smirk import SmirkTokenizerFast, train_gpe

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "smiles.txt"
            corpus.write_text("CCCCCCCCO\n" * 8)
            tok = train_gpe(files=[str(corpus)], ref=SmirkTokenizerFast(), vocab_size=180, min_frequency=1)
        item = {"representation": "smiles"}
        s = "CC(C)CCCCCCCCCCCCCCNc1ccc(C(=O)NCC(=O)[O-])cc1[N+](=O)[O-]"
        words, _, _ = convert(s, item, tok, None)
        self.assertEqual(words.native_ids, tok(s, add_special_tokens=False)["input_ids"])
        self.assertTrue(all(isinstance(i, int) for i in words.native_ids))

    def test_frag_separator_roundtrip(self):
        self.assertTrue(molecules_equal("CCOc1ccccc1", decode_frag_words(["C", "C", "O", "<0>", "c1ccccc1"])))

    def test_greedtok_native_ids(self):
        from pcatt.hf.greedtok import GreedTok

        tok = GreedTok(ranked_tokens=["C", "O", "CC"])
        item = {"representation": "smiles"}
        words, decoder, _ = convert("CCO", item, tok, None)
        expected = tok("CCO", add_special_tokens=False)["input_ids"]
        if expected and isinstance(expected[0], list):
            expected = expected[0]
        self.assertEqual(words.native_ids, expected)
        self.assertIsNone(decoder)

    def test_ambiguous_tsdy_not_claimed_reconstructable(self):
        from utils.tsmiles_tokenizer import TSMILESTokenizer

        item = {"representation": "shared", "variant": "TSDY"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            path.write_text(json.dumps({"[UNK]": 0, "&": 1, "^": 2}))
            tok = TSMILESTokenizer(str(path), preserve_components=True)
        words, decoder, _ = convert("CCO", item, tok, None)
        self.assertIsNone(decoder)
        self.assertFalse(hasattr(words, "raw_decoded"))


if __name__ == "__main__":
    unittest.main()
