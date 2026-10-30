import unittest
from scripts.evaluate_intrinsic import percentile, summarize, load, convert, conditions, decode_frag_words
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
        item = next(x for x in conditions() if x["name"] == "CA_SMILY_SMIRK_GPE")
        tok = load(item)
        s = "CC(C)CCCCCCCCCCCCCCNc1ccc(C(=O)NCC(=O)[O-])cc1[N+](=O)[O-]"
        words, _, _ = convert(s, item, tok, None)
        self.assertEqual(words.native_ids, tok(s, add_special_tokens=False)["input_ids"])
        self.assertTrue(all(isinstance(i, int) for i in words.native_ids))

    def test_frag_separator_roundtrip(self):
        self.assertTrue(molecules_equal("CCOc1ccccc1", decode_frag_words(["C", "C", "O", "<0>", "c1ccccc1"])))

    def test_greedtok_native_ids(self):
        item = next(x for x in conditions() if x["name"] == "CA_SMILY_SPCATT")
        tok = load(item)
        words, decoder, _ = convert("CCO", item, tok, None)
        expected = tok("CCO", add_special_tokens=False)["input_ids"]
        if expected and isinstance(expected[0], list):
            expected = expected[0]
        self.assertEqual(words.native_ids, expected)
        self.assertIsNone(decoder)

    def test_ambiguous_tsdy_not_claimed_reconstructable(self):
        item = next(x for x in conditions() if x["name"] == "shared_base_TSDY")
        tok = load(item)
        words, decoder, _ = convert("CCO", item, tok, None)
        self.assertIsNone(decoder)
        self.assertFalse(hasattr(words, "raw_decoded"))


if __name__ == "__main__":
    unittest.main()
