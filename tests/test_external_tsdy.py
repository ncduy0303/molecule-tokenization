import json
from pathlib import Path
import unittest
from rdkit import Chem
from utils.external_tsdy import (
    encode,
    decode,
    model_tokens,
    decode_tokens,
    verbose_model_tokens,
    COMPACT_GRAMMAR,
    molecules_equal,
    model_tokens_v2,
    parse_core,
)

CASES = [
    "[CH3]",
    "[CH]c1ccccc1",
    "CCC(CC)(CC)O[Si]CCO",
    "C",
    "CC(C)C",
    "Cc1ccccc1",
    "Cc1ccccc1C",
    "Cc1ccc(C)cc1",
    "Oc1cccc(O)c1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "[Na+].[Cl-]",
    "C[N+](C)(C)C",
    "[2H]C(O)C",
    "[13CH3]CO",
    "[H][H]",
    "C[C@H](O)C(=O)O",
    "C[C@@H](O)C(=O)O",
    "N[C@@H](Cc1ccccc1)C(=O)O",
    "F/C=C/F",
    "F/C=C\\F",
    "C/C=C/C=C\\C",
    "C1CC[C@H](C)C1",
    "O=C([O-])CC[NH3+]",
]


class ExternalTSDYTests(unittest.TestCase):
    def test_full_corpus_capping_regressions(self):
        cases = json.loads((Path(__file__).parent / "fixtures/external_tsdy_capping.json").read_text())
        for case in cases:
            with self.subTest(row=case["row"]):
                smi = case["smiles"]
                text = encode(smi)
                self.assertTrue(molecules_equal(smi, decode(text)))
                self.assertTrue(molecules_equal(smi, decode_tokens(model_tokens(text))))
                # Cores must remain canonical under randomized source ordering.
                randomized = Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)
                self.assertEqual(model_tokens(text), model_tokens(encode(randomized)))

    def test_core_bond_state_preserved_and_invalid_aromaticity_rejected(self):
        core = parse_core("O=[IH]=O")
        self.assertTrue(all(b.GetBondType() == Chem.BondType.DOUBLE for b in core.GetBonds()))
        self.assertTrue(all(a.GetFormalCharge() == 0 for a in core.GetAtoms()))
        localized = parse_core("C1=CC=CC=C1")
        self.assertFalse(any(a.GetIsAromatic() for a in localized.GetAtoms()))
        self.assertTrue(all(a.GetIsAromatic() for a in parse_core("c1ccccc1").GetAtoms()))
        sulfur_core = parse_core("c1ccsc1")
        self.assertTrue(all(a.GetNumRadicalElectrons() == 0 for a in sulfur_core.GetAtoms()))
        self.assertIn("c1ccsc1", json.loads(encode("Cc1ccsc1"))["components"][0]["cores"])
        with self.assertRaises(ValueError):
            parse_core("c1cccc1")

    def test_roundtrip_and_randomized_inputs(self):
        for smi in CASES:
            mol = Chem.MolFromSmiles(smi)
            expected = Chem.MolToSmiles(mol)
            for input_text in [smi] + [Chem.MolToSmiles(mol, doRandom=True) for _ in range(3)]:
                with self.subTest(smiles=input_text):
                    doc = encode(input_text)
                    self.assertEqual(decode(doc), expected)
                    self.assertEqual(decode_tokens(model_tokens(doc)), expected)
                    self.assertEqual(doc, encode(smi))

    def test_core_shared_across_attachment_positions_and_counts(self):
        for smi in ["Cc1ccccc1", "Cc1ccccc1C", "Cc1ccc(C)cc1", "Cc1cc(C)cc(C)c1"]:
            doc = json.loads(encode(smi))
            self.assertIn("c1ccccc1", doc["components"][0]["cores"])
            self.assertTrue(all("*" not in s for s in doc["components"][0]["cores"]))

    def test_bad_connections_fail(self):
        doc = json.loads(encode("CCO"))
        doc["components"][0]["links"][0][0][1] = 999
        with self.assertRaises(ValueError):
            decode(json.dumps(doc))
        doc = json.loads(encode("CCO"))
        doc["components"][0]["links"].append(doc["components"][0]["links"][0])
        with self.assertRaises(ValueError):
            decode(json.dumps(doc))
        with self.assertRaises(ValueError):
            decode_tokens(["<EXT_TSDY_V1>", "[UNK]"])

    def test_unsupported_chemistry_rejected(self):
        for s in ["N->[Cu+2]", "[*:1]C"]:
            with self.assertRaises(ValueError):
                encode(s)


class CompactSyntaxTests(unittest.TestCase):
    def test_compact_and_verbose_equivalent_and_shorter(self):
        for smi in CASES + ["CC1" + "C" * 149 + "1"]:
            doc = encode(smi)
            compact, verbose = model_tokens(doc), verbose_model_tokens(doc)
            self.assertEqual(decode_tokens(compact), decode_tokens(verbose))
            self.assertEqual(decode_tokens(compact), decode_tokens(model_tokens_v2(doc)))
            self.assertLess(len(compact), len(verbose))
            cores = {s for c in json.loads(doc)["components"] for s in c["cores"]}
            self.assertTrue(set(compact) <= COMPACT_GRAMMAR | cores)
        self.assertIn("<BIG>", model_tokens(encode("CC1" + "C" * 149 + "1")))

    def test_truncated_and_invalid_records_rejected(self):
        tokens = model_tokens(encode("CCO"))
        for bad in [tokens + ["<F>"], ["<DOT>"], ["<EXT_TSDY_V2>"], ["<EXT_TSDY_V2>", "<COMPONENT>", "<LINK>", "<N0>"]]:
            with self.assertRaises(ValueError):
                decode_tokens(bad)


class MolecularEqualityTests(unittest.TestCase):
    def test_stereo_and_chemical_changes_not_accepted(self):
        for a, b in [
            ("C[C@H](O)F", "C[C@@H](O)F"),
            ("F/C=C/F", "F/C=C\\F"),
            ("[13CH4]", "C"),
            ("[CH3]", "C"),
            ("[NH4+]", "N"),
            ("Cc1ccccc1C", "Cc1ccc(C)cc1"),
        ]:
            self.assertFalse(molecules_equal(a, b))

    def test_symmetric_macrocycle_graph_roundtrip(self):
        smi = "CCCCCC1c2cc3c4c5c2OCOc2c1cc1c6c2CNC(=O)CCCCCOc2ccccc2-c2c7nc(c8c9ccc([nH]9)c(c9nc(c(c%10ccc2[nH]%10)-c2ccccc2OCCCCCC(=O)NC5)C=C9)-c2ccccc2OCCCCCC(=O)NCc2c(c(cc5c2OCOc2c(cc(c(c2CNC(=O)CCCCCOc2ccccc2-8)OCO6)C1CCCCC)C5CCCCC)C3CCCCC)OCO4)C=C7"
        self.assertTrue(molecules_equal(smi, decode_tokens(model_tokens(encode(smi)))))


class ImplicitTreeTests(unittest.TestCase):
    def test_tree_and_site_inference(self):
        self.assertEqual(model_tokens(encode("CCO")), ["C", "C", "<NULL>", "O"])
        self.assertEqual(model_tokens(encode("Cc1ccccc1")), ["C", "c1ccccc1", "<N2>"])
        self.assertEqual(model_tokens(encode("[Na+].[Cl-]")), ["[Cl-]", "<DOT>", "[Na+]"])
        # Distinguish a chain from branching despite identical core words.
        self.assertNotEqual(model_tokens(encode("CCCC")), model_tokens(encode("CC(C)C")))
        self.assertNotEqual(model_tokens(encode("Cc1ccccc1C")), model_tokens(encode("Cc1ccc(C)cc1")))

    def test_compact_stereo_and_no_silent_truncation(self):
        from utils.external_tsdy import checked_training_tokens

        a = model_tokens(encode("C[C@H](O)C(=O)O"))
        b = model_tokens(encode("C[C@@H](O)C(=O)O"))
        self.assertNotEqual(a, b)
        self.assertTrue(any(t in ("<ACW>", "<ACCW>") for t in a))
        self.assertEqual(checked_training_tokens("CCO"), model_tokens(encode("CCO")))
        with self.assertRaises(ValueError):
            checked_training_tokens("CCO", max_content_tokens=3)
        for tokens in [["C", "c1ccccc1"], ["C", "<ACW>", "<N99>"], ["C", "<DOT>"], ["C", "<N0>"]]:
            with self.assertRaises(ValueError):
                decode_tokens(tokens)
