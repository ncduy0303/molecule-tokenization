"""Offline chemistry, mining, and dataset integration tests for the shared study."""

from collections import Counter
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import chemicalgof
from datasets import Dataset
from hydra import compose, initialize_config_dir
import networkx as nx
from omegaconf import OmegaConf
from rdkit import Chem

from utils.shared_fragmentation import (
    EXOCYCLIC_SMARTS,
    binary_tree_tokens,
    decorated_fragments,
    decode_tsid,
    encode_shared,
    partition_molecule,
    serialize_fragsmiles,
    serialize_tsmiles,
)
from utils.shared_ps import MergeState, SharedPSModel, load_model, train_shared_ps
from utils.cache_utils import CacheMismatchError, representation_identity, validate_tokenizer_representation

ROOT = Path(__file__).resolve().parents[1]
MOLECULES = [
    "CCO",
    "CC(=O)O",
    "N#N",
    "O=C=O",
    "CC/C=C/CC",
    "CC/C=C\\CC",
    "c1ccccc1CCO",
    "c1ccc2ccccc2c1",
    "C1CCC2(CC1)CCCC2",
    "C1CC2CCC1C2",
    "N[C@@H](C)C(=O)O",
    "C[C@H](O)C(=O)O",
    "C[N+](=O)[O-]",
    "[13CH3]CO",
    "[Na+].[Cl-]",
    "CC(C)(C)CO",
    "[NH4+].[O-]C=O",
]


class PartitionTests(unittest.TestCase):
    def test_explicit_hydrogen_fragment_mapping_preserves_source_atoms(self):
        from chemicalgof.reduce import Decompositer
        from utils.shared_fragmentation import _fragment_strings, component_partitions
        from utils.fragsmiles_utils import split_fragsmiles_tokens

        smiles = "F[B-](F)(F)F.[H][H-][B-]1([H-][H])[n+]2ccccc2-c2cccc[n+]21"
        partition = partition_molecule(smiles)
        for part in component_partitions(partition):
            pairs = [(part.mol.GetBondWithIdx(i).GetBeginAtomIdx(), part.mol.GetBondWithIdx(i).GetEndAtomIdx()) for i in part.cut_bonds]
            d = Decompositer()
            fragments = d.fragment(part.mol, pairs)
            strings = _fragment_strings(d, fragments)
            mapped = [source for mapping in d.mapsFrag2Mol for source in mapping.values()]
            self.assertEqual(sorted(mapped), list(range(part.mol.GetNumAtoms())))
            for text, mapping in zip(strings, d.mapsFrag2Mol):
                mol = Chem.MolFromSmiles(text)
                for local, source in mapping.items():
                    a, b = mol.GetAtomWithIdx(local), part.mol.GetAtomWithIdx(source)
                    self.assertEqual((a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge()), (b.GetAtomicNum(), b.GetIsotope(), b.GetFormalCharge()))
        for model in (None, SharedPSModel(("CC",), {})):
            encoded = encode_shared(smiles, "fragsmiles", model)
            self.assertIn("[HH]", split_fragsmiles_tokens(encoded, True))

    def test_encoding_error_identifies_input(self):
        with self.assertRaisesRegex(ValueError, "Shared fragsmiles encoding failed for SMILES 'bad-input'"):
            encode_shared("bad-input", "fragsmiles")

    def test_hiv_metal_complex_does_not_refragment_into_cycle(self):
        from utils.fragsmiles_utils import split_fragsmiles_tokens

        smiles = "C1C[N+]23CCC[N+]4(C1)CC[S-]1[Fe+2]24([SH+]CC3)[S-]2CC[N+]34CCC[N+]5(CCC3)CC[SH+][Fe+2]1254"
        for model in (None, SharedPSModel(("CC",), {})):
            partition = partition_molecule(smiles, model)
            self.assertEqual(partition.cut_bonds, ())
            encoded = serialize_fragsmiles(partition)
            # A single fragment, not connection markers accidentally parsed
            # from RDKit's dative arrows, and the source graph is retained.
            self.assertEqual(split_fragsmiles_tokens(encoded, True), [encoded])
            self.assertEqual(encoded, Chem.MolToSmiles(partition.mol))

    def test_tsis_preserves_fragments_and_reconstructs_without_tree_placeholders(self):
        model = train_shared_ps(MOLECULES, num_merges=12, min_frequency=1)
        for smiles in MOLECULES:
            for merges in (None, model):
                p = partition_molecule(smiles, merges)
                tsid = serialize_tsmiles(p, "TSID")
                tsis = serialize_tsmiles(p, "TSIS")
                self.assertNotIn("&", tsis)
                self.assertEqual(decode_tsid(tsis), Chem.CanonSmiles(smiles))
                for left, right in zip(tsid.split("."), tsis.split(".")):
                    self.assertEqual(Counter(filter(None, re.split(r"[&^]", left))), Counter(right.split("^")))
        p = partition_molecule("CC(C)CO")
        fragments = decorated_fragments(p)
        self.assertEqual(serialize_tsmiles(p, "TSIS"), "^".join(fragments[i] for i in [0, 1, 2, 3, 4]))

    def test_pubchem_cage_with_incomplete_ring_membership(self):
        # PubChem source row 2832359 in the pinned dataset. The original SMARTS
        # selects an internal cage edge on RDKit 2025.09.4.
        smiles = "CC1C2C3CC45C6C7C38CC1(C)C8(C)C7(C)C4(C)C1(C)C6C3C4CC26CC35C1(C)C4(C)C6C"
        base = partition_molecule(smiles)
        graph = nx.Graph((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in base.mol.GetBonds())
        for idx in base.cut_bonds:
            bond = base.mol.GetBondWithIdx(idx)
            reduced = graph.copy()
            reduced.remove_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            self.assertFalse(nx.has_path(reduced, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        model = train_shared_ps([smiles, "CCO"], num_merges=3, min_frequency=1)
        for merge_model in (None, model):
            p = partition_molecule(smiles, merge_model)
            p.validate()
            tsid = serialize_tsmiles(p, "TSID")
            self.assertEqual(decode_tsid(tsid), Chem.CanonSmiles(smiles))
            self.assertEqual(Chem.CanonSmiles(chemicalgof.decode(serialize_fragsmiles(p))), Chem.CanonSmiles(smiles))
            self.assertEqual(serialize_tsmiles(p, "TSDY"), re.sub(r"\[\d+\*\]", "*", tsid))

    def test_exact_chemicalgof_partition_and_baseline_output(self):
        from chemicalgof.reduce import Decompositer

        self.assertEqual(EXOCYCLIC_SMARTS, Decompositer.SINGLEXOCYCLICPATT)
        for smiles in MOLECULES:
            with self.subTest(smiles=smiles):
                partition = partition_molecule(smiles)
                partition.validate()
                self.assertTrue(nx.is_forest(partition.graph()))
                expected = ";".join(chemicalgof.encode(s, canonical=True) for s in Chem.CanonSmiles(smiles).split("."))
                self.assertEqual(serialize_fragsmiles(partition), expected)
                expected_cuts = partition.mol.GetSubstructMatches(Chem.MolFromSmarts(EXOCYCLIC_SMARTS))
                self.assertEqual(
                    set(partition.cut_bonds),
                    {partition.mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in expected_cuts},
                )

    def test_tsid_exact_reconstruction_and_tsdy_only_removes_dummy_ids(self):
        model = train_shared_ps(MOLECULES, num_merges=12, min_frequency=1)
        for model_option in (None, model):
            for smiles in MOLECULES:
                with self.subTest(smiles=smiles, merged=model_option is not None):
                    p = partition_molecule(smiles, model_option)
                    tsid = serialize_tsmiles(p, "TSID")
                    self.assertEqual(decode_tsid(tsid), Chem.CanonSmiles(smiles))
                    self.assertEqual(serialize_tsmiles(p, "TSDY"), re.sub(r"\[\d+\*\]", "*", tsid))

    def test_non_single_bonds_and_rings_cannot_be_cut_even_when_unseen(self):
        empty = SharedPSModel((), {})
        for smiles in ("N#N", "O=C=O", "C1CCCCC1", "C1CCC2(CC1)CCCC2"):
            p = partition_molecule(smiles, empty)
            self.assertEqual(p.cut_bonds, ())
            self.assertEqual(len(p.groups), 1)
            self.assertEqual(Chem.CanonSmiles(chemicalgof.decode(serialize_fragsmiles(p))), Chem.CanonSmiles(smiles))

    def test_canonical_input_order_and_unsupported_inputs(self):
        for smiles in ("CC(O)C", "OC(C)C"):
            self.assertEqual(encode_shared(smiles, "TSID"), encode_shared("CC(C)O", "TSID"))
        for smiles in ("", "not-a-molecule", "*CC", "[CH3:1]CO"):
            with self.assertRaises(ValueError):
                partition_molecule(smiles)
        with self.assertRaisesRegex(ValueError, "TSID, TSDY and TSIS"):
            encode_shared("CC", "TSSA")

    def test_binary_tree_matches_upstream_without_using_its_fragmenter(self):
        from t_smiles.dataset.graph.cnjt_mol import CNJTMolTree
        from utils.tsmiles_utils import _CTOKEN

        for smiles in ("CCO", "CC(C)(C)CO", "CC", "c1ccccc1", "CC(C)C(C)C"):
            p = partition_molecule(smiles)
            graph, fragments = p.graph(), decorated_fragments(p)
            upstream = CNJTMolTree(None, ctoken=_CTOKEN)
            upstream.graph_nx = graph.copy()
            upstream.n_nodes = len(graph)
            upstream.bfs_idx = list(nx.bfs_tree(graph, 0, sort_neighbors=sorted))
            upstream.neighbor_map = [sorted(graph.neighbors(n)) for n in graph]
            for n in graph:
                upstream.graph_nx.nodes[n].update(smile=fragments[n], smarts=fragments[n])
            upstream.build_relation(upstream.bfs_idx)
            root = upstream.create_bfs_binary_tree_ex()
            expected = CNJTMolTree.get_bfs_ex(_CTOKEN, root, extra_dummy=True)[2]
            self.assertEqual(binary_tree_tokens(graph, fragments), expected)


class MiningTests(unittest.TestCase):
    def test_incremental_mining_matches_full_recount_and_ordered_replay(self):
        corpus = ["CCCC", "CCO", "CCCO", "CCN", "CC(C)O", "CCCCCC", "CCO", "c1ccccc1CCO"]
        states = [MergeState(partition_molecule(s)) for s in corpus]
        rules = []
        for _ in range(20):
            counts = Counter()
            for state in states:
                counts.update(state.counts())
            if not counts:
                break
            key = min(counts, key=lambda s: (-counts[s], s))
            rules.append(key)
            for state in states:
                state.merge(key)
        model = train_shared_ps(corpus, num_merges=20, min_frequency=1)
        self.assertEqual(model.rules, tuple(rules))
        for smiles, state in zip(corpus, states):
            self.assertEqual(partition_molecule(smiles, model).groups, state.partition().groups)

    def test_cached_counts_keep_multiplicity_and_merges_only_coarsen(self):
        state = MergeState(partition_molecule("CCCC"))
        self.assertEqual(state.counts()["CC"], 3)
        self.assertEqual(state.counts()["CC"], 3)
        self.assertFalse(state.merge("NO"))
        self.assertEqual(state.counts()["CC"], 3)
        model = train_shared_ps(MOLECULES, num_merges=10, min_frequency=1)
        for smiles in MOLECULES:
            before, after = partition_molecule(smiles), partition_molecule(smiles, model)
            self.assertLessEqual(set(after.cut_bonds), set(before.cut_bonds))
            for group in before.groups:
                self.assertTrue(any(set(group) <= set(merged) for merged in after.groups))

    def test_model_roundtrip_rejects_legacy_and_protects_output(self):
        model = train_shared_ps(["CCO", "CCO", "CCN"], num_merges=5, min_frequency=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merges.json"
            model.save(path)
            self.assertEqual(load_model(path).rules, model.rules)
            with self.assertRaises(FileExistsError):
                model.save(path)
            path.write_text('{"kekulize": true}')
            with self.assertRaisesRegex(ValueError, "legacy"):
                load_model(path)


class SharedPipelineTests(unittest.TestCase):
    def test_component_delimiters_survive_tokenizer_save_and_reload(self):
        from utils.fragsmiles_utils import split_fragsmiles_tokens
        from utils.fragsmiles_tokenizer import FragSMILESTokenizer
        from utils.tsmiles_tokenizer import TSMILESTokenizer, split_tsmiles_tokens

        for representation, cls, splitter in [
            ("fragsmiles", FragSMILESTokenizer, split_fragsmiles_tokens),
            ("TSID", TSMILESTokenizer, split_tsmiles_tokens),
            ("TSIS", TSMILESTokenizer, split_tsmiles_tokens),
        ]:
            text = encode_shared("[Na+].[Cl-]", representation)
            tokens = splitter(text, preserve_components=True)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "vocab.json"
                vocab = list(dict.fromkeys(["[UNK]"] + tokens))
                path.write_text(json.dumps({token: i for i, token in enumerate(vocab)}))
                tokenizer = cls(str(path), preserve_components=True)
                tokenizer.save_pretrained(tmp)
                restored = cls.from_pretrained(tmp)
                self.assertEqual(restored.tokenize(text), tokens)
                if representation == "fragsmiles":
                    self.assertEqual(
                        Chem.CanonSmiles(restored.convert_tokens_to_string(tokens)), Chem.CanonSmiles("[Na+].[Cl-]")
                    )
                else:
                    self.assertEqual(restored.convert_tokens_to_string(tokens), text)

    def test_training_cache_provenance_and_all_dataset_stages(self):
        from experiments.tokenizer_training import TokenizerTrainingExperiment
        from datamodules.molecule_datasets.pubchem10m_mlm_train import load_pubchem10m_mlm_train
        from datamodules.molecule_datasets.molnet import load_molnet
        from utils.fragsmiles_tokenizer import FragSMILESTokenizer
        from utils.tsmiles_tokenizer import TSMILESTokenizer
        import pandas as pd

        # No downloads, real corpus selection, encoding, vocab saving/loading,
        # and MLM/downstream tokenization. Neural training is covered separately.
        source = Dataset.from_dict({"SMILES": ["C" * n + "O" for n in range(1, 31)]})
        with tempfile.TemporaryDirectory() as tmp, patch("datasets.load_dataset", return_value=source):
            root = Path(tmp)
            with initialize_config_dir(config_dir=str(ROOT / "configurations"), version_base=None):
                base = compose(
                    config_name="config",
                    overrides=[
                        "algorithm=train_shared_ps",
                        "dataset.corpus_size=8",
                        "dataset.hf_revision=fixed",
                        "algorithm.tokenizer.num_merges=4",
                        "algorithm.tokenizer.min_frequency=1",
                        f'dataset.data_dir={root / "corpus"}',
                        "+name=shared_test",
                    ],
                )
            base.dataset.num_proc = None
            OmegaConf.update(base, "algorithm._name", "train_shared_ps", force_add=True)
            output = root / "miner"
            output.mkdir()
            TokenizerTrainingExperiment(base, output).training()
            merge_path = output / "ps_merges.json"

            for use_merges in (False, True):
                for representation in ("fragsmiles", "TSID", "TSDY", "TSIS"):
                    family = "fragsmiles" if representation == "fragsmiles" else "tsmiles"
                    folder = root / f"{representation}_{use_merges}"
                    with initialize_config_dir(config_dir=str(ROOT / "configurations"), version_base=None):
                        cfg = compose(
                            config_name="config",
                            overrides=[
                                f"algorithm=train_{family}",
                                "dataset.corpus_size=8",
                                "dataset.hf_revision=fixed",
                                f'dataset.data_dir={folder / "corpus"}',
                                "dataset.fragmentation=exocyclic",
                                f"dataset.use_{family}=true",
                                "+name=shared_test",
                            ],
                        )
                    cfg.dataset.num_proc = None
                    OmegaConf.update(cfg, "algorithm._name", f"train_{family}", force_add=True)
                    if family == "tsmiles":
                        cfg.dataset.tsmiles_variant = representation
                    if use_merges:
                        cfg.dataset.ps_merges_path = str(merge_path)
                    folder.mkdir()
                    TokenizerTrainingExperiment(cfg, folder).training()
                    cls = FragSMILESTokenizer if family == "fragsmiles" else TSMILESTokenizer
                    tokenizer = cls.from_pretrained(folder / "tokenizer")
                    validate_tokenizer_representation(cfg.dataset, tokenizer)
                    with initialize_config_dir(config_dir=str(ROOT / "configurations"), version_base=None):
                        mlm = compose(config_name="config", overrides=["dataset=pubchem10m_mlm_train"]).dataset
                    for key in (
                        "use_fragsmiles",
                        "use_tsmiles",
                        "tsmiles_variant",
                        "fragmentation",
                        "ps_merges_path",
                        "hf_revision",
                    ):
                        mlm[key] = cfg.dataset[key]
                    mlm.data_dir = str(folder / "mlm")
                    mlm.tokenizer_indices_path = str(folder / "corpus/subset_indices.json")
                    mlm.train_size, mlm.val_size, mlm.test_size = 6, 2, 2
                    mlm.num_proc = 2 if use_merges and representation == "TSID" else None
                    result = load_pubchem10m_mlm_train(mlm, tokenizer)
                    self.assertEqual(len(result["train"]), 6)
                    with patch("datasets.load_dataset", side_effect=AssertionError("must reuse cache")):
                        load_pubchem10m_mlm_train(mlm, tokenizer)
                    # Test both downstream dataset names with synthetic labels/splits.
                    for task, target in [("bbbp", "p_np"), ("hiv", "HIV_active")]:
                        molnet = OmegaConf.create(OmegaConf.to_container(cfg.dataset))
                        molnet.molnet_name = task
                        molnet.max_length = 32
                        molnet.data_dir = str(folder / task)
                        Path(molnet.data_dir).mkdir()
                        rows = pd.DataFrame(
                            {"smiles": ["CCO", "CC", "CO", "CN", "CCC", "CCCC"], target: [0, 1, 0, 1, 0, 1]}
                        )
                        with (
                            patch("datamodules.molecule_datasets.molnet._download_csv", return_value=rows),
                            patch(
                                "datamodules.molecule_datasets.molnet._deepchem_split",
                                return_value=([0, 1], [2, 3], [4, 5]),
                            ),
                        ):
                            self.assertEqual(len(load_molnet(molnet, tokenizer)["test"]), 2)
                    wrong = OmegaConf.create(OmegaConf.to_container(cfg.dataset))
                    wrong.fragmentation = None
                    wrong.ps_merges_path = None
                    with self.assertRaisesRegex(ValueError, "Tokenizer representation"):
                        validate_tokenizer_representation(wrong, tokenizer)

    def test_cache_identity_and_selection_are_sensitive_to_merge_contents(self):
        from utils.cache_utils import validate_ps_selection, file_digest, source_identity

        with tempfile.TemporaryDirectory() as tmp:
            path, indices = Path(tmp) / "merges.json", Path(tmp) / "indices.json"
            indices.write_text('{"indices": [1, 2]}')
            cfg = OmegaConf.create(
                dict(
                    use_tsmiles=True,
                    tsmiles_variant="TSID",
                    fragmentation="exocyclic",
                    ps_merges_path=str(path),
                    hf_dataset="fixture",
                    hf_revision="fixed",
                    smiles_column="SMILES",
                )
            )
            model = SharedPSModel(
                ("CC",), {"tokenizer_indices_sha256": file_digest(indices), "source": source_identity(cfg)}
            )
            model.save(path)
            before = representation_identity(cfg)
            validate_ps_selection(cfg, indices)
            payload = json.loads(path.read_text())
            payload["rules"].append("CCC")
            path.write_text(json.dumps(payload))
            self.assertNotEqual(before, representation_identity(cfg))
            indices.write_text('{"indices": [3, 4]}')
            with self.assertRaisesRegex(ValueError, "same source"):
                validate_ps_selection(cfg, indices)


if __name__ == "__main__":
    unittest.main()
