"""Regression checks for RNG state before Trainer/model initialization."""

import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import RobertaConfig, RobertaForSequenceClassification

from experiments.tokenizer_benchmark import TokenizerBenchmarkExperiment
from experiments.downstream_classification import DownstreamClassificationExperiment
from algorithms.tokenizer_benchmark.algo import MoleculeMLMAlgo


def config(seed):
    return OmegaConf.create(
        {
            "debug": False,
            "dataset": {"seed": 42},
            "experiment": {"training": {"seed": seed, "data_seed": "${.seed}"}},
            "algorithm": {
                "_name": "bpe_roberta",
                "model": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "max_position_embeddings": 32,
                },
            },
        }
    )


class InitializationSeedTests(unittest.TestCase):
    def test_mlm_initial_weights_and_rngs_follow_training_seed(self):
        def build(seed):
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            exp = TokenizerBenchmarkExperiment(config(seed), Path("/tmp"))
            with patch.object(MoleculeMLMAlgo, "_build_tokenizer", return_value=list(range(10))):
                model = exp._build_algo().model
            return torch.cat([p.detach().flatten() for p in model.parameters()]), (random.random(), np.random.rand())

        a, ar = build(42)
        b, br = build(42)
        c, cr = build(43)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(ar, br)
        self.assertFalse(torch.equal(a, c))
        self.assertNotEqual(ar, cr)

    def test_classifier_initialization_ignores_cache_rng_consumption(self):
        class Captured(Exception):
            pass

        tiny = RobertaConfig(
            vocab_size=10, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_labels=1
        )

        def build(seed, draws, method):
            exp = DownstreamClassificationExperiment(config(seed), Path("/tmp"))
            weights = []

            def load(*args):
                torch.rand(draws)
                np.random.rand(draws)
                return {"train": [{"labels": [1.0]}]}

            def model(num_labels):
                m = RobertaForSequenceClassification(tiny)
                weights.append(m.classifier.dense.weight.detach().clone())
                raise Captured

            exp.algo = SimpleNamespace(tokenizer=None, build_model=model)
            with patch("experiments.downstream_classification.load_molnet", side_effect=load):
                with self.assertRaises(Captured):
                    getattr(exp, method)()
            return weights[0]

        for method in ("training", "evaluation"):
            with self.subTest(method=method):
                a = build(42, 1, method)
                self.assertTrue(torch.equal(a, build(42, 100, method)))
                self.assertFalse(torch.equal(a, build(43, 100, method)))

    def test_sampler_seed_tracks_training_seed_without_changing_split_seed(self):
        cfg = config(42)
        cfg.experiment.training.seed = 43
        self.assertEqual(cfg.experiment.training.data_seed, 43)
        self.assertEqual(cfg.dataset.seed, 42)
        cfg.experiment.training.data_seed = 99
        self.assertEqual(cfg.experiment.training.data_seed, 99)


if __name__ == "__main__":
    unittest.main()
