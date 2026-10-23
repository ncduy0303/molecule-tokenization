"""Exercise real Trainer checkpoints through both project training methods on CPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from hydra import compose, initialize_config_dir
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from utils.ckpt_utils import resolve_resume_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def cpu_arguments(*args, **kwargs):
    return TrainingArguments(*args, **dict(kwargs, use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False))


class TrainerResumeTests(unittest.TestCase):
    def test_both_experiments_continue_saved_training_state(self):
        from experiments.tokenizer_benchmark import TokenizerBenchmarkExperiment
        from experiments.downstream_classification import DownstreamClassificationExperiment

        backend = Tokenizer(
            WordLevel({"[BOS]": 0, "[PAD]": 1, "[EOS]": 2, "[UNK]": 3, "[MASK]": 4, "C": 5, "O": 6}, unk_token="[UNK]")
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="[BOS]",
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            mask_token="[MASK]",
        )
        model_cfg = RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=64,
            pad_token_id=1,
            num_labels=1,
            problem_type="multi_label_classification",
        )
        for stage, cls, algo, dataset_name in [
            ("tokenizer_benchmark", TokenizerBenchmarkExperiment, "bpe_roberta", "pubchem10m_mlm_train"),
            ("downstream_classification", DownstreamClassificationExperiment, "bpe_classifier", "molnet_bbbp"),
        ]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                with initialize_config_dir(config_dir=str(ROOT / "configurations"), version_base=None):
                    cfg = compose(
                        config_name="config",
                        overrides=[
                            f"experiment={stage}",
                            f"algorithm={algo}",
                            f"dataset={dataset_name}",
                            "+name=resume_test",
                        ],
                    )
                t = cfg.experiment.training
                t.num_train_epochs = 1
                t.per_device_train_batch_size = 2
                t.per_device_eval_batch_size = 2
                t.dataloader_num_workers = 0
                if stage == "tokenizer_benchmark":
                    t.save_strategy = "epoch"
                    t.mlm_probability = 1.0
                rows = {"input_ids": [[5, 6] * 8 for _ in range(4)], "attention_mask": [[1] * 16 for _ in range(4)]}
                if stage == "downstream_classification":
                    rows["labels"] = [[0.0], [1.0], [0.0], [1.0]]
                ds = Dataset.from_dict(rows)
                splits = DatasetDict({k: ds for k in ("train", "validation", "test")})
                root = Path(tmp)

                def experiment_at(path, checkpoint=None):
                    exp = cls(cfg, path, checkpoint)
                    if stage == "tokenizer_benchmark":
                        exp.algo = SimpleNamespace(tokenizer=tokenizer, model=RobertaForMaskedLM(model_cfg))
                    else:
                        exp.algo = SimpleNamespace(
                            tokenizer=tokenizer,
                            build_model=lambda num_labels: RobertaForSequenceClassification(model_cfg),
                        )
                    return exp

                trainers = []

                def capture_trainer(*args, **kwargs):
                    trainer = Trainer(*args, **kwargs)
                    trainers.append(trainer)
                    return trainer

                with (
                    patch("transformers.Trainer", side_effect=capture_trainer),
                    patch("transformers.TrainingArguments", side_effect=cpu_arguments),
                    patch(
                        "experiments.tokenizer_benchmark.TokenizerBenchmarkExperiment._load_dataset",
                        return_value=splits,
                    ),
                    patch("experiments.downstream_classification.load_molnet", return_value=splits),
                ):
                    experiment_at(root / "first").training()
                    checkpoint = next((root / "first").glob("checkpoint-*"))
                    original_step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
                    cfg.resume = str(checkpoint)
                    validated = resolve_resume_checkpoint(cfg)
                    self.assertEqual(validated, checkpoint.resolve())
                    t.num_train_epochs = 2
                    experiment_at(root / "resumed", validated).training()
                    state = trainers[-1].state
                    self.assertGreater(state.global_step, original_step)
                    # A resumed Trainer retains the earlier run's log history.
                    original_logs = json.loads((checkpoint / "trainer_state.json").read_text())["log_history"]
                    self.assertEqual(state.log_history[: len(original_logs)], original_logs)


if __name__ == "__main__":
    unittest.main()
