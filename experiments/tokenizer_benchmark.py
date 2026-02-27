"""
Tokenizer Benchmark Experiment.

Trains a from-scratch RoBERTa MLM model using HuggingFace Trainer,
allowing easy comparison of different molecule tokenizers (smirk, atomwise, etc.)
within the research template's Hydra + wandb + Slurm infrastructure.

Usage (PubChem10M benchmark):
    python -m main +name=smirk_pubchem \
        experiment=tokenizer_benchmark \
        dataset=pubchem10m_mlm_train \
        algorithm=smirk_roberta
"""

import os
import math
from typing import Optional, Union
from pathlib import Path

from omegaconf import DictConfig

from experiments.exp_base import BaseExperiment
from algorithms.tokenizer_benchmark import MoleculeMLMAlgo
from datamodules.molecule_datasets import dataset_registry

from utils.print_utils import cyan


class TokenizerBenchmarkExperiment(BaseExperiment):
    """
    Benchmark different molecule tokenizers via masked language modeling.

    Uses HuggingFace Trainer (not PyTorch Lightning) because:
    - Tokenizers are HF PreTrainedTokenizer subclasses
    - DataCollatorForLanguageModeling handles MLM masking
    - HF Trainer has built-in wandb integration
    """

    # Map algorithm yaml names -> algo class
    # All use the same class; the yaml config determines the tokenizer type
    compatible_algorithms = {
        "pcatt_roberta": MoleculeMLMAlgo,
        "smirk_roberta": MoleculeMLMAlgo,
        "ape_roberta": MoleculeMLMAlgo,
        "bpe_roberta": MoleculeMLMAlgo,
        "ape_roberta_pretrained": MoleculeMLMAlgo,
        "bpe_roberta_pretrained": MoleculeMLMAlgo,
    }

    def __init__(
        self,
        root_cfg: DictConfig,
        output_dir: Optional[Union[str, Path]],
        ckpt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(root_cfg, output_dir, ckpt_path)

    def _setup_wandb_env(self):
        """Configure wandb via environment variables for HF Trainer."""
        wandb_cfg = self.root_cfg.wandb

        if wandb_cfg.mode == "disabled":
            os.environ["WANDB_DISABLED"] = "true"
            return "none"

        if wandb_cfg.mode == "offline":
            os.environ["WANDB_MODE"] = "offline"

        if wandb_cfg.entity:
            os.environ["WANDB_ENTITY"] = wandb_cfg.entity
        if wandb_cfg.project:
            os.environ["WANDB_PROJECT"] = wandb_cfg.project

        return "wandb"

    def _load_dataset(self):
        """Load and tokenize the configured dataset."""
        if not self.algo:
            self._build_algo()

        dataset_name = self.root_cfg.dataset._name
        if dataset_name not in dataset_registry:
            raise ValueError(
                f"Dataset '{dataset_name}' not in registry. " f"Available: {list(dataset_registry.keys())}"
            )
        return dataset_registry[dataset_name](self.root_cfg.dataset, self.algo.tokenizer)  # type: ignore

    def training(self):
        """Train a RoBERTa MLM model with the configured tokenizer and dataset."""
        from transformers import (
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )

        # Build tokenizer + model via the algorithm config
        if not self.algo:
            self._build_algo()

        tokenizer = self.algo.tokenizer  # type: ignore
        model = self.algo.model  # type: ignore

        print(cyan("Tokenizer:"), type(tokenizer).__name__)
        print(cyan("Vocab size:"), len(tokenizer))
        print(
            cyan("Model params:"),
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        )

        # Load and tokenize dataset (supports both 2-way and 3-way splits)
        dataset = self._load_dataset()

        has_val = "validation" in dataset
        eval_split = "validation" if has_val else "test"
        has_test = "test" in dataset

        print(cyan("Train samples:"), len(dataset["train"]))
        if has_val:
            print(cyan("Val samples:"), len(dataset["validation"]))
        if has_test:
            print(cyan("Test samples:"), len(dataset["test"]))

        # Configure wandb
        report_to = self._setup_wandb_env()

        # Build training arguments from experiment config
        t = self.cfg.training
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.root_cfg.name,
            # Training
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            num_train_epochs=t.num_train_epochs,
            learning_rate=t.learning_rate,
            warmup_steps=t.warmup_steps,
            weight_decay=t.weight_decay,
            lr_scheduler_type=t.lr_scheduler_type,
            adam_beta1=t.adam_beta1,
            adam_beta2=t.adam_beta2,
            adam_epsilon=t.adam_epsilon,
            max_grad_norm=t.max_grad_norm,
            fp16=t.fp16,
            bf16=t.bf16,
            seed=t.seed,
            # Evaluation & Logging
            eval_strategy=t.eval_strategy,
            eval_steps=t.get("eval_steps", None),
            logging_steps=t.logging_steps,
            report_to=report_to,
            # Checkpointing – save only the last model
            save_strategy=t.save_strategy,
            save_total_limit=t.save_total_limit,
            # Data
            dataloader_num_workers=t.dataloader_num_workers,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=t.mlm_probability,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset[eval_split],
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        # Validation eval (final)
        val_results = trainer.evaluate(eval_dataset=dataset[eval_split])  # type: ignore
        val_ppl = math.exp(val_results["eval_loss"])
        print(cyan(f"Final {eval_split} loss:"), f"{val_results['eval_loss']:.4f}")
        print(cyan(f"Final {eval_split} perplexity:"), f"{val_ppl:.2f}")

        # Test evaluation (separate split, run once at the very end)
        if has_val and has_test:
            test_results = trainer.evaluate(eval_dataset=dataset["test"])  # type: ignore
            test_ppl = math.exp(test_results["eval_loss"])
            print(cyan("Test loss:"), f"{test_results['eval_loss']:.4f}")
            print(cyan("Test perplexity:"), f"{test_ppl:.2f}")

        # Save the final model checkpoint
        final_ckpt_dir = self.output_dir / "final_checkpoint"
        trainer.save_model(str(final_ckpt_dir))
        tokenizer.save_pretrained(str(final_ckpt_dir))
        print(cyan("Saved final checkpoint to:"), final_ckpt_dir)

    def evaluation(self):
        """Evaluate a model on the validation and test sets."""
        from transformers import (
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )

        if not self.algo:
            self._build_algo()

        tokenizer = self.algo.tokenizer  # type: ignore
        model = self.algo.model  # type: ignore

        print(cyan("Tokenizer:"), type(tokenizer).__name__)
        print(cyan("Vocab size:"), len(tokenizer))
        print(
            cyan("Model params:"),
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        )

        # Load dataset
        dataset = self._load_dataset()

        has_val = "validation" in dataset
        has_test = "test" in dataset

        if has_val:
            print(cyan("Val samples:"), len(dataset["validation"]))
        if has_test:
            print(cyan("Test samples:"), len(dataset["test"]))

        report_to = self._setup_wandb_env()

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.root_cfg.name,
            per_device_eval_batch_size=self.cfg.training.per_device_eval_batch_size,
            report_to=report_to,
            seed=self.cfg.training.seed,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.cfg.training.mlm_probability,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        # Evaluate on validation set
        if has_val:
            val_results = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="val")  # type: ignore
            val_ppl = math.exp(val_results["val_loss"])
            print(cyan("Val loss:"), f"{val_results['val_loss']:.4f}")
            print(cyan("Val perplexity:"), f"{val_ppl:.2f}")

        # Evaluate on test set
        if has_test:
            test_results = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")  # type: ignore
            test_ppl = math.exp(test_results["test_loss"])
            print(cyan("Test loss:"), f"{test_results['test_loss']:.4f}")
            print(cyan("Test perplexity:"), f"{test_ppl:.2f}")
