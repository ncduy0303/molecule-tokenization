"""
Tokenizer Benchmark Experiment.

Trains a from-scratch RoBERTa MLM model using HuggingFace Trainer,
allowing easy comparison of different molecule tokenizers (smirk, atomwise, etc.)
within the research template's Hydra + wandb + Slurm infrastructure.

Usage:
    python -m main +name=smirk_qm9 \
        experiment=tokenizer_benchmark \
        dataset=qm9 \
        algorithm=smirk_roberta

    python -m main +name=atomwise_qm9 \
        experiment=tokenizer_benchmark \
        dataset=qm9 \
        algorithm=atomwise_roberta
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
        "smirk_roberta": MoleculeMLMAlgo,
        "atomwise_roberta": MoleculeMLMAlgo,
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

        tokenizer = self.algo.tokenizer
        model = self.algo.model

        print(cyan("Tokenizer:"), type(tokenizer).__name__)
        print(cyan("Vocab size:"), len(tokenizer))
        print(
            cyan("Model params:"),
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        )

        # Load and tokenize dataset
        dataset_name = self.root_cfg.dataset._name
        if dataset_name not in dataset_registry:
            raise ValueError(
                f"Dataset '{dataset_name}' not in registry. "
                f"Available: {list(dataset_registry.keys())}"
            )
        dataset = dataset_registry[dataset_name](self.root_cfg.dataset, tokenizer)

        print(cyan("Train samples:"), len(dataset["train"]))
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
            fp16=t.fp16,
            bf16=t.bf16,
            seed=t.seed,
            # Evaluation & Logging
            eval_strategy=t.eval_strategy,
            eval_steps=t.eval_steps,
            logging_steps=t.logging_steps,
            report_to=report_to,
            # Checkpointing
            save_steps=t.save_steps,
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
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        # Final evaluation
        results = trainer.evaluate()
        perplexity = math.exp(results["eval_loss"])
        print(cyan("Final eval loss:"), f"{results['eval_loss']:.4f}")
        print(cyan("Final perplexity:"), f"{perplexity:.2f}")

    def evaluation(self):
        """Evaluate a trained model on the test set."""
        from transformers import (
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )

        if not self.algo:
            self._build_algo()

        tokenizer = self.algo.tokenizer
        model = self.algo.model

        # Load dataset
        dataset_name = self.root_cfg.dataset._name
        dataset = dataset_registry[dataset_name](self.root_cfg.dataset, tokenizer)

        report_to = self._setup_wandb_env()

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.cfg.training.per_device_eval_batch_size,
            report_to=report_to,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.cfg.training.mlm_probability,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        results = trainer.evaluate()
        perplexity = math.exp(results["eval_loss"])
        print(cyan("Eval loss:"), f"{results['eval_loss']:.4f}")
        print(cyan("Perplexity:"), f"{perplexity:.2f}")
