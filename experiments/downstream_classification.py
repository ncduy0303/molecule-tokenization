"""
Downstream Classification Finetuning Experiment.

Finetunes a pretrained RoBERTa model on MoleculeNet classification tasks
(HIV, BBBP, Tox21, etc.) with ROC-AUC evaluation and early stopping.

Usage:
    python -m main +name=bpe_hiv \
        experiment=downstream_classification \
        dataset=molnet_hiv \
        algorithm=bpe_classifier
"""

import os
import math
import numpy as np
from typing import Optional, Union
from pathlib import Path

from omegaconf import DictConfig

from experiments.exp_base import BaseExperiment
from algorithms.downstream_finetune import MoleculeClassificationAlgo
from datamodules.molecule_datasets.molnet import load_molnet

from utils.print_utils import cyan


def _compute_roc_auc(eval_pred):
    """Compute ROC-AUC, handling multi-label and NaN-masked labels."""
    import torch
    from sklearn.metrics import roc_auc_score

    logits, labels = eval_pred
    # Sigmoid to get probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels)

    # Per-task ROC-AUC, ignoring masked labels (-1)
    aucs = []
    for i in range(labels.shape[1]):
        mask = labels[:, i] >= 0  # valid labels only
        if mask.sum() == 0:
            continue
        y_true = labels[mask, i]
        y_score = probs[mask, i]
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            continue
        aucs.append(roc_auc_score(y_true, y_score))

    mean_auc = np.mean(aucs) if aucs else 0.0
    return {"roc_auc": mean_auc}


class DownstreamClassificationExperiment(BaseExperiment):
    """
    Finetune a pretrained RoBERTa on MoleculeNet classification tasks.
    """

    compatible_algorithms = {
        "ape_classifier": MoleculeClassificationAlgo,
        "bpe_classifier": MoleculeClassificationAlgo,
        "smirk_classifier": MoleculeClassificationAlgo,
        "ape_classifier_pretrained_hiv": MoleculeClassificationAlgo,
        "bpe_classifier_pretrained_hiv": MoleculeClassificationAlgo,
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
        """Finetune pretrained RoBERTa on a classification task."""
        from transformers import (
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )

        if not self.algo:
            self._build_algo()

        tokenizer = self.algo.tokenizer

        # ── Load dataset ────────────────────────────────────────────────
        dataset = load_molnet(self.root_cfg.dataset, tokenizer)

        # Infer num_labels from data
        sample_labels = dataset["train"][0]["labels"]
        num_labels = len(sample_labels)
        print(cyan("Num labels:"), num_labels)

        # ── Build classification model ──────────────────────────────────
        model = self.algo.build_model(num_labels=num_labels)

        print(cyan("Tokenizer:"), type(tokenizer).__name__)
        print(cyan("Vocab size:"), len(tokenizer))
        print(
            cyan("Model params:"),
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        )
        print(cyan("Train samples:"), len(dataset["train"]))
        print(cyan("Val samples:"), len(dataset["validation"]))
        print(cyan("Test samples:"), len(dataset["test"]))

        # ── Configure wandb ─────────────────────────────────────────────
        report_to = self._setup_wandb_env()

        # ── Training arguments ──────────────────────────────────────────
        t = self.cfg.training
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.root_cfg.name,
            # Training
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            num_train_epochs=t.num_train_epochs,
            learning_rate=t.learning_rate,
            warmup_ratio=t.warmup_ratio,
            weight_decay=t.weight_decay,
            lr_scheduler_type=t.lr_scheduler_type,
            max_grad_norm=t.max_grad_norm,
            fp16=t.fp16,
            bf16=t.bf16,
            seed=t.seed,
            # Evaluation & Logging
            eval_strategy="epoch",
            logging_steps=t.logging_steps,
            report_to=report_to,
            # Checkpointing – save best by ROC-AUC
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            # Data
            dataloader_num_workers=t.dataloader_num_workers,
            remove_unused_columns=False,
        )

        # ── Custom data collator for classification ─────────────────────
        from transformers import DataCollatorWithPadding
        base_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

        def collate_fn(features):
            """Pad input_ids/attention_mask and stack labels."""
            import torch

            labels = [f.pop("labels") for f in features]
            batch = base_collator(features)
            batch["labels"] = torch.tensor(labels, dtype=torch.float)
            return batch

        # ── Early stopping ──────────────────────────────────────────────
        early_stopping_patience = t.early_stopping_patience

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            data_collator=collate_fn,
            compute_metrics=_compute_roc_auc,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                ),
            ],
        )

        trainer.train()

        # ── Final validation ────────────────────────────────────────────
        val_results = trainer.evaluate(eval_dataset=dataset["validation"])
        print(cyan("Val ROC-AUC:"), f"{val_results.get('eval_roc_auc', 0):.4f}")
        print(cyan("Val loss:"), f"{val_results['eval_loss']:.4f}")

        # ── Test evaluation ─────────────────────────────────────────────
        test_results = trainer.evaluate(
            eval_dataset=dataset["test"], metric_key_prefix="test"
        )
        print(cyan("Test ROC-AUC:"), f"{test_results.get('test_roc_auc', 0):.4f}")
        print(cyan("Test loss:"), f"{test_results['test_loss']:.4f}")

        # ── Save ────────────────────────────────────────────────────────
        final_dir = self.output_dir / "best_model"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(cyan("Saved best model to:"), final_dir)

    def evaluation(self):
        """Evaluate a saved model on val + test sets."""
        from transformers import Trainer, TrainingArguments

        if not self.algo:
            self._build_algo()

        tokenizer = self.algo.tokenizer
        dataset = load_molnet(self.root_cfg.dataset, tokenizer)

        num_labels = len(dataset["train"][0]["labels"])
        model = self.algo.build_model(num_labels=num_labels)

        report_to = self._setup_wandb_env()

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.root_cfg.name,
            per_device_eval_batch_size=self.cfg.training.per_device_eval_batch_size,
            report_to=report_to,
            remove_unused_columns=False,
        )

        from transformers import DataCollatorWithPadding
        base_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

        def collate_fn(features):
            import torch

            labels = [f.pop("labels") for f in features]
            batch = base_collator(features)
            batch["labels"] = torch.tensor(labels, dtype=torch.float)
            return batch

        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=collate_fn,
            compute_metrics=_compute_roc_auc,
        )

        val_results = trainer.evaluate(
            eval_dataset=dataset["validation"], metric_key_prefix="val"
        )
        print(cyan("Val ROC-AUC:"), f"{val_results.get('val_roc_auc', 0):.4f}")

        test_results = trainer.evaluate(
            eval_dataset=dataset["test"], metric_key_prefix="test"
        )
        print(cyan("Test ROC-AUC:"), f"{test_results.get('test_roc_auc', 0):.4f}")
