"""
Tokenizer Training Experiment.

Trains a tokenizer (BPE, WordPiece, etc.) on a fixed corpus of SMILES
strings from PubChem10M.  This is CPU-only — no model is involved.

Usage:
    python -m main +name=train_bpe_tokenizer \
        experiment=tokenizer_training \
        dataset=pubchem10m_tokenizer_train \
        algorithm=train_bpe
"""

import os
from typing import Optional, Union
from pathlib import Path

from omegaconf import DictConfig

from experiments.exp_base import BaseExperiment
from datamodules.molecule_datasets.pubchem10m_tokenizer_train import (
    load_pubchem10m_tokenizer_corpus,
)

from utils.print_utils import cyan


class TokenizerTrainer:
    """Dummy algo wrapper — holds config only, no model needed."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg


class TokenizerTrainingExperiment(BaseExperiment):
    """
    Train a tokenizer on a fixed corpus of SMILES strings.
    CPU-only — no GPU needed.
    """

    compatible_algorithms = {
        "train_bpe": TokenizerTrainer,
        "train_smirk_gpe": TokenizerTrainer,
    }

    def __init__(
        self,
        root_cfg: DictConfig,
        output_dir: Optional[Union[str, Path]],
        ckpt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(root_cfg, output_dir, ckpt_path)

    def training(self):
        """Train a tokenizer on the SMILES corpus."""
        if not self.algo:
            self._build_algo()

        algo_cfg = self.algo.cfg
        tok_type = algo_cfg.tokenizer.type

        # Load the fixed SMILES corpus (returns list + path to corpus.txt)
        corpus, corpus_path = load_pubchem10m_tokenizer_corpus(self.root_cfg.dataset)
        print(cyan("Corpus size:"), f"{len(corpus):,} SMILES strings")
        print(cyan("Corpus file:"), str(corpus_path))

        if tok_type == "bpe":
            self._train_bpe(corpus, algo_cfg)
        elif tok_type == "smirk_gpe":
            self._train_smirk_gpe(corpus_path, algo_cfg)
        else:
            raise ValueError(
                f"Unknown tokenizer training type: '{tok_type}'. "
                "Supported: 'bpe', 'smirk_gpe'"
            )

    def _train_bpe(self, corpus, algo_cfg):
        """Train a BPE tokenizer using HuggingFace tokenizers library."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from transformers import PreTrainedTokenizerFast

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        min_frequency = tok_cfg.min_frequency

        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

        print(cyan("Training BPE tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)

        # Set up tokenizer with byte-level pre-tokenization
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Train from iterator (memory-efficient)
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        print(cyan("Trained vocab size:"), tokenizer.get_vocab_size())

        # Wrap in HuggingFace PreTrainedTokenizerFast
        tok_tf = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )

        # Save tokenizer
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        tok_tf.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # Print some example tokenizations
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tok_tf.tokenize(smi)
            print(f"  {smi}")
            print(f"    -> {tokens}")
            print(f"    -> {len(tokens)} tokens")

    def _train_smirk_gpe(self, corpus_path, algo_cfg):
        """Train a Smirk-GPE tokenizer using the smirk library."""
        from smirk import SmirkTokenizerFast, train_gpe

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        min_frequency = tok_cfg.min_frequency
        merge_brackets = tok_cfg.merge_brackets
        split_structure = tok_cfg.split_structure

        print(cyan("Training Smirk-GPE tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)
        print(cyan("  Merge brackets:"), merge_brackets)
        print(cyan("  Split structure:"), split_structure)

        tokenizer = train_gpe(
            files=[str(corpus_path)],
            ref=SmirkTokenizerFast(),
            min_frequency=min_frequency,
            vocab_size=vocab_size,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
        )

        print(cyan("Trained vocab size:"), len(tokenizer))

        # Save tokenizer
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # Print some example tokenizations
        examples = corpus_path.read_text().splitlines()[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tokenizer.tokenize(smi)
            print(f"  {smi}")
            print(f"    -> {tokens}")
            print(f"    -> {len(tokens)} tokens")
