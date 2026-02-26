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

import json
import os
from typing import Optional, Union
from pathlib import Path

from omegaconf import DictConfig

from experiments.exp_base import BaseExperiment
from datamodules.molecule_datasets.pubchem10m_tokenizer_train import (
    load_pubchem10m_tokenizer_corpus,
    build_word_counts,
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
        "train_ape": TokenizerTrainer,
        "train_bpe": TokenizerTrainer,
        "train_smirk": TokenizerTrainer,
        "train_smirk_gpe": TokenizerTrainer,
        "train_pcatt": TokenizerTrainer,
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

        if tok_type == "ape":
            self._train_ape(corpus, algo_cfg)
        elif tok_type == "bpe":
            self._train_bpe(corpus, algo_cfg)
        elif tok_type == "smirk":
            self._train_smirk(corpus_path, algo_cfg)
        elif tok_type == "smirk_gpe":
            self._train_smirk_gpe(corpus_path, algo_cfg)
        elif tok_type == "pcatt":
            self._train_pcatt(algo_cfg)
        else:
            raise ValueError(
                f"Unknown tokenizer training type: '{tok_type}'. "
                "Supported: 'ape', 'bpe', 'smirk', 'smirk_gpe', 'pcatt'"
            )

    def _train_ape(self, corpus, algo_cfg):
        """Train an APE tokenizer using the apetokenizer library."""
        from apetokenizer.ape_tokenizer import APETokenizer

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        min_frequency = tok_cfg.min_frequency

        print(cyan("Training APE tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)

        tokenizer = APETokenizer(
            pad_token=tok_cfg.pad_token,
            bos_token=tok_cfg.bos_token,
            eos_token=tok_cfg.eos_token,
            unk_token=tok_cfg.unk_token,
            mask_token=tok_cfg.mask_token,
        )

        tokenizer.train(corpus, max_vocab_size=vocab_size, min_freq_for_merge=min_frequency)

        print(cyan("Trained vocab size:"), len(tokenizer))

        # Save tokenizer
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # Set unk_token_id (Not set by APETokenizer)
        tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)  # type: ignore

        # Bind tokenize method
        def ape_tokenize(self, text: str) -> list[str]:
            return self.convert_ids_to_tokens(self.encode(text))

        tokenizer.tokenize = ape_tokenize.__get__(tokenizer)  # type: ignore

        # Print some example tokenizations
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tokenizer.tokenize(smi)  # type: ignore
            print(f"  {smi}")
            print(f"    -> {tokens}")
            print(f"    -> {len(tokens)} tokens")

    def _train_bpe(self, corpus, algo_cfg):
        """Train a BBPE tokenizer using HuggingFace tokenizers library."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from transformers import PreTrainedTokenizerFast

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        min_frequency = tok_cfg.min_frequency

        special_tokens = [
            tok_cfg.bos_token,
            tok_cfg.pad_token,
            tok_cfg.eos_token,
            tok_cfg.unk_token,
            tok_cfg.cls_token,
            tok_cfg.sep_token,
            tok_cfg.mask_token,
        ]

        print(cyan("Training BBPE tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)

        # Set up tokenizer. (Note: unk_token is rarely/never triggered in BBPE,
        # but kept here to align with your special_tokens list).
        tokenizer = Tokenizer(BPE(unk_token=tok_cfg.unk_token))

        # 1. Set add_prefix_space=False for SMILES strings
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        # 2. Explicitly pass the ByteLevel alphabet to the trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            show_progress=True,
        )

        # Train from iterator (memory-efficient)
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        print(cyan("Trained vocab size:"), tokenizer.get_vocab_size())

        # Wrap in HuggingFace PreTrainedTokenizerFast
        tok_tf = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=tok_cfg.unk_token,
            pad_token=tok_cfg.pad_token,
            bos_token=tok_cfg.bos_token,
            eos_token=tok_cfg.eos_token,
            mask_token=tok_cfg.mask_token,
            cls_token=tok_cfg.cls_token,
            sep_token=tok_cfg.sep_token,
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

    def _train_smirk(self, corpus_path, algo_cfg):
        """Rearrange the vocab indexes of the base Smirk tokenizer to match RoBERTa's expected special token indices."""
        from smirk import SmirkTokenizerFast

        default_tokenizer = SmirkTokenizerFast()
        base_vocab = default_tokenizer.get_vocab()

        special_tokens = [
            default_tokenizer.cls_token,
            default_tokenizer.pad_token,
            default_tokenizer.sep_token,
            default_tokenizer.unk_token,
            default_tokenizer.mask_token,
            default_tokenizer.bos_token,
            default_tokenizer.eos_token,
        ]
        for st in special_tokens:
            base_vocab.pop(st, None)

        # Move all special tokens to the front of the vocab with fixed indices for RoBERTa compatibility
        # Padding index must be 1
        new_vocab = {
            "[BOS]": 0,
            "[PAD]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
            "[CLS]": 4,
            "[SEP]": 5,
            "[MASK]": 6,
        }

        # Add the rest of the chemical tokens
        current_idx = len(new_vocab)
        for token in sorted(base_vocab.keys()):  # Sort for deterministic assignment
            new_vocab[token] = current_idx
            current_idx += 1

        with open("vocab.json", "w") as f:
            json.dump(new_vocab, f, indent=4)

        # Re-instantiate the SmirkTokenizerFast with the new vocab
        tokenizer = SmirkTokenizerFast(vocab_file="vocab.json")

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

    def _train_smirk_gpe(self, corpus_path, algo_cfg):
        """Train a Smirk-GPE tokenizer using the smirk library."""
        from smirk import SmirkTokenizerFast, train_gpe

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        min_frequency = tok_cfg.min_frequency
        merge_brackets = tok_cfg.merge_brackets
        split_structure = tok_cfg.split_structure
        base_tokenizer_file = tok_cfg.tokenizer_file

        print(cyan("Training Smirk-GPE tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)
        print(cyan("  Merge brackets:"), merge_brackets)
        print(cyan("  Split structure:"), split_structure)

        tokenizer = train_gpe(
            files=[str(corpus_path)],
            ref=SmirkTokenizerFast(base_tokenizer_file),
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

    def _train_pcatt(self, algo_cfg):
        """Train a PCATT (GreedTok) tokenizer using the pcatt library."""
        from pcatt.hf.greedtok import GreedTok

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        min_frequency = tok_cfg.min_frequency

        special_tokens = {
            "bos_token": tok_cfg.bos_token,
            "pad_token": tok_cfg.pad_token,
            "eos_token": tok_cfg.eos_token,
            "unk_token": tok_cfg.unk_token,
            "cls_token": tok_cfg.cls_token,
            "sep_token": tok_cfg.sep_token,
            "mask_token": tok_cfg.mask_token,
        }

        # Build (or load cached) word counts from the corpus
        word_count, longest_struct_len = build_word_counts(self.root_cfg.dataset)

        print(cyan("Training PCATT (GreedTok) tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)
        print(cyan("  Unique substructures:"), f"{len(word_count):,}")
        print(cyan("  Longest substructure:"), longest_struct_len)

        tokenizer = GreedTok().train_new_from_counts(
            word_count,
            vocab_size=vocab_size - 256 + len(special_tokens),  # [DEBUG]
            special_tokens_map=special_tokens,
            min_word_count=min_frequency,
            max_token_length=longest_struct_len,
        )

        print(cyan("Trained vocab size:"), len(tokenizer))

        # Save tokenizer
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # Print some example tokenizations
        corpus, _ = load_pubchem10m_tokenizer_corpus(self.root_cfg.dataset)
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tokenizer.tokenize(smi)
            print(f"  {smi}")
            print(f"    -> {tokens}")
            print(f"    -> {len(tokens)} tokens")
