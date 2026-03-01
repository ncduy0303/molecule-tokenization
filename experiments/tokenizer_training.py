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


import re


class CustomSmilesSplitter:
    """Custom pre-tokenizer splitter for SMILES strings using regex."""

    def __init__(self, mode: str):
        if mode == "atom_split":
            # Removed the outer capture group () as finditer doesn't need it
            self.pattern = re.compile(
                r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]"
            )
        elif mode == "structure_split":
            self.pattern = re.compile(r"\.|%\d{2}|[\(\)]|[/\\]|\[.*?\]|\d")
        else:
            raise ValueError(f"Unknown pretokenizer mode: {mode}")

    def split(self, i: int, normalized_string) -> list:
        text = str(normalized_string)
        splits = []
        last_idx = 0

        # Use finditer to locate matches, then slice the NormalizedString directly
        for match in self.pattern.finditer(text):
            start, end = match.span()

            # 1. Add any non-matched text that occurred before the current match
            if start > last_idx:
                splits.append(normalized_string[last_idx:start])

            # 2. Add the matched text itself
            splits.append(normalized_string[start:end])

            last_idx = end

        # 3. Add any remaining non-matched text after the final match
        if last_idx < len(text):
            splits.append(normalized_string[last_idx:])

        return splits

    def pre_tokenize(self, pretok):
        pretok.split(self.split)


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

        algo_cfg = self.algo.cfg  # type: ignore
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
            self._train_pcatt(corpus, algo_cfg)
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
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer, PreTokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder, BPEDecoder
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

        pretokenizer = self.root_cfg.dataset.get("pretokenizer", None)
        print(cyan("  Pre-tokenizer:"), pretokenizer)

        # 1. Setup Pre-tokenizer and Decoder
        if pretokenizer is None:
            tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False)
            tokenizer.decoder = ByteLevelDecoder()
        elif pretokenizer in ["atom_split", "structure_split"]:
            tokenizer.pre_tokenizer = PreTokenizer.custom(CustomSmilesSplitter(pretokenizer))
            tokenizer.decoder = BPEDecoder()
        else:
            raise ValueError(f"Unknown pretokenizer: {pretokenizer}")

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

        if pretokenizer in ["atom_split", "structure_split"]:
            # Remove custom pre-tokenizer for HuggingFace compatibility
            # We only use it for training
            tokenizer.pre_tokenizer = None

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
            base_vocab.pop(st, None)  # type: ignore

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

        print(cyan("Training Smirk-GPE tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)
        print(cyan("  Merge brackets:"), merge_brackets)
        print(cyan("  Split structure:"), split_structure)

        # [TODO]: why -4 for SGPE/+1 for GPE to reach len(tokenizer) == vocab_size?
        offset = -4 if split_structure else +1

        tokenizer = train_gpe(
            files=[str(corpus_path)],
            ref=SmirkTokenizerFast(),
            min_frequency=min_frequency,
            vocab_size=vocab_size - len(SmirkTokenizerFast().special_tokens_map) + offset,
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

    def _train_pcatt(self, corpus, algo_cfg):
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
        word_count, longest_word_len = build_word_counts(self.root_cfg.dataset)

        print(cyan("Training PCATT (GreedTok) tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)
        print(cyan("  Unique substructures:"), f"{len(word_count):,}")
        print(cyan("  Longest substructure:"), longest_word_len)

        pretokenizer = self.root_cfg.dataset.get("pretokenizer", None)
        print(cyan("  Pre-tokenizer:"), pretokenizer)
        if pretokenizer == "atom_split":

            def batch_iterator_split():
                batch_size = 1024
                # https://github.com/datamol-io/safe/blob/main/safe/tokenizer.py#50
                pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
                pat = re.compile(pattern)
                for i in range(0, len(corpus), batch_size):
                    yield [re.findall(pat, text) for text in corpus[i : i + batch_size]]

            # Switch to train from iterator with pre-tokenization, which is more memory-efficient
            tokenizer = GreedTok().train_new_from_iterator(
                batch_iterator_split(),
                vocab_size=vocab_size - 256,
                special_tokens_map=special_tokens,
                min_word_count=min_frequency,
                max_token_length=longest_word_len,
            )
        elif pretokenizer == "structure_split":
            tokenizer = GreedTok().train_new_from_counts(
                word_count,
                vocab_size=vocab_size - 256,
                special_tokens_map=special_tokens,
                min_word_count=min_frequency,
                max_token_length=longest_word_len,
            )
        elif pretokenizer is None:

            def batch_iterator_split():
                batch_size = 1000000
                pattern = r"\S+"  # No pre-tokenization, treat whole SMILES as one token
                pat = re.compile(pattern)
                for i in range(0, len(corpus), batch_size):
                    yield [re.findall(pat, text) for text in corpus[i : i + batch_size]]

            # Switch to train from iterator with pre-tokenization, which is more memory-efficient
            tokenizer = GreedTok().train_new_from_iterator(
                batch_iterator_split(),
                vocab_size=vocab_size - 256,
                special_tokens_map=special_tokens,
                min_word_count=min_frequency,
                max_token_length=longest_word_len,
            )
        else:
            raise ValueError(f"Unknown pretokenizer: {pretokenizer}")

        print(cyan("Trained vocab size:"), len(tokenizer))  # type: ignore

        # Save tokenizer
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))  # type: ignore
        print(cyan("Saved tokenizer to:"), save_dir)

        # Print some example tokenizations
        corpus, _ = load_pubchem10m_tokenizer_corpus(self.root_cfg.dataset)
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tokenizer.tokenize(smi)  # type: ignore
            print(f"  {smi}")
            print(f"    -> {tokens}")
            print(f"    -> {len(tokens)} tokens")
