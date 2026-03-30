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
    build_smirk_pcatt_word_counts,
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
        "train_spe": TokenizerTrainer,
        "train_smirk": TokenizerTrainer,
        "train_smirk_gpe": TokenizerTrainer,
        "train_pcatt": TokenizerTrainer,
        "train_smirk_pcatt": TokenizerTrainer,
        "train_fragsmiles": TokenizerTrainer,
        "train_tsmiles": TokenizerTrainer,
        "train_ps_fragsmiles": TokenizerTrainer,
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
        elif tok_type == "spe":
            self._train_spe(corpus, algo_cfg)
        elif tok_type == "smirk_pcatt":
            self._train_smirk_pcatt(corpus, algo_cfg)
        elif tok_type == "fragsmiles":
            self._train_fragsmiles(corpus, algo_cfg)
        elif tok_type == "tsmiles":
            self._train_tsmiles(corpus, algo_cfg)
        elif tok_type == "ps_fragsmiles":
            self._train_ps(corpus_path, algo_cfg)
        else:
            raise ValueError(
                f"Unknown tokenizer training type: '{tok_type}'. "
                "Supported: 'ape', 'bpe', 'spe', 'smirk', 'smirk_gpe', 'pcatt', 'smirk_pcatt', 'fragsmiles', 'tsmiles', 'ps_fragsmiles'"
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

    def _train_spe(self, corpus, algo_cfg):
        """Train a SMILES Pair Encoding (SPE) tokenizer using the SmilesPE library."""
        import codecs
        from SmilesPE.learner import learn_SPE
        from SmilesPE.pretokenizer import atomwise_tokenizer
        from utils.spe_tokenizer import SMILES_SPE_Tokenizer

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

        print(cyan("Training SPE tokenizer..."))
        print(cyan("  Target vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)

        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Train SPE merge pairs ───────────────────────────────
        spe_file = str(save_dir / "spe_voc.txt")
        print(cyan("  Training SPE merge pairs..."))
        learn_SPE(
            infile=corpus,
            outfile=codecs.open(spe_file, "w"),
            num_symbols=vocab_size - len(special_tokens),
            min_frequency=min_frequency,
            augmentation=0,
            total_symbols=True,
            verbose=True,
        )

        # ── Step 2: Collect unique atom-level tokens from the corpus ────
        print(cyan("  Collecting atom-level tokens..."))
        atom_tokens: set[str] = set()
        for smi in corpus:
            atom_tokens.update(atomwise_tokenizer(smi))
        atom_tokens_sorted = sorted(atom_tokens)
        print(cyan("  Unique atom tokens:"), len(atom_tokens_sorted))

        # ── Step 3: Collect SPE merge tokens ────────────────────────────
        print(cyan("  Collecting SPE merge tokens..."))
        spe_tokens: list[str] = []
        with open(spe_file, "r") as f:
            for line in f:
                pair = line.strip()
                if pair:
                    # The SPE file has space-separated pairs; the merged token
                    # is the concatenation of the two parts.
                    merged = "".join(pair.split()[:2])
                    spe_tokens.append(merged)
        print(cyan("  SPE merge tokens:"), len(spe_tokens))

        # ── Step 4: Build complete vocabulary ───────────────────────────
        # Order: special tokens -> atom tokens -> SPE merged tokens
        all_tokens = list(special_tokens)
        seen = set(all_tokens)
        for tok in atom_tokens_sorted:
            if tok not in seen:
                all_tokens.append(tok)
                seen.add(tok)
        for tok in spe_tokens:
            if tok not in seen:
                all_tokens.append(tok)
                seen.add(tok)

        vocab_file = str(save_dir / "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            for tok in all_tokens:
                f.write(tok + "\n")

        print(cyan("  Total vocab size:"), len(all_tokens))

        # ── Step 5: Create and save the tokenizer ───────────────────────
        tokenizer = SMILES_SPE_Tokenizer(
            vocab_file=vocab_file,
            spe_file=spe_file,
            unk_token=tok_cfg.unk_token,
            sep_token=tok_cfg.sep_token,
            pad_token=tok_cfg.pad_token,
            cls_token=tok_cfg.cls_token,
            mask_token=tok_cfg.mask_token,
            bos_token=tok_cfg.bos_token,
            eos_token=tok_cfg.eos_token,
        )
        tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # Print some example tokenizations
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tokenizer.tokenize(smi)
            print(f"  {smi}")
            print(f"    -> {tokens}")
            print(f"    -> {len(tokens)} tokens")

    def _train_smirk_pcatt(self, corpus, algo_cfg):
        """Train a Smirk-PCATT (GreedTok) tokenizer.

        Uses the Smirk tokenizer for pre-tokenization, maps glyphs to bytes
        via SmirkPCATTAdapter, then trains a GreedTok tokenizer on the
        byte-encoded word counts. Supports the structure_split pretokenizer.
        """
        from pcatt.hf.greedtok import GreedTok
        from smirk import SmirkTokenizerFast

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

        # Build (or load cached) Smirk-adapted byte-level word counts
        word_count, longest_word_len, adapter = build_smirk_pcatt_word_counts(self.root_cfg.dataset)

        pretokenizer = self.root_cfg.dataset.get("pretokenizer", None)

        print(cyan("Training Smirk-PCATT (GreedTok) tokenizer..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Min frequency:"), min_frequency)
        print(cyan("  Pre-tokenizer:"), pretokenizer)
        print(cyan("  Unique substructures:"), f"{len(word_count):,}")
        print(cyan("  Longest substructure:"), longest_word_len)

        tokenizer = GreedTok().train_new_from_counts(
            word_count,
            vocab_size=vocab_size - 256,
            special_tokens_map=special_tokens,
            min_word_count=min_frequency,
            max_token_length=longest_word_len,
        )

        print(cyan("Trained vocab size:"), len(tokenizer))  # type: ignore

        # Save tokenizer
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))  # type: ignore

        # Save adapter metadata so SmirkPCATTTokenizer.from_pretrained works
        meta = {"smirk_vocab": adapter.smirk_vocab}
        with open(save_dir / "smirk_pcatt_meta.json", "w") as f:
            json.dump(meta, f)

        print(cyan("Saved tokenizer to:"), save_dir)

        # Print some example tokenizations
        smirk_tokenizer = SmirkTokenizerFast()
        corpus, _ = load_pubchem10m_tokenizer_corpus(self.root_cfg.dataset)
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            # Encode via Smirk + adapter, then tokenize with the trained PCATT
            smirk_tokens = smirk_tokenizer.tokenize(smi)
            pcatt_input = adapter.encode_for_pcatt(smirk_tokens)
            tokens = tokenizer.tokenize(pcatt_input)  # type: ignore
            # Decode byte tokens back to Smirk glyphs for readability
            decoded_tokens = ["".join(adapter.decode_from_pcatt(t)) for t in tokens]  # type: ignore
            print(f"  {smi}")
            print(f"    -> Smirk: {smirk_tokens}")
            print(f"    -> Smirk-PCATT: {decoded_tokens}")
            print(f"    -> {len(decoded_tokens)} tokens")

    def _train_fragsmiles(self, corpus, algo_cfg):
        """Train a FragSMILES tokenizer from a precomputed fragSMILES corpus.

        The vocabulary is built by splitting each fragSMILES string with
        chemicalgof.split and counting fragment frequencies.
        """
        import chemicalgof
        from utils.fragsmiles_tokenizer import FragSMILESTokenizer

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size

        special_tokens = [
            tok_cfg.bos_token,
            tok_cfg.pad_token,
            tok_cfg.eos_token,
            tok_cfg.unk_token,
            tok_cfg.cls_token,
            tok_cfg.sep_token,
            tok_cfg.mask_token,
        ]

        print(cyan("Training FragSMILES tokenizer..."))
        print(cyan("  Target vocab size:"), vocab_size)
        if not self.root_cfg.dataset.get("use_fragsmiles", False):
            print(cyan("  Warning:"), "dataset.use_fragsmiles=false; expected precomputed fragSMILES corpus")

        # ── Step 1: Build fragment vocabulary from corpus ─────────────────
        print(cyan("  Building fragment vocabulary from corpus..."))
        fragment_counts: dict[str, int] = {}

        for fragsmi in corpus:
            try:
                fragments = chemicalgof.split(fragsmi)

                for frag in fragments:
                    if frag:  # Skip empty fragments
                        fragment_counts[frag] = fragment_counts.get(frag, 0) + 1
            except Exception:
                # Skip problematic SMILES strings
                pass

        # Sort fragments by frequency (most common first)
        sorted_fragments = sorted(fragment_counts.items(), key=lambda x: x[1], reverse=True)
        print(cyan("  Unique fragments found:"), len(sorted_fragments))

        # ── Step 2: Build vocabulary with special tokens first ──────────
        vocab_dict: dict[str, int] = {}

        # Add special tokens first with fixed indices
        for i, token in enumerate(special_tokens):
            if token not in vocab_dict:
                vocab_dict[token] = i

        # Add fragments in order of frequency until we reach vocab_size
        for frag, count in sorted_fragments:
            if frag not in vocab_dict:
                if len(vocab_dict) >= vocab_size:
                    break
                vocab_dict[frag] = len(vocab_dict)

        print(cyan("  Final vocab size:"), len(vocab_dict))
        print(cyan("  Special tokens:"), len(special_tokens))
        print(cyan("  Fragment tokens:"), len(vocab_dict) - len(special_tokens))

        # ── Step 3: Create and save tokenizer ──────────────────────────
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_file = save_dir / "vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # Save the sorted fragment list for reference (optional)
        fragments_file = save_dir / "fragments.txt"
        with open(fragments_file, "w", encoding="utf-8") as f:
            for frag, count in sorted_fragments:
                f.write(f"{frag}\t{count}\n")

        # Create tokenizer instance
        tokenizer = FragSMILESTokenizer(
            vocab_file=str(vocab_file),
            unk_token=tok_cfg.unk_token,
            sep_token=tok_cfg.sep_token,
            pad_token=tok_cfg.pad_token,
            cls_token=tok_cfg.cls_token,
            mask_token=tok_cfg.mask_token,
            bos_token=tok_cfg.bos_token,
            eos_token=tok_cfg.eos_token,
        )

        # Save full tokenizer
        tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # ── Step 4: Print example tokenizations ────────────────────────
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for smi in examples:
            tokens = tokenizer.tokenize(smi)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            print(f"  {smi}")
            print(f"    -> Fragments: {tokens}")
            print(f"    -> Token IDs: {token_ids}")
            print(f"    -> {len(tokens)} tokens")

    def _train_tsmiles(self, corpus, algo_cfg):
        """Train a t-SMILES tokenizer from a precomputed t-SMILES corpus.

        The vocabulary is built by splitting each t-SMILES string with the
        structural regex r'&{1}|\\^{1}|[^&\\^]+' and counting token frequencies.
        The final vocabulary is capped at vocab_size (most frequent first).
        """
        import re
        from utils.tsmiles_tokenizer import TSMILESTokenizer

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size

        special_tokens = [
            tok_cfg.bos_token,
            tok_cfg.pad_token,
            tok_cfg.eos_token,
            tok_cfg.unk_token,
            tok_cfg.cls_token,
            tok_cfg.sep_token,
            tok_cfg.mask_token,
        ]

        tsmiles_re = re.compile(r"&{1}|\^{1}|[^&\^]+")

        print(cyan("Training t-SMILES tokenizer..."))
        print(cyan("  Target vocab size:"), vocab_size)
        if not self.root_cfg.dataset.get("use_tsmiles", False):
            print(cyan("  Warning:"), "dataset.use_tsmiles=false; expected precomputed t-SMILES corpus")

        # ── Step 1: Build token vocabulary from corpus ───────────────────
        print(cyan("  Building token vocabulary from corpus..."))
        token_counts: dict[str, int] = {}

        for tsmiles_str in corpus:
            try:
                tokens = tsmiles_re.findall(tsmiles_str)
                for tok in tokens:
                    if tok:
                        token_counts[tok] = token_counts.get(tok, 0) + 1
            except Exception:
                pass

        # Sort tokens by frequency (most common first)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        print(cyan("  Unique tokens found:"), len(sorted_tokens))

        # ── Step 2: Build vocabulary with special tokens first ───────────
        vocab_dict: dict[str, int] = {}

        for token in special_tokens:
            if token not in vocab_dict:
                vocab_dict[token] = len(vocab_dict)

        for tok, _count in sorted_tokens:
            if tok not in vocab_dict:
                if len(vocab_dict) >= vocab_size:
                    break
                vocab_dict[tok] = len(vocab_dict)

        print(cyan("  Final vocab size:"), len(vocab_dict))
        print(cyan("  Special tokens:"), len(special_tokens))
        print(cyan("  Fragment tokens:"), len(vocab_dict) - len(special_tokens))

        # ── Step 3: Create and save tokenizer ────────────────────────────
        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)

        vocab_file = save_dir / "vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # Save the sorted token list for reference
        tokens_file = save_dir / "tokens.txt"
        with open(tokens_file, "w", encoding="utf-8") as f:
            for tok, count in sorted_tokens:
                f.write(f"{tok}\t{count}\n")

        tokenizer = TSMILESTokenizer(
            vocab_file=str(vocab_file),
            unk_token=tok_cfg.unk_token,
            sep_token=tok_cfg.sep_token,
            pad_token=tok_cfg.pad_token,
            cls_token=tok_cfg.cls_token,
            mask_token=tok_cfg.mask_token,
            bos_token=tok_cfg.bos_token,
            eos_token=tok_cfg.eos_token,
        )

        tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved tokenizer to:"), save_dir)

        # ── Step 4: Print example tokenizations ──────────────────────────
        examples = corpus[:5]
        print(cyan("\nExample tokenizations:"))
        for tsmiles_str in examples:
            tokens = tokenizer.tokenize(tsmiles_str)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            print(f"  {tsmiles_str}")
            print(f"    -> Tokens: {tokens}")
            print(f"    -> Token IDs: {token_ids}")
            print(f"    -> {len(tokens)} tokens")

    def _train_ps(self, corpus_path, algo_cfg):
        """Train a PS-fragSMILES tokenizer.

        Stage 1: Train PS-VAE principal-subgraph vocab with graph BPE and save
                 the raw tab-delimited PS vocab.
        Stage 2: Encode corpus SMILES -> PS-fragSMILES via utils.ps.ps_encode.
        Stage 3: Build a WordLevel-style vocab.json (token -> id) from
                 chemicalgof.split token frequencies and save a
                 PSFragSMILESTokenizer.
        """
        import chemicalgof
        from utils.ps.mol_bpe_ring import graph_bpe, Tokenizer
        from utils.ps_fragsmiles_utils import encode_ps_fragsmiles_batch
        from utils.ps_fragsmiles_tokenizer import PSFragSMILESTokenizer

        tok_cfg = algo_cfg.tokenizer
        vocab_size = tok_cfg.vocab_size
        kekulize = tok_cfg.get("kekulize", True)
        min_frequency = tok_cfg.get("min_frequency", 10)
        workers = tok_cfg.get("workers", 16)

        print(cyan("Training PS principal-subgraph vocabulary..."))
        print(cyan("  Vocab size:"), vocab_size)
        print(cyan("  Kekulize:"), kekulize)
        print(cyan("  Min ring freq:"), min_frequency)
        print(cyan("  Workers:"), workers)
        print(cyan("  Corpus:"), str(corpus_path))

        save_dir = self.output_dir / "tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = save_dir / "vocab.txt"

        graph_bpe(
            fname=str(corpus_path),
            vocab_len=vocab_size,
            vocab_path=str(vocab_path),
            cpus=workers,
            kekulize=kekulize,
            min_frequency=min_frequency,
        )

        print(cyan("Saved PS vocab to:"), vocab_path)

        # Save a JSON index for raw PS subgraph vocabulary for inspection.
        ps_vocab_json_path = save_dir / "ps_vocab.json"
        ps_vocab_json: dict[str, int] = {}
        with open(vocab_path, "r", encoding="utf-8") as fin:
            lines = fin.read().strip().splitlines()
        for line in lines[1:]:
            token = line.split("\t", 1)[0]
            if token not in ps_vocab_json:
                ps_vocab_json[token] = len(ps_vocab_json)
        with open(ps_vocab_json_path, "w", encoding="utf-8") as fout:
            json.dump(ps_vocab_json, fout, ensure_ascii=False, indent=2)
        print(cyan("Saved PS vocab JSON to:"), ps_vocab_json_path)

        # Verify the vocab loads correctly and show example tokenizations
        tokenizer = Tokenizer(str(vocab_path))
        print(cyan("Loaded vocab size:"), len(tokenizer))

        # Build PS-fragSMILES corpus and token frequency table for HF vocab.
        print(cyan("Building PS-fragSMILES corpus and token frequencies..."))
        ps_frag_corpus_path = save_dir / "corpus_ps_fragsmiles.txt"
        token_counts: dict[str, int] = {}
        sample_ps_frag: list[str] = []

        batch_size = 1024
        batch_smiles: list[str] = []
        with open(corpus_path, "r", encoding="utf-8") as fin, open(
            ps_frag_corpus_path, "w", encoding="utf-8"
        ) as ps_out:
            for line in fin:
                smi = line.strip()
                if not smi:
                    continue
                batch_smiles.append(smi)
                if len(batch_smiles) < batch_size:
                    continue

                encoded_batch = encode_ps_fragsmiles_batch(batch_smiles, str(vocab_path))
                for encoded in encoded_batch:
                    ps_out.write(encoded + "\n")
                    if len(sample_ps_frag) < 5:
                        sample_ps_frag.append(encoded)
                    try:
                        fragments = [
                            token
                            for single_fragsmiles in encoded.split(";")
                            for token in chemicalgof.split(single_fragsmiles)
                            if token
                        ]
                        for tok in fragments:
                            token_counts[tok] = token_counts.get(tok, 0) + 1
                    except Exception:
                        pass
                batch_smiles = []

            if batch_smiles:
                encoded_batch = encode_ps_fragsmiles_batch(batch_smiles, str(vocab_path))
                for encoded in encoded_batch:
                    ps_out.write(encoded + "\n")
                    if len(sample_ps_frag) < 5:
                        sample_ps_frag.append(encoded)
                    try:
                        fragments = [
                            token
                            for single_fragsmiles in encoded.split(";")
                            for token in chemicalgof.split(single_fragsmiles)
                            if token
                        ]
                        for tok in fragments:
                            token_counts[tok] = token_counts.get(tok, 0) + 1
                    except Exception:
                        pass

        print(cyan("Saved PS-fragSMILES corpus to:"), ps_frag_corpus_path)
        print(cyan("Unique PS-fragSMILES tokens:"), len(token_counts))

        special_tokens = [
            tok_cfg.bos_token,
            tok_cfg.pad_token,
            tok_cfg.eos_token,
            tok_cfg.unk_token,
            tok_cfg.cls_token,
            tok_cfg.sep_token,
            tok_cfg.mask_token,
        ]

        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        hf_vocab: dict[str, int] = {}
        for token in special_tokens:
            if token not in hf_vocab:
                hf_vocab[token] = len(hf_vocab)

        for token, _count in sorted_tokens:
            if token in hf_vocab:
                continue
            if len(hf_vocab) >= vocab_size:
                break
            hf_vocab[token] = len(hf_vocab)

        hf_vocab_path = save_dir / "vocab.json"
        with open(hf_vocab_path, "w", encoding="utf-8") as fout:
            json.dump(hf_vocab, fout, ensure_ascii=False, indent=2)

        print(cyan("Saved HF tokenizer vocab to:"), hf_vocab_path)
        print(cyan("Final HF vocab size:"), len(hf_vocab))

        hf_tokenizer = PSFragSMILESTokenizer(
            vocab_file=str(hf_vocab_path),
            ps_vocab_file=str(vocab_path),
            unk_token=tok_cfg.unk_token,
            sep_token=tok_cfg.sep_token,
            pad_token=tok_cfg.pad_token,
            cls_token=tok_cfg.cls_token,
            mask_token=tok_cfg.mask_token,
            bos_token=tok_cfg.bos_token,
            eos_token=tok_cfg.eos_token,
        )
        hf_tokenizer.save_pretrained(str(save_dir))
        print(cyan("Saved PS-fragSMILES tokenizer to:"), save_dir)

        examples = corpus_path.read_text().splitlines()[:5]
        print(cyan("\nExample tokenizations:"))
        for i, smi in enumerate(examples):
            try:
                mol = tokenizer.tokenize(smi)
                subgraphs = (
                    mol.get_smis_subgraphs()
                    if not isinstance(mol, list)
                    else [sg for frag in mol for sg in frag.get_smis_subgraphs()]
                )
                tokens = [sg[0] for sg in subgraphs]
                print(f"  {smi}")
                print(f"    -> {tokens}")
                print(f"    -> {len(tokens)} subgraphs")

                if i < len(sample_ps_frag):
                    print(f"    -> PS-fragSMILES: {sample_ps_frag[i]}")
                hf_tokens = hf_tokenizer.tokenize(smi)
                hf_ids = hf_tokenizer.convert_tokens_to_ids(hf_tokens)
                print(f"    -> HF tokens: {hf_tokens}")
                print(f"    -> HF ids: {hf_ids}")
            except Exception as e:
                print(f"  {smi}  [tokenization failed: {e}]")
