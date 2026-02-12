"""
Load a fixed 2M-molecule subset of PubChem10M for tokenizer training.

This is separate from the 1.2M pretraining split used for RoBERTa MLM.
The indices are saved to data/pubchem10m_tokenizer_train/subset_indices.json
and the SMILES strings are written to data/pubchem10m_tokenizer_train/corpus.txt
(one per line) so that every tokenizer is trained on the exact same molecules.
The corpus.txt file can be passed directly to tokenizer training functions
that accept file paths (e.g. smirk train_gpe).

A word-count dictionary (SMILES substructures split by regex) is also cached
to data/pubchem10m_tokenizer_train/word_counts.json for PCATT training.
"""

import json
import re
import numpy as np
from pathlib import Path

from omegaconf import DictConfig


def load_pubchem10m_tokenizer_corpus(cfg: DictConfig):
    """
    Load a fixed subset of raw SMILES strings for tokenizer training.

    If a cached corpus.txt already exists on disk, loads from it directly
    (no HuggingFace download needed). Otherwise downloads, samples, and
    saves both subset_indices.json and corpus.txt.

    Args:
        cfg: Dataset config (configurations/dataset/pubchem10m_tokenizer_train.yaml).

    Returns:
        tuple: (smiles_list, corpus_path)
            smiles_list: list of SMILES strings
            corpus_path: Path to corpus.txt on disk
    """
    from datasets import load_dataset

    smiles_col = cfg.smiles_column
    seed = cfg.get("seed", 42)
    corpus_size = cfg.get("corpus_size", 2_000_000)
    data_dir = Path(cfg.get("data_dir", "data/pubchem10m_tokenizer_train"))

    data_dir.mkdir(parents=True, exist_ok=True)
    indices_path = data_dir / "subset_indices.json"
    corpus_path = data_dir / "corpus.txt"

    # ── Fast path: load from cached corpus.txt ──────────────────────────
    if corpus_path.exists():
        print(f"Loading cached tokenizer-training corpus from {corpus_path}")
        smiles_list = corpus_path.read_text().splitlines()
        print(f"Loaded {len(smiles_list):,} SMILES strings from cache")
        return smiles_list, corpus_path

    # ── Slow path: download, sample, and save ───────────────────────────
    if indices_path.exists():
        print(f"Loading fixed tokenizer-training indices from {indices_path}")
        with open(indices_path) as f:
            idx_info = json.load(f)
        indices = idx_info["indices"]
    else:
        print("Downloading PubChem10M to sample tokenizer-training indices...")
        full_ds = load_dataset(cfg.hf_dataset, split="train")
        n_total = len(full_ds)
        assert n_total >= corpus_size, (
            f"Dataset has {n_total} molecules but we need {corpus_size}"
        )

        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(n_total, size=corpus_size, replace=False).tolist())

        idx_info = {
            "seed": seed,
            "total_source": n_total,
            "corpus_size": corpus_size,
            "indices": indices,
        }
        with open(indices_path, "w") as f:
            json.dump(idx_info, f)
        print(f"Saved fixed tokenizer-training indices ({corpus_size:,} molecules) to {indices_path}")
        del full_ds

    # ── Load the selected SMILES strings ────────────────────────────────
    print(f"Loading {len(indices):,} SMILES strings for tokenizer training...")
    full_ds = load_dataset(cfg.hf_dataset, split="train")
    subset = full_ds.select(indices)
    smiles_list = subset[smiles_col]
    del full_ds, subset

    # ── Save corpus.txt ─────────────────────────────────────────────────
    corpus_path.write_text("\n".join(smiles_list))
    print(f"Saved corpus ({len(smiles_list):,} SMILES) to {corpus_path}")

    return smiles_list, corpus_path


def build_word_counts(cfg: DictConfig):
    """
    Build (or load cached) SMILES substructure word counts for PCATT training.

    Splits each SMILES string on structural elements using a regex pattern,
    then counts occurrences. The result is cached to word_counts.json.

    Args:
        cfg: Dataset config (configurations/dataset/pubchem10m_tokenizer_train.yaml).

    Returns:
        tuple: (word_count, longest_struct_len)
            word_count: dict mapping substructure string -> count
            longest_struct_len: length of the longest substructure seen
    """
    data_dir = Path(cfg.get("data_dir", "data/pubchem10m_tokenizer_train"))
    word_counts_path = data_dir / "word_counts.json"

    # ── Fast path: load from cache ──────────────────────────────────────
    if word_counts_path.exists():
        print(f"Loading cached word counts from {word_counts_path}")
        with open(word_counts_path) as f:
            cached = json.load(f)
        word_count = cached["word_count"]
        longest_struct_len = cached["longest_struct_len"]
        print(f"Loaded {len(word_count):,} unique substructures (longest: {longest_struct_len})")
        return word_count, longest_struct_len

    # ── Slow path: compute from corpus ──────────────────────────────────
    smiles_list, _ = load_pubchem10m_tokenizer_corpus(cfg)

    print(f"Building word counts from {len(smiles_list):,} SMILES strings...")
    word_count: dict[str, int] = {}
    longest_struct_len = 0

    for smi in smiles_list:
        structures = [s for s in re.split(r"(\.|%\d{2}|[\(\)]|[/\\]|\[.*?]|\d)", smi) if s]
        for struct in structures:
            if len(struct) > longest_struct_len:
                longest_struct_len = len(struct)
            if struct in word_count:
                word_count[struct] += 1
            else:
                word_count[struct] = 1

    # ── Save to disk ────────────────────────────────────────────────────
    cached = {
        "corpus_size": len(smiles_list),
        "unique_substructures": len(word_count),
        "longest_struct_len": longest_struct_len,
        "word_count": word_count,
    }
    with open(word_counts_path, "w") as f:
        json.dump(cached, f)
    print(f"Saved word counts ({len(word_count):,} unique substructures, "
          f"longest: {longest_struct_len}) to {word_counts_path}")

    return word_count, longest_struct_len
