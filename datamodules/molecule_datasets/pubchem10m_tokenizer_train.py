"""
Load a fixed 2M-molecule subset of PubChem10M for tokenizer training.

This is separate from the 1.2M pretraining split used for RoBERTa MLM.
The indices are saved to data/pubchem10m_tokenizer_train/subset_indices.json
and the SMILES strings are written to data/pubchem10m_tokenizer_train/corpus.txt
(one per line) so that every tokenizer is trained on the exact same molecules.
The corpus.txt file can be passed directly to tokenizer training functions
that accept file paths (e.g. smirk train_gpe).
"""

import json
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
