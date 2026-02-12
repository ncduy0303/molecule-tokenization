"""
Load a fixed 2M-molecule subset of PubChem10M for tokenizer training.

This is separate from the 1.2M pretraining split used for RoBERTa MLM.
The indices are saved to data/pubchem10m_tokenizer_train/subset_indices.json
so that every tokenizer is trained on the exact same molecules.
"""

import json
import numpy as np
from pathlib import Path

from omegaconf import DictConfig


def load_pubchem10m_tokenizer_corpus(cfg: DictConfig):
    """
    Load a fixed subset of raw SMILES strings for tokenizer training.

    Args:
        cfg: Dataset config (configurations/dataset/pubchem10m_tokenizer_train.yaml).

    Returns:
        A list of SMILES strings.
    """
    from datasets import load_dataset

    smiles_col = cfg.smiles_column
    seed = cfg.get("seed", 42)
    corpus_size = cfg.get("corpus_size", 2_000_000)
    data_dir = Path(cfg.get("data_dir", "data/pubchem10m_tokenizer_train"))

    # ── Step 1: Get (or create) the fixed indices ───────────────────────
    indices_path = data_dir / "subset_indices.json"
    data_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Step 2: Load the selected SMILES strings ────────────────────────
    print(f"Loading {len(indices):,} SMILES strings for tokenizer training...")
    full_ds = load_dataset(cfg.hf_dataset, split="train")
    subset = full_ds.select(indices)
    smiles_list = subset[smiles_col]
    del full_ds, subset

    print(f"Loaded {len(smiles_list):,} SMILES strings")
    return smiles_list
