"""
PubChem10M molecule dataset loader for tokenizer benchmarking.

Loads from the HuggingFace dataset `mikemayuare/PubChem10M_SMILES_SELFIES`,
selects a fixed 1.2M-molecule subset (1M train / 100K val / 100K test),
tokenizes SMILES strings, and returns a HuggingFace DatasetDict.

The molecule indices are saved locally so that every tokenizer variant
trains on the exact same molecules.  Tokenized datasets are cached per
tokenizer in data/pubchem10m/processed/<tokenizer_name>/.
"""

import json
import numpy as np
from pathlib import Path

from omegaconf import DictConfig


def load_pubchem10m(cfg: DictConfig, tokenizer):
    """
    Load and tokenize a fixed subset of PubChem10M.

    Args:
        cfg: Dataset config from configurations/dataset/pubchem10m.yaml.
        tokenizer: A HuggingFace-compatible tokenizer.

    Returns:
        A datasets.DatasetDict with "train", "validation", and "test" splits,
        each containing tokenized columns (input_ids, attention_mask).
    """
    from datasets import DatasetDict, load_dataset

    smiles_col = cfg.smiles_column
    max_length = cfg.max_length
    seed = cfg.seed
    data_dir = Path(cfg.data_dir)

    train_size = cfg.train_size
    val_size = cfg.val_size
    test_size = cfg.test_size
    total_subset = train_size + val_size + test_size

    # Determine tokenizer name for cache key
    # tok_name = type(tokenizer).__name__.lower()
    # cache_path = data_dir / "processed" / tok_name

    # If tokenized dataset already cached, load from disk
    # if cache_path.exists():
    #     print(f"Loading cached tokenized dataset from {cache_path}")
    #     return DatasetDict.load_from_disk(str(cache_path))

    # ── Step 1: Get (or create) the fixed molecule indices ──────────────
    indices_path = data_dir / "subset_indices.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    if indices_path.exists():
        print(f"Loading fixed subset indices from {indices_path}")
        with open(indices_path) as f:
            idx_info = json.load(f)
        train_indices = idx_info["train"]
        val_indices = idx_info["validation"]
        test_indices = idx_info["test"]
    else:
        # Download full dataset to determine total size & sample indices
        print("Downloading PubChem10M from HuggingFace Hub (first time only)...")
        full_ds = load_dataset(
            cfg.hf_dataset,
            split="train",
        )

        # [DEBUG] Filter datasets to only take entries with SMILES string of length < max_length
        # full_ds = full_ds.filter(lambda x: len(x[smiles_col]) < max_length)
        
        n_total = len(full_ds)
        assert n_total >= total_subset, (
            f"Dataset has {n_total} molecules but we need {total_subset}"
        )

        rng = np.random.default_rng(seed)
        all_indices = rng.choice(n_total, size=total_subset, replace=False).tolist()

        train_indices = sorted(all_indices[:train_size])
        val_indices = sorted(all_indices[train_size : train_size + val_size])
        test_indices = sorted(all_indices[train_size + val_size :])

        idx_info = {
            "seed": seed,
            "total_source": n_total,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices,
        }
        with open(indices_path, "w") as f:
            json.dump(idx_info, f)
        print(f"Saved fixed subset indices ({total_subset:,} molecules) to {indices_path}")

        del full_ds  # free memory before re-selecting

    # ── Step 2: Load only the selected rows ─────────────────────────────
    print("Loading PubChem10M subset splits...")
    full_ds = load_dataset(cfg.hf_dataset, split="train")

    splits = DatasetDict(
        {
            "train": full_ds.select(train_indices),
            "validation": full_ds.select(val_indices),
            "test": full_ds.select(test_indices),
        }
    )
    del full_ds

    # Keep only the SMILES column
    cols_to_remove = [c for c in splits["train"].column_names if c != smiles_col]
    if cols_to_remove:
        splits = splits.remove_columns(cols_to_remove)

    # ── Step 3: Tokenize ────────────────────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(
            examples[smiles_col],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollator at training time
        )

    splits = splits.map(
        tokenize_fn,
        batched=True,
        remove_columns=[smiles_col],
        desc="Tokenizing",
    )

    # ── Step 4: Cache to disk ───────────────────────────────────────────
    # cache_path.mkdir(parents=True, exist_ok=True)
    # splits.save_to_disk(str(cache_path))
    # print(f"Cached tokenized dataset to {cache_path}")

    return splits
