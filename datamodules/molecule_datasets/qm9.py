"""
QM9 molecule dataset loader for tokenizer benchmarking.

Downloads MoleculeNet's QM9 CSV, tokenizes SMILES strings,
and returns a HuggingFace DatasetDict with train/test splits.

Raw CSV is cached in data/qm9/raw/, tokenized dataset is cached in
data/qm9/processed/<tokenizer_name>/ so subsequent runs skip 
downloading and re-tokenizing.
"""

import os
from pathlib import Path

from omegaconf import DictConfig


def load_qm9(cfg: DictConfig, tokenizer):
    """
    Load and tokenize the QM9 dataset.

    Args:
        cfg: Dataset config from configurations/dataset/qm9.yaml.
        tokenizer: A HuggingFace-compatible tokenizer (smirk, etc.).

    Returns:
        A datasets.DatasetDict with "train" and "test" splits,
        each containing tokenized columns (input_ids, attention_mask).
    """
    from datasets import load_dataset

    smiles_col = cfg.smiles_column
    max_length = cfg.get("max_length", 128)
    test_size = cfg.get("test_size", 0.2)
    seed = cfg.get("seed", 42)
    data_dir = Path(cfg.get("data_dir", "data/qm9"))

    # Determine tokenizer name for cache key
    tok_name = type(tokenizer).__name__.lower()
    cache_path = data_dir / "processed" / tok_name

    # If tokenized dataset already cached, load from disk
    if cache_path.exists():
        from datasets import DatasetDict

        print(f"Loading cached tokenized dataset from {cache_path}")
        return DatasetDict.load_from_disk(str(cache_path))

    # Download raw CSV to data/qm9/raw/ if not already there
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_csv = raw_dir / "qm9.csv"

    if local_csv.exists():
        data_source = str(local_csv)
    else:
        # Download and save locally for future runs
        import urllib.request

        print(f"Downloading QM9 to {local_csv}...")
        urllib.request.urlretrieve(cfg.data_url, str(local_csv))
        data_source = str(local_csv)

    # Load raw CSV
    dataset = load_dataset("csv", data_files=[data_source])["train"]

    # Keep only the SMILES column
    dataset = dataset.select_columns([smiles_col])

    # Train/test split
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples[smiles_col],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollator at training time
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[smiles_col],
        desc="Tokenizing",
    )

    # Cache to disk
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    print(f"Cached tokenized dataset to {cache_path}")

    return dataset
