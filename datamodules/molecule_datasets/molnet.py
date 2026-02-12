"""
MoleculeNet dataset loader for downstream classification tasks.

Downloads CSV files from DeepChem S3 and applies appropriate train/val/test
splits:
  - HIV, BBBP, BACE: scaffold split (group by Murcko scaffold)
  - All others: random 80/10/10 split

Reference: loader.py and utils.py from TokenizerStats.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from omegaconf import DictConfig

from utils.print_utils import cyan


MOLNET_URLS = {
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz",
    "lipo": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
    "sider": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "muv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "toxcast": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
}

# Datasets that use scaffold split; all others use random split
SCAFFOLD_SPLIT_DATASETS = {"hiv", "bbbp", "bace"}

# SMILES column name differs across datasets
SMILES_COLUMN = {
    "bace": "mol",
}


def _get_smiles_col(dataset_name: str) -> str:
    return SMILES_COLUMN.get(dataset_name, "smiles")


def _get_target_columns(dataset_name: str, df: pd.DataFrame) -> list[str]:
    """Return the numeric target column(s) for a dataset."""
    # Explicit target columns for datasets with non-numeric metadata
    KNOWN_TARGETS = {
        "hiv": ["HIV_active"],
        "bbbp": ["p_np"],
        "bace": ["Class"],
        "esol": ["measured log solubility in mols per litre"],
        "freesolv": ["expt"],
        "lipo": ["exp"],
    }
    if dataset_name in KNOWN_TARGETS:
        return KNOWN_TARGETS[dataset_name]

    # For tox21, sider, clintox, etc.: use all numeric columns except SMILES
    smi_col = _get_smiles_col(dataset_name)
    exclude = {smi_col, "scaffold", "mol_id", "Unnamed: 0"}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _download_csv(dataset_name: str, data_dir: Path) -> pd.DataFrame:
    """Download and cache a MoleculeNet CSV file."""
    csv_path = data_dir / f"{dataset_name}.csv"
    if csv_path.exists():
        print(cyan(f"Loading cached {dataset_name} from"), str(csv_path))
        return pd.read_csv(csv_path)

    url = MOLNET_URLS[dataset_name]
    print(cyan(f"Downloading {dataset_name} from"), url)
    data_dir.mkdir(parents=True, exist_ok=True)

    # pandas handles .csv.gz automatically
    df = pd.read_csv(url)
    df.to_csv(csv_path, index=False)
    print(cyan(f"  Saved to"), str(csv_path), f"({len(df)} molecules)")
    return df


def _scaffold_split(df: pd.DataFrame, smi_col: str, seed: int = 42):
    """Scaffold-based train/val/test split (80/10/10) using Murcko scaffolds."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    from sklearn.model_selection import GroupShuffleSplit

    def scaffold_hash(smi: str) -> str:
        try:
            return MurckoScaffoldSmiles(smi)
        except Exception:
            return smi

    scaffolds = df[smi_col].apply(scaffold_hash).values

    # First split: 80% train, 20% other
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, other_idx = next(gss1.split(df.index, groups=scaffolds))

    # Second split: 50/50 of "other" -> 10% val, 10% test
    other_scaffolds = scaffolds[other_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx_rel, test_idx_rel = next(gss2.split(other_idx, groups=other_scaffolds))

    val_idx = other_idx[val_idx_rel]
    test_idx = other_idx[test_idx_rel]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _random_split(df: pd.DataFrame, seed: int = 42):
    """Random train/val/test split (80/10/10)."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    train_idx, other_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    val_idx, test_idx = train_test_split(other_idx, test_size=0.5, random_state=seed)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def load_molnet(cfg: DictConfig, tokenizer):
    """
    Load a MoleculeNet classification dataset, tokenize SMILES,
    and return a HF DatasetDict with train/validation/test splits.

    Each example has:
      - input_ids, attention_mask  (tokenized SMILES)
      - labels  (float tensor of targets, NaN -> -1 for masking)

    Args:
        cfg: Dataset config. Must have:
            - molnet_name: str  (e.g. "hiv", "bbbp", "tox21")
        tokenizer: HuggingFace tokenizer.

    Returns:
        datasets.DatasetDict with train/validation/test splits.
    """
    from datasets import Dataset, DatasetDict

    dataset_name = cfg.molnet_name
    assert dataset_name in MOLNET_URLS, (
        f"Unknown MoleculeNet dataset: '{dataset_name}'. "
        f"Available: {list(MOLNET_URLS.keys())}"
    )

    data_dir = Path(cfg.data_dir)
    seed = cfg.seed
    max_length = cfg.max_length

    # ── Step 1: Download CSV ────────────────────────────────────────────
    df = _download_csv(dataset_name, data_dir)

    smi_col = _get_smiles_col(dataset_name)
    target_cols = _get_target_columns(dataset_name, df)

    print(cyan("  SMILES column:"), smi_col)
    print(cyan("  Target columns:"), target_cols)
    print(cyan("  Num targets:"), len(target_cols))

    # ── Step 2: Get (or load cached) split indices ──────────────────────
    split_file = data_dir / f"{dataset_name}_split_indices.json"

    if split_file.exists():
        print(cyan("  Loading cached split indices from"), str(split_file))
        with open(split_file) as f:
            idx_info = json.load(f)
        train_idx = idx_info["train"]
        val_idx = idx_info["validation"]
        test_idx = idx_info["test"]
    else:
        if dataset_name in SCAFFOLD_SPLIT_DATASETS:
            print(cyan("  Applying scaffold split..."))
            train_idx, val_idx, test_idx = _scaffold_split(df, smi_col, seed)
        else:
            print(cyan("  Applying random 80/10/10 split..."))
            train_idx, val_idx, test_idx = _random_split(df, seed)

        idx_info = {
            "seed": seed,
            "split_type": "scaffold" if dataset_name in SCAFFOLD_SPLIT_DATASETS else "random",
            "total": len(df),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        }
        with open(split_file, "w") as f:
            json.dump(idx_info, f)
        print(cyan("  Saved split indices to"), str(split_file))

    print(cyan("  Train:"), len(train_idx), "Val:", len(val_idx), "Test:", len(test_idx))

    # ── Step 3: Build per-split DataFrames ──────────────────────────────
    def df_to_hf(indices):
        sub = df.iloc[indices].reset_index(drop=True)
        smiles = sub[smi_col].tolist()
        # Build label vectors: replace NaN with -1 for masking
        labels = sub[target_cols].fillna(-1).values.astype(float).tolist()

        # Tokenize
        from apetokenizer.ape_tokenizer import APETokenizer
        is_ape = isinstance(tokenizer, APETokenizer)
        # Do not pass `truncation` to APETokenizer
        if is_ape:
            encodings = tokenizer(
                smiles,
                max_length=max_length,
                padding=False,
            )
        else:
            encodings = tokenizer(
                smiles,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })

    dataset = DatasetDict({
        "train": df_to_hf(train_idx),
        "validation": df_to_hf(val_idx),
        "test": df_to_hf(test_idx),
    })

    return dataset
