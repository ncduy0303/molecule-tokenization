"""
MoleculeNet dataset loader for downstream classification tasks.

Downloads CSV files from DeepChem S3, cleans invalid SMILES, and applies
appropriate DeepChem 80/10/10 train/val/test splits:
  - HIV, BBBP: ScaffoldSplitter (group by Murcko scaffold)
  - ClinTox, Tox21: RandomSplitter
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from deepchem.data import NumpyDataset
from deepchem.splits import ScaffoldSplitter, RandomSplitter
from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.safe_utils import encode_safe_batch
from utils.fragsmiles_utils import encode_fragsmiles_batch

from rdkit import Chem
import logging

MOLNET_URLS = {
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}

# Standard DeepChem splitting rules
SCAFFOLD_SPLIT_DATASETS = {"hiv", "bbbp"}

# SMILES column name differs across datasets
SMILES_COLUMN = {}


def _get_smiles_col(dataset_name: str) -> str:
    return SMILES_COLUMN.get(dataset_name, "smiles")


def _get_target_columns(dataset_name: str, df: pd.DataFrame) -> list[str]:
    """Return the specific target column(s) requested for the dataset."""
    KNOWN_TARGETS = {
        "hiv": ["HIV_active"],
        "bbbp": ["p_np"],
        "clintox": ["CT_TOX"],
        "tox21": ["SR-p53"],
    }
    if dataset_name in KNOWN_TARGETS:
        return KNOWN_TARGETS[dataset_name]
    else:
        raise ValueError(f"Targets not configured for {dataset_name}")


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


def _deepchem_split(df: pd.DataFrame, smi_col: str, dataset_name: str, seed: int = 42):
    """
    Applies DeepChem's exact Scaffold or Random splitter.
    Uses the dummy X array to keep track of the original dataframe indices.
    """
    # X stores the dataframe indices so we can retrieve them after the split
    original_indices = df.index.values.reshape(-1, 1)
    smiles = np.asarray(df[smi_col].values, dtype=object)
    dummy_y = np.zeros(len(df))

    # DeepChem ScaffoldSplitter REQUIRES the smiles to be in the 'ids' parameter
    dc_dataset = NumpyDataset(X=original_indices, y=dummy_y, ids=smiles)

    if dataset_name in SCAFFOLD_SPLIT_DATASETS:
        print(cyan("  Applying DeepChem ScaffoldSplitter..."))
        splitter = ScaffoldSplitter()
    else:
        print(cyan("  Applying DeepChem RandomSplitter..."))
        splitter = RandomSplitter()

    # Perform the 80/10/10 split
    train_dc, valid_dc, test_dc = splitter.train_valid_test_split(
        dc_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
    )

    # Extract the original dataframe indices back out from the X arrays
    train_idx = train_dc.X.flatten().astype(int).tolist()
    val_idx = valid_dc.X.flatten().astype(int).tolist()
    test_idx = test_dc.X.flatten().astype(int).tolist()

    return train_idx, val_idx, test_idx


def load_molnet(cfg: DictConfig, tokenizer):
    """
    Load a MoleculeNet classification dataset, tokenize SMILES,
    and return a HF DatasetDict with train/validation/test splits.
    """
    from datasets import Dataset, DatasetDict

    dataset_name = cfg.molnet_name
    assert dataset_name in MOLNET_URLS, (
        f"Unknown MoleculeNet dataset: '{dataset_name}'. " f"Available: {list(MOLNET_URLS.keys())}"
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

    # Skip rows where target does not exist (i.e., Tox21 SR-p53 has some missing labels)
    print(cyan("  Before dropping rows with missing targets:"), len(df), "molecules")
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    print(cyan(f"  After dropping rows with missing targets: {len(df)} molecules remain"))

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
        # Run DeepChem split
        train_idx, val_idx, test_idx = _deepchem_split(df, smi_col, dataset_name, seed)

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

    # ── Step 3: Build/Load cached processed strings per split ───────────
    use_safe = cfg.get("use_safe", False)
    safe_slicer = cfg.get("safe_slicer", "brics")
    use_fragsmiles = cfg.get("use_fragsmiles", False)

    if use_safe and use_fragsmiles:
        raise ValueError("Only one of dataset.use_safe and dataset.use_fragsmiles can be true.")

    if use_safe:
        mode_prefix = "canon_safe"
        label = "SAFE"
        print(cyan("  SAFE encoding:"), f"enabled (slicer={safe_slicer})")
    elif use_fragsmiles:
        mode_prefix = "canon_fragsmiles"
        label = "fragSMILES"
        print(cyan("  fragSMILES encoding:"), "enabled")
    else:
        mode_prefix = "canon_smiles"
        label = "SMILES"

    cache_paths = {
        split: data_dir / f"{dataset_name}_{mode_prefix}_{split}_seed{seed}.txt"
        for split in ("train", "validation", "test")
    }
    all_cached = all(path.exists() for path in cache_paths.values())

    def process_smiles_list(smiles: list[str]) -> list[str]:
        if use_safe:
            return encode_safe_batch(smiles, slicer=safe_slicer)
        if use_fragsmiles:
            return encode_fragsmiles_batch(smiles)

        logging.getLogger("rdkit").setLevel(logging.CRITICAL)
        canon = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                # If RDKit fails to parse, just keep the original string
                canon.append(Chem.MolToSmiles(mol) if mol is not None else smi)
            except Exception:
                canon.append(smi)
        logging.getLogger("rdkit").setLevel(logging.WARNING)
        return canon

    index_map = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }

    split_smiles_cache: dict[str, list[str]] = {}
    if all_cached:
        print(cyan(f"  Loading cached {label} strings from"), str(data_dir))
        for split, path in cache_paths.items():
            lines = path.read_text().splitlines()
            split_smiles_cache[split] = lines
            print(cyan(f"    {split}:"), f"{len(lines):,} {label} strings")
    else:
        print(cyan(f"  Building cached {label} strings for MolNet splits..."))
        for split, indices in index_map.items():
            sub = df.iloc[indices].reset_index(drop=True)
            smiles = sub[smi_col].tolist()
            processed = process_smiles_list(smiles)
            split_smiles_cache[split] = processed
            cache_paths[split].write_text("\n".join(processed))
            print(cyan(f"    {split}:"), f"{len(processed):,} {label} strings")
        print(cyan("  Cached processed strings to"), f"{data_dir}/{dataset_name}_{mode_prefix}_*_seed{seed}.txt")

    def df_to_hf(split_name, indices):
        sub = df.iloc[indices].reset_index(drop=True)
        smiles = split_smiles_cache[split_name]

        # Build label vectors: replace NaN with -1 for masking
        labels = sub[target_cols].fillna(-1).values.astype(float).tolist()

        # Tokenize
        from apetokenizer.ape_tokenizer import APETokenizer

        is_ape = isinstance(tokenizer, APETokenizer)

        if is_ape:
            # APETokenizer does not support batched tokenization
            all_input_ids = []
            all_attention_mask = []
            for smi in smiles:
                enc = tokenizer(
                    smi,
                    max_length=max_length,
                    padding=False,
                )
                all_input_ids.append(enc["input_ids"])
                all_attention_mask.append(enc["attention_mask"])
            return Dataset.from_dict(
                {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "labels": labels,
                }
            )
        else:
            encodings = tokenizer(
                smiles,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
            )
            return Dataset.from_dict(
                {
                    "input_ids": encodings["input_ids"],
                    "attention_mask": encodings["attention_mask"],
                    "labels": labels,
                }
            )

    dataset = DatasetDict(
        {
            "train": df_to_hf("train", train_idx),
            "validation": df_to_hf("validation", val_idx),
            "test": df_to_hf("test", test_idx),
        }
    )

    return dataset
