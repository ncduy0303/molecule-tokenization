"""
PubChem10M molecule dataset loader for tokenizer benchmarking.

Loads from the HuggingFace dataset `mikemayuare/PubChem10M_SMILES_SELFIES`,
selects a fixed 1.2M-molecule subset (1M train / 100K val / 100K test),
canonicalizes SMILES strings using RDKit, tokenizes them,
and returns a HuggingFace DatasetDict.

The molecule indices are saved locally so that every tokenizer variant
trains on the exact same molecules.
"""

import json
import numpy as np
from pathlib import Path

from omegaconf import DictConfig

# Import RDKit for canonicalization
from rdkit import Chem
from rdkit import RDLogger

from utils.safe_utils import encode_safe_batch
from utils.fragsmiles_utils import encode_fragsmiles_batch


def load_pubchem10m_mlm_train(cfg: DictConfig, tokenizer):
    """
    Load, canonicalize, and tokenize a fixed subset of PubChem10M.

    Args:
        cfg: Dataset config from configurations/dataset/pubchem10m_mlm_train.yaml
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

    # ── Step 1: Get (or create) the fixed molecule indices ──────────────
    indices_path = data_dir / "subset_indices.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    needs_generation = True
    if indices_path.exists():
        with open(indices_path) as f:
            idx_info = json.load(f)

        # Check if the existing indices were strictly validated by RDKit
        if idx_info.get("rdkit_validated", False) and idx_info.get("seed") == seed:
            print(f"Loading fixed, RDKit-validated subset indices from {indices_path}")
            train_indices = idx_info["train"]
            val_indices = idx_info["validation"]
            test_indices = idx_info["test"]
            needs_generation = False
        else:
            print(f"Existing indices at {indices_path} are outdated or not RDKit-validated. Regenerating...")

    if needs_generation:
        # Download full dataset to determine total size & sample indices
        print("Downloading/Loading full PubChem10M to select valid indices (first time only)...")
        full_ds = load_dataset(
            cfg.hf_dataset,
            split="train",
        )

        n_total = len(full_ds)
        assert n_total >= total_subset, f"Dataset has {n_total} molecules but we need {total_subset}"

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(n_total).tolist()

        valid_indices = []
        batch_size = 100000  # Process in chunks to avoid loading all SMILES at once

        # Disable RDKit warnings to prevent spamming the console during search
        RDLogger.DisableLog("rdApp.*")  # type: ignore
        print(f"Scanning dataset for {total_subset} valid molecules...")

        for i in range(0, n_total, batch_size):
            batch_idx = shuffled_indices[i : i + batch_size]
            batch_smiles = full_ds.select(batch_idx)[smiles_col]

            for idx, smi in zip(batch_idx, batch_smiles):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        # Verify it can be successfully canonicalized
                        _ = Chem.MolToSmiles(mol)
                        valid_indices.append(idx)
                        if len(valid_indices) == total_subset:
                            break
                except Exception:
                    pass  # Skip this invalid molecule and continue

            print(f"  Found {len(valid_indices)} / {total_subset} valid molecules...")
            if len(valid_indices) == total_subset:
                break

        RDLogger.EnableLog("rdApp.*")  # type: ignore

        if len(valid_indices) < total_subset:
            raise RuntimeError(f"Only found {len(valid_indices)} valid molecules in the dataset!")

        train_indices = sorted(valid_indices[:train_size])
        val_indices = sorted(valid_indices[train_size : train_size + val_size])
        test_indices = sorted(valid_indices[train_size + val_size :])

        idx_info = {
            "seed": seed,
            "total_source": n_total,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "rdkit_validated": True,
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices,
        }
        with open(indices_path, "w") as f:
            json.dump(idx_info, f)
        print(f"Saved fixed subset indices ({total_subset:,} valid molecules) to {indices_path}")

        del full_ds  # free memory before re-selecting

    # ── Step 2: Load canonicalized (and optionally SAFE-encoded) SMILES ──
    # Cache canonical SMILES per split so we never re-download / re-canonicalize.
    use_safe = cfg.get("use_safe", False)
    safe_slicer = cfg.get("safe_slicer", "brics")
    use_fragsmiles = cfg.get("use_fragsmiles", False)
    if use_safe and use_fragsmiles:
        raise ValueError("Only one of dataset.use_safe and dataset.use_fragsmiles can be true.")

    if use_safe:
        prefix = "canon_safe"
        label = "SAFE"
    elif use_fragsmiles:
        prefix = "canon_fragsmiles"
        label = "fragSMILES"
    else:
        prefix = "canon_smiles"
        label = "SMILES"
    cache_paths = {split: data_dir / f"{prefix}_{split}.txt" for split in ("train", "validation", "test")}
    all_cached = all(p.exists() for p in cache_paths.values())

    if all_cached:
        # ── Fast path: load processed SMILES from text files ────────────
        from datasets import Dataset, DatasetDict

        print(f"Loading cached {label} strings from {data_dir}...")
        split_smiles: dict[str, list[str]] = {}
        for split, path in cache_paths.items():
            lines = path.read_text().splitlines()
            split_smiles[split] = lines
            print(f"  {split}: {len(lines):,} {label} strings")

        # Build a DatasetDict with just the SMILES column (tokenized below)
        splits = DatasetDict(
            {split: Dataset.from_dict({smiles_col: smiles_list}) for split, smiles_list in split_smiles.items()}
        )
    else:
        # ── Slow path: download, canonicalize, optionally SAFE-encode ───
        print("Loading PubChem10M subset splits...")
        full_ds = load_dataset(cfg.hf_dataset, split="train")

        splits = DatasetDict(
            {
                "train": full_ds.select(train_indices),  # type: ignore
                "validation": full_ds.select(val_indices),  # type: ignore
                "test": full_ds.select(test_indices),  # type: ignore
            }
        )
        del full_ds

        # Keep only the SMILES column
        cols_to_remove = [c for c in splits["train"].column_names if c != smiles_col]
        if cols_to_remove:
            splits = splits.remove_columns(cols_to_remove)

        if use_safe:
            # ── Step 2a: SAFE encode (produces canonical output directly) ─
            print(f"Encoding SMILES to SAFE (slicer={safe_slicer})...")

            def safe_encode_batch_fn(examples):
                return {smiles_col: encode_safe_batch(examples[smiles_col], slicer=safe_slicer)}

            splits = splits.map(
                safe_encode_batch_fn,
                batched=True,
                desc="Encoding SMILES to SAFE",
            )
        elif use_fragsmiles:
            # ── Step 2a: Convert SMILES to fragSMILES once and cache ────
            print("Encoding SMILES to fragSMILES...")

            def fragsmiles_encode_batch_fn(examples):
                return {smiles_col: encode_fragsmiles_batch(examples[smiles_col])}

            splits = splits.map(
                fragsmiles_encode_batch_fn,
                batched=True,
                desc="Encoding SMILES to fragSMILES",
            )
        else:
            # ── Step 2a: Canonicalize SMILES with RDKit ─────────────────
            RDLogger.DisableLog("rdApp.*")  # type: ignore

            def canonicalize_batch(examples):
                canon_smiles = []
                for smi in examples[smiles_col]:
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            canon_smiles.append(Chem.MolToSmiles(mol))
                        else:
                            canon_smiles.append(smi)
                    except Exception:
                        canon_smiles.append(smi)
                return {smiles_col: canon_smiles}

            splits = splits.map(
                canonicalize_batch,
                batched=True,
                desc="Canonicalizing with RDKit",
            )

            RDLogger.EnableLog("rdApp.*")  # type: ignore

        # Cache processed SMILES/SAFE strings to disk
        for split in ("train", "validation", "test"):
            cache_paths[split].write_text("\n".join(splits[split][smiles_col]))
        print(f"Cached {label} strings to {data_dir}/{prefix}_*.txt")

    # ── Step 3: Tokenize ────────────────────────────────────────────────
    from apetokenizer.ape_tokenizer import APETokenizer

    is_ape = isinstance(tokenizer, APETokenizer)

    def tokenize_fn(examples):
        # Do not pass `truncation` to APETokenizer
        if is_ape:
            return tokenizer(
                examples[smiles_col],
                max_length=max_length,
                padding=False,
            )
        return tokenizer(
            examples[smiles_col],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    splits = splits.map(
        tokenize_fn,
        batched=not is_ape,  # APETokenizer does not support batched tokenization
        remove_columns=[smiles_col],
        desc="Tokenizing",
    )

    # ── Step 4: Cache to disk ───────────────────────────────────────────
    # cache_path.mkdir(parents=True, exist_ok=True)
    # splits.save_to_disk(str(cache_path))
    # print(f"Cached tokenized dataset to {cache_path}")

    return splits
