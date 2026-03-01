"""
Load a fixed 2M-molecule subset of PubChem10M for tokenizer training.

This is separate from the 1.2M pretraining split used for RoBERTa MLM.
The indices are saved to data/pubchem10m_tokenizer_train/subset_indices.json
and the canonical SMILES strings are written to data/pubchem10m_tokenizer_train/corpus.txt
(one per line) so that every tokenizer is trained on the exact same molecules.
The corpus.txt file can be passed directly to tokenizer training functions
that accept file paths (e.g. smirk train_gpe).

A word-count dictionary is also cached to data/pubchem10m_tokenizer_train for PCATT training.
"""

import json
import re
import numpy as np
from pathlib import Path

from omegaconf import DictConfig

# Import RDKit for validation and canonicalization
from rdkit import Chem
from rdkit import RDLogger


def load_pubchem10m_tokenizer_corpus(cfg: DictConfig):
    """
    Load a fixed subset of canonical SMILES strings for tokenizer training.

    If a cached corpus.txt already exists on disk, loads from it directly
    (no HuggingFace download needed). Otherwise downloads, scans for valid
    molecules, canonicalizes them, and saves both subset_indices.json and corpus.txt.

    Args:
        cfg: Dataset config (configurations/dataset/pubchem10m_tokenizer_train.yaml).

    Returns:
        tuple: (smiles_list, corpus_path)
            smiles_list: list of canonical SMILES strings
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
        print(f"Loaded {len(smiles_list):,} canonical SMILES strings from cache")
        return smiles_list, corpus_path

    # ── Slow path: download, sample valid indices, and save ─────────────
    needs_generation = True
    if indices_path.exists():
        with open(indices_path) as f:
            idx_info = json.load(f)

        # Check if the existing indices were strictly validated by RDKit
        if idx_info.get("rdkit_validated", False) and idx_info.get("seed") == seed:
            print(f"Loading fixed, RDKit-validated tokenizer-training indices from {indices_path}")
            indices = idx_info["indices"]
            needs_generation = False
        else:
            print(f"Existing indices at {indices_path} are outdated or not RDKit-validated. Regenerating...")

    if needs_generation:
        print("Downloading/Loading full PubChem10M to select valid indices (first time only)...")
        full_ds = load_dataset(cfg.hf_dataset, split="train")
        n_total = len(full_ds)
        assert n_total >= corpus_size, f"Dataset has {n_total} molecules but we need {corpus_size}"

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(n_total).tolist()

        valid_indices = []
        batch_size = 100000  # Process in chunks to avoid loading all SMILES at once

        # Disable RDKit warnings to prevent spamming the console during search
        RDLogger.DisableLog("rdApp.*")  # type: ignore
        print(f"Scanning dataset for {corpus_size:,} valid molecules...")

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
                        if len(valid_indices) == corpus_size:
                            break
                except Exception:
                    pass  # Skip invalid molecule

            print(f"  Found {len(valid_indices):,} / {corpus_size:,} valid molecules...")
            if len(valid_indices) == corpus_size:
                break

        RDLogger.EnableLog("rdApp.*")  # type: ignore

        if len(valid_indices) < corpus_size:
            raise RuntimeError(f"Only found {len(valid_indices):,} valid molecules in the dataset!")

        indices = sorted(valid_indices)

        idx_info = {
            "seed": seed,
            "total_source": n_total,
            "corpus_size": corpus_size,
            "rdkit_validated": True,
            "indices": indices,
        }
        with open(indices_path, "w") as f:
            json.dump(idx_info, f)
        print(f"Saved fixed tokenizer-training indices ({corpus_size:,} valid molecules) to {indices_path}")
        del full_ds

    # ── Load and canonicalize the selected SMILES strings ───────────────
    print(f"Loading and canonicalizing {len(indices):,} SMILES strings for tokenizer training...")  # type: ignore
    full_ds = load_dataset(cfg.hf_dataset, split="train")
    subset = full_ds.select(indices)  # type: ignore
    raw_smiles_list = subset[smiles_col]
    del full_ds, subset

    # Canonicalize the extracted dataset
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    canonical_smiles_list = []

    for smi in raw_smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical_smiles_list.append(Chem.MolToSmiles(mol))
            else:
                # Should be unreachable due to index validation, but kept for safety
                canonical_smiles_list.append(smi)
        except Exception:
            canonical_smiles_list.append(smi)

    RDLogger.EnableLog("rdApp.*")  # type: ignore

    # ── Save corpus.txt ─────────────────────────────────────────────────
    corpus_path.write_text("\n".join(canonical_smiles_list))
    print(f"Saved canonical corpus ({len(canonical_smiles_list):,} SMILES) to {corpus_path}")

    return canonical_smiles_list, corpus_path


def build_word_counts(cfg: DictConfig):
    """
    Build (or load cached) SMILES word counts for PCATT training.

    Splits each canonical SMILES string on structural elements using a regex pattern,
    then counts occurrences. The result is cached to word_counts.json.

    Args:
        cfg: Dataset config (configurations/dataset/pubchem10m_tokenizer_train.yaml).

    Returns:
        tuple: (word_count, longest_word_len)
            word_count: dict mapping word to count
            longest_word_len: length of the longest word seen
    """
    data_dir = Path(cfg.get("data_dir", "data/pubchem10m_tokenizer_train"))
    pretokenizer = cfg.get("pretokenizer", None)
    if pretokenizer is None:
        word_counts_path = data_dir / "word_counts.json"
    elif pretokenizer in ["atom_split", "structure_split"]:
        word_counts_path = data_dir / f"{pretokenizer}_word_counts.json"
    else:
        raise ValueError(f"Unknown pretokenizer: {pretokenizer}")

    # ── Fast path: load from cache ──────────────────────────────────────
    if word_counts_path.exists():
        print(f"Loading cached word counts from {word_counts_path}")
        with open(word_counts_path) as f:
            cached = json.load(f)
        word_count = cached["word_count"]
        longest_word_len = cached["longest_word_len"]
        print(f"Loaded {len(word_count):,} unique words (longest: {longest_word_len})")
        return word_count, longest_word_len

    # ── Slow path: compute from canonical corpus ────────────────────────
    smiles_list, _ = load_pubchem10m_tokenizer_corpus(cfg)

    print(f"Building word counts from {len(smiles_list):,} canonical SMILES strings...")
    word_count: dict[str, int] = {}
    longest_word_len = 0

    for smi in smiles_list:
        words = [smi]
        if pretokenizer == "atom_split":
            # https://github.com/datamol-io/safe/blob/main/safe/tokenizer.py#50
            words = [
                s
                for s in re.findall(
                    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])",
                    smi,
                )
                if s
            ]
        elif pretokenizer == "structure_split":
            # https://github.com/BattModels/smirk/blob/main/src/pre_tokenizers/split_smiles.rs#L40
            words = [s for s in re.split(r"(\.|%\d{2}|[\(\)]|[/\\]|\[.*?]|\d)", smi) if s]
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
                if len(word) > longest_word_len:
                    longest_word_len = len(word)

    # ── Save to disk ────────────────────────────────────────────────────
    cached = {
        "corpus_size": len(smiles_list),
        "unique_words": len(word_count),
        "longest_word_len": longest_word_len,
        "word_count": word_count,
    }
    with open(word_counts_path, "w") as f:
        json.dump(cached, f)
    print(
        f"Saved word counts ({len(word_count):,} unique words, " f"longest: {longest_word_len}) to {word_counts_path}"
    )

    return word_count, longest_word_len


def build_smirk_pcatt_word_counts(cfg: DictConfig):
    """
    Build (or load cached) Smirk-adapted word counts for Smirk-PCATT training.

    Uses the Smirk tokenizer to pre-tokenize each SMILES string (optionally
    after structure_split pre-tokenization), then encodes the resulting glyphs
    into raw bytes via SmirkPCATTAdapter. The byte-level word counts are
    cached to smirk_pcatt_word_counts.json (or smirk_pcatt_structure_split_word_counts.json).

    Args:
        cfg: Dataset config (configurations/dataset/pubchem10m_tokenizer_train.yaml).

    Returns:
        tuple: (word_count, longest_word_len, adapter)
            word_count: dict mapping bytes -> count
            longest_word_len: length of the longest byte-word seen
            adapter: SmirkPCATTAdapter instance for decoding
    """
    from smirk import SmirkTokenizerFast
    from utils.smirk_pcatt_adapter import SmirkPCATTAdapter

    data_dir = Path(cfg.get("data_dir", "data/pubchem10m_tokenizer_train"))
    pretokenizer = cfg.get("pretokenizer", None)

    if pretokenizer is None:
        word_counts_path = data_dir / "smirk_pcatt_word_counts.json"
    elif pretokenizer in ["atom_split", "structure_split"]:
        word_counts_path = data_dir / f"smirk_pcatt_{pretokenizer}_word_counts.json"
    else:
        raise ValueError(f"Unknown pretokenizer: {pretokenizer}")

    # ── Build the Smirk tokenizer and adapter ───────────────────────────
    smirk_tokenizer = SmirkTokenizerFast()
    smirk_vocab = [token for token, _ in sorted(smirk_tokenizer.get_vocab().items(), key=lambda item: item[1])]
    adapter = SmirkPCATTAdapter(smirk_vocab)

    # ── Fast path: load from cache ──────────────────────────────────────
    if word_counts_path.exists():
        print(f"Loading cached Smirk-PCATT word counts from {word_counts_path}")
        with open(word_counts_path) as f:
            cached = json.load(f)
        # Restore bytes keys from latin-1 encoded strings
        word_count: dict[bytes, int] = {k.encode("latin-1"): v for k, v in cached["word_count"].items()}
        longest_word_len = cached["longest_word_len"]
        print(f"Loaded {len(word_count):,} unique words (longest: {longest_word_len})")
        return word_count, longest_word_len, adapter

    # ── Slow path: compute from canonical corpus ────────────────────────
    smiles_list, _ = load_pubchem10m_tokenizer_corpus(cfg)

    print(f"Building Smirk-PCATT word counts from {len(smiles_list):,} canonical SMILES strings...")
    word_count = {}
    longest_word_len = 0

    for smi in smiles_list:
        parts = [smi]
        if pretokenizer == "atom_split":
            # https://github.com/datamol-io/safe/blob/main/safe/tokenizer.py#50
            parts = [
                s
                for s in re.findall(
                    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])",
                    smi,
                )
                if s
            ]
        elif pretokenizer == "structure_split":
            # https://github.com/BattModels/smirk/blob/main/src/pre_tokenizers/split_smiles.rs#L40
            parts = [s for s in re.split(r"(\.|%\d{2}|[\(\)]|[/\\]|\[.*?]|\d)", smi) if s]

        for part in parts:
            smirk_tokens = smirk_tokenizer.tokenize(part)
            encoded = adapter.encode_for_pcatt(smirk_tokens)
            if encoded in word_count:
                word_count[encoded] += 1
            else:
                word_count[encoded] = 1
            if len(encoded) > longest_word_len:
                longest_word_len = len(encoded)

    # ── Save to disk (encode bytes keys as latin-1 for JSON) ────────────
    serializable_word_count = {k.decode("latin-1"): v for k, v in word_count.items()}
    cached_data = {
        "corpus_size": len(smiles_list),
        "unique_words": len(word_count),
        "longest_word_len": longest_word_len,
        "word_count": serializable_word_count,
    }
    with open(word_counts_path, "w") as f:
        json.dump(cached_data, f)
    print(
        f"Saved Smirk-PCATT word counts ({len(word_count):,} unique words, "
        f"longest: {longest_word_len}) to {word_counts_path}"
    )

    return word_count, longest_word_len, adapter
