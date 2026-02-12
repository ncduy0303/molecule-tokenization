from .qm9 import load_qm9
from .pubchem10m import load_pubchem10m

# Add new molecule datasets here.
# Each loader takes (cfg, tokenizer) and returns a HF DatasetDict with train/val/test splits.
dataset_registry = {
    "qm9": load_qm9,
    "pubchem10m": load_pubchem10m,
}

__all__ = ["dataset_registry"]
