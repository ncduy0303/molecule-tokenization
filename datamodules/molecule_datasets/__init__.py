from .pubchem10m_mlm_train import load_pubchem10m_mlm_train

from .prepared import load_prepared

# Add new molecule datasets here.
# Each loader takes (cfg, tokenizer) and returns a HF DatasetDict with train/val/test splits.
dataset_registry = {
    "external_tsdy_mlm": load_prepared,
    "pubchem10m_mlm_train": load_pubchem10m_mlm_train,
}

__all__ = ["dataset_registry"]
