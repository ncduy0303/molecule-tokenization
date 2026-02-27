from .pubchem10m_mlm_train import load_pubchem10m_mlm_train

# Add new molecule datasets here.
# Each loader takes (cfg, tokenizer) and returns a HF DatasetDict with train/val/test splits.
dataset_registry = {
    "pubchem10m_mlm_train": load_pubchem10m_mlm_train,
}

__all__ = ["dataset_registry"]
