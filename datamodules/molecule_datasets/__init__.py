from .qm9 import load_qm9

# Add new molecule datasets here.
# Each loader takes (cfg, tokenizer) and returns a HF DatasetDict with "train"/"test" splits.
dataset_registry = {
    "qm9": load_qm9,
}

__all__ = ["dataset_registry"]
