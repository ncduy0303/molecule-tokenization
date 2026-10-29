"""Read-only loader for audited, tokenized Arrow datasets. Never encodes on a GPU job."""

import json
from pathlib import Path
from datasets import load_from_disk
from utils.cache_utils import file_digest
from utils.external_tsdy_tokenizer import tokenizer_identity


def load_prepared(cfg, tokenizer):
    root = Path(cfg.prepared_path)
    ready = json.loads((root / "ready.json").read_text())
    if not ready.get("passed") or ready["tokenizer"] != tokenizer_identity(tokenizer):
        raise ValueError("Prepared dataset is incomplete or belongs to a different tokenizer")
    if ready["task"] != cfg.prepared_task or ready["max_length"] != cfg.max_length:
        raise ValueError("Prepared task or model context differs")
    if ready["smoke"] and not cfg.get("allow_smoke", False):
        raise ValueError("Smoke-only data cannot be used in a full training run")
    for file, checksum in ready["files"].items():
        if file_digest(root / file) != checksum:
            raise ValueError(f"Prepared artifact checksum mismatch: {root / file}")
    data = load_from_disk(str(root / "dataset"))
    if {k: len(v) for k, v in data.items()} != ready["split_sizes"]:
        raise ValueError("Prepared split sizes differ from audit")
    return data


def record_prepared_input(cfg, output):
    """Small immutable-in-practice snapshot alongside model metrics for analysis."""
    root = Path(cfg.prepared_path)
    payload = {name: json.loads((root / name).read_text()) for name in ("manifest.json", "ready.json")}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "prepared_input.json").write_text(json.dumps(payload, indent=2) + "\n")
