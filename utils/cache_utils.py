"""Validated cache provenance. Existing incompatible artifacts are never overwritten."""

import hashlib
import json
from pathlib import Path

CACHE_VERSION = 1


class CacheMismatchError(ValueError):
    pass


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metadata_path(path):
    return Path(str(path) + ".meta.json")


def cache_error(path, reason):
    return CacheMismatchError(
        f"Cache {path}: {reason}. Preserve this cache for old runs and use a new "
        "dataset.data_dir (or explicitly move the old artifact and its .meta.json aside)."
    )


def read_metadata(path):
    try:
        metadata = json.loads(metadata_path(path).read_text())
        if metadata["version"] != CACHE_VERSION or metadata["sha256"] != file_digest(path):
            raise ValueError("version or checksum mismatch")
        return metadata["identity"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise cache_error(path, "missing, outdated, or invalid provenance/checksum") from exc


def cache_matches(path, identity):
    path = Path(path)
    if not path.exists():
        return False
    if read_metadata(path) != identity:
        raise cache_error(path, "settings or upstream data changed")
    return True


def record_cache(path, identity):
    metadata_path(path).write_text(
        json.dumps(
            {"version": CACHE_VERSION, "identity": identity, "sha256": file_digest(path)}, sort_keys=True, indent=2
        )
    )


def source_identity(cfg):
    return {
        "dataset": cfg.hf_dataset,
        "revision": cfg.get("hf_revision", None),
        "split": "train",
        "smiles_column": cfg.smiles_column,
    }


def representation_identity(cfg):
    active = [key for key in ("use_safe", "use_fragsmiles", "use_tsmiles") if cfg.get(key, False)]
    if len(active) > 1:
        raise ValueError("Only one dataset representation flag may be true.")
    mode = active[0] if active else "smiles"
    identity = {"mode": mode}
    if mode == "use_safe":
        identity["slicer"] = cfg.get("safe_slicer", "brics")
    elif mode == "use_tsmiles":
        identity.update(slicer=cfg.get("tsmiles_slicer", "brics"), variant=cfg.get("tsmiles_variant", "TSDY"))
    return identity


def validate_indices(groups, total, expected_sizes=None, excluded=()):
    seen = set(excluded)
    for name, values in groups.items():
        if expected_sizes is not None and len(values) != expected_sizes[name]:
            raise ValueError(f"Incorrect number of indices in {name}")
        for index in values:
            if type(index) is not int or not 0 <= index < total or index in seen:
                raise ValueError(f"Invalid, duplicate, or overlapping index in {name}: {index}")
            seen.add(index)


def tokenizer_exclusion(cfg):
    """Require a verified tokenizer selection from the same PubChem source."""
    path_value = cfg.get("tokenizer_indices_path", None)
    if not path_value:
        raise ValueError("Set dataset.tokenizer_indices_path to the tokenizer corpus subset_indices.json.")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Tokenizer selection {path} is missing. Prepare the tokenizer corpus first.")
    identity = read_metadata(path)
    if identity.get("kind") != "tokenizer_indices" or identity.get("source") != source_identity(cfg):
        raise cache_error(path, "tokenizer and MLM source identities do not match")
    info = json.loads(path.read_text())
    validate_indices({"tokenizer": info["indices"]}, info["total_source"], {"tokenizer": identity["corpus_size"]})
    return set(info["indices"]), file_digest(path), info["total_source"], info["source_fingerprint"]
