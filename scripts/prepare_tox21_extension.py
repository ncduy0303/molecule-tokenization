"""Freeze SR-p53 splits, audit common encoding domain, and export model inputs."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from urllib.request import urlopen
import numpy as np
from utils.cache_utils import file_digest, value_digest, validate_indices
from utils.shared_fragmentation import encode_shared
from utils.external_tsdy import encode, model_tokens, decode_tokens, molecules_equal
from scripts.prepare_external_tsdy import write_json
from scripts.prepare_external_tsdy_training import prepare_task, SPLITS

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
DATA = Path("data/tox21_sr_p53")
TOKENIZERS = {
    "enhanced": Path("outputs/enhanced_tsdy_v3_1/tokenizer"),
    "TSDY": Path("outputs/shared_study_2m_v2/base_TSDY/tokenizer"),
    "fragsmiles": Path("outputs/shared_study_2m_v2/base_fragsmiles/tokenizer"),
}


def select_rows(rows):
    eligible, missing = [], []
    for i, row in enumerate(rows):
        label = row["SR-p53"].strip()
        if label == "" or label.lower() == "nan":
            missing.append(i)
        elif label in ("0", "1", "0.0", "1.0"):
            eligible.append(i)
        else:
            raise ValueError(f"Unexpected target at source row {i}: {label!r}")
    # Explicit RandomState permutation, equivalent 80/10/10 random allocation.
    order = np.random.RandomState(42).permutation(eligible).tolist()
    a, b = int(0.8 * len(order)), int(0.9 * len(order))
    return dict(train=order[:a], validation=order[a:b], test=order[b:]), missing


def audit_row(item):
    i, smiles = item
    texts, errors = {}, {}
    for kind in TOKENIZERS:
        try:
            if kind == "enhanced":
                if not molecules_equal(smiles, decode_tokens(model_tokens(encode(smiles)))):
                    raise ValueError("Enhanced raw token-stream reconstruction failed")
                texts[kind] = smiles
            else:
                texts[kind] = encode_shared(smiles, kind)
        except Exception as exc:
            errors[kind] = f"{type(exc).__name__}: {exc}"
    return dict(source_row=i, texts=texts, errors=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DATA)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".preparation-lock"
    lock.mkdir()
    try:
        source = root / "tox21.csv.gz"
        if not source.exists():
            with urlopen(URL, timeout=120) as response:
                payload = response.read()
            gzip.decompress(payload)  # Validate before publishing the source.
            tmp = source.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.replace(source)
        with gzip.open(source, "rt") as f:
            rows = list(csv.DictReader(f))
        identity = dict(
            source_url=URL,
            source_sha256=file_digest(source),
            target="SR-p53",
            split_seed=42,
            split="RandomState permutation 80/10/10 before chemistry exclusions",
            code={
                str(p): file_digest(p)
                for p in [Path(__file__), Path("utils/shared_fragmentation.py"), Path("utils/external_tsdy.py")]
            },
            versions={k: version(k) for k in ("numpy", "rdkit", "chemicalgof")},
        )
        manifest = root / "source_manifest.json"
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text()) != identity:
                raise ValueError("Existing study differs or --resume not supplied; preserve it and choose a new output")
        else:
            write_json(manifest, identity)
        original, missing = select_rows(rows)
        validate_indices(original, len(rows))
        items = [(i, rows[i]["smiles"]) for s in SPLITS for i in original[s]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            audited = list(pool.map(audit_row, items, chunksize=32))
        by_id = {r["source_row"]: r for r in audited}
        groups = {s: [i for i in original[s] if not by_id[i]["errors"]] for s in SPLITS}
        failures = [dict(r, smiles=rows[r["source_row"]]["smiles"]) for r in audited if r["errors"]]
        counts = {s: dict(Counter(str(int(float(rows[i]["SR-p53"]))) for i in groups[s])) for s in SPLITS}
        audit = dict(
            identity=identity,
            original_splits=original,
            common_splits=groups,
            missing_label_rows=missing,
            excluded_chemistry=failures,
            class_counts=counts,
            policy="Encoding support in all three; enhanced raw reconstruction required. TSDY is lossy. No SMILES fallback.",
            source_row_definition="Zero-based original CSV data row, before any filtering",
        )
        write_json(root / "domain_audit.json", audit)
        if any(set(c) != {"0", "1"} for c in counts.values()):
            raise ValueError("Each split must contain both labels")
        source_identity = dict(
            identity, domain_sha256=file_digest(root / "domain_audit.json"), split_rows_sha256=value_digest(groups)
        )
        for kind, tokenizer in TOKENIZERS.items():

            def provider(split, start, stop):
                return [
                    dict(source_row=i, text=by_id[i]["texts"][kind], labels=[float(rows[i]["SR-p53"])])
                    for i in groups[split][start:stop]
                ]

            prepare_task(
                root / kind,
                groups,
                provider,
                source_identity,
                tokenizer,
                kind,
                "tox21_sr_p53",
                workers=args.workers,
                length_policy="truncate",
                resume=args.resume,
            )
        print(
            json.dumps(
                dict(class_counts=counts, missing_labels=len(missing), excluded_chemistry=len(failures)), indent=2
            )
        )
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
