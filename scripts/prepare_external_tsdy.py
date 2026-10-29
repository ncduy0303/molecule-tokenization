"""Resumable full-corpus preflight and frequency counting, not model training.

Each completed shard has an integrity checksum. No unsupported or overlength
row is silently dropped from the audit. No trained tokenizer is exported.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import islice
import json
import os
from pathlib import Path

from rdkit import RDLogger
from utils.cache_utils import file_digest, read_metadata
from utils.external_tsdy import encode, model_tokens, decode_tokens, molecules_equal, COMPACT_GRAMMAR, TOKEN_FORMAT
from utils.shared_fragmentation import partition_molecule
from rdkit import Chem


def prepare(item):
    row, smiles, limit = item
    RDLogger.DisableLog("rdApp.*")
    try:
        text = encode(smiles)
        tokens = model_tokens(text)
        canonical = Chem.MolToSmiles(partition_molecule(smiles).mol)
        if not molecules_equal(decode_tokens(tokens), canonical):
            raise ValueError("Token reconstruction mismatch")
        return dict(
            row=row,
            ok=True,
            overlength=len(tokens) > limit,
            tokens=tokens,
            cores=[s for c in json.loads(text)["components"] for s in c["cores"]],
        )
    except Exception as exc:
        return dict(row=row, ok=False, smiles=smiles, error=str(exc))


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/shared_study_2m_v2/miner/corpus.txt"))
    p.add_argument("--output", type=Path, default=Path("outputs/enhanced_tsdy_v3/preflight_2m"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--limit", type=int, default=2000000)
    p.add_argument("--max-content-tokens", type=int, default=510)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.workers, a.shard_size, a.limit, a.max_content_tokens) < 1:
        p.error("Arguments must be positive")
    source_meta = Path(str(a.input) + ".meta.json")
    if source_meta.exists():
        read_metadata(a.input)
    identity = dict(
        input_sha256=file_digest(a.input),
        token_format=TOKEN_FORMAT,
        limit=a.limit,
        shard_size=a.shard_size,
        max_content_tokens=a.max_content_tokens,
        rdkit=version("rdkit"),
        networkx=version("networkx"),
        code={
            f: file_digest(f)
            for f in ["utils/external_tsdy.py", "utils/shared_fragmentation.py", "scripts/prepare_external_tsdy.py"]
        },
    )
    if source_meta.exists():
        identity["source_metadata_sha256"] = file_digest(source_meta)
    manifest = a.output / "manifest.json"
    if a.output.exists():
        if not a.resume:
            p.error("Output exists; use --resume with identical settings")
        if not manifest.exists() or json.loads(manifest.read_text()) != identity:
            p.error("Resume provenance differs or manifest missing; choose a new output directory")
    else:
        if a.resume:
            p.error("No output to resume")
        a.output.mkdir(parents=True)
        write_json(manifest, identity)
    lock = a.output / ".lock"
    try:
        lock.mkdir()
    except FileExistsError:
        p.error("Another preparer may be active; inspect .lock before retrying")
    try:
        counts = Counter()
        stats = Counter()
        reasons = Counter()
        with a.input.open() as source, ProcessPoolExecutor(max_workers=a.workers) as pool:
            start = 0
            while start < a.limit:
                batch = list(islice(source, min(a.shard_size, a.limit - start)))
                if not batch:
                    break
                path = a.output / f"shard_{start:09d}.jsonl"
                checksum = Path(str(path) + ".sha256")
                if path.exists() and checksum.exists():
                    if file_digest(path) != checksum.read_text().strip():
                        raise ValueError(f"Shard checksum mismatch: {path}")
                    print(f"Reusing rows {start}:{start+len(batch)}", flush=True)
                else:
                    temp = Path(str(path) + ".tmp")
                    with temp.open("w") as out:
                        for result in pool.map(
                            prepare,
                            ((start + i, s.strip(), a.max_content_tokens) for i, s in enumerate(batch)),
                            chunksize=32,
                        ):
                            out.write(json.dumps(result, separators=(",", ":")) + "\n")
                    os.replace(temp, path)
                    checksum.write_text(file_digest(path) + "\n")
                n = 0
                with path.open() as records:
                    for line in records:
                        r = json.loads(line)
                        if r["row"] != start + n:
                            raise ValueError("Shard row ordering mismatch")
                        n += 1
                        stats["rows"] += 1
                        if r["ok"]:
                            stats["roundtrip_ok"] += 1
                            stats["overlength"] += r["overlength"]
                            stats["tokens"] += len(r["tokens"])
                            stats["max_tokens"] = max(stats["max_tokens"], len(r["tokens"]))
                            counts.update(r["cores"])
                        else:
                            stats["unsupported"] += 1
                            reasons[r["error"]] += 1
                if n != len(batch):
                    raise ValueError("Incomplete shard")
                start += len(batch)
                write_json(a.output / "progress.json", dict(stats, requested=a.limit))
                print(
                    f"Completed {start}/{a.limit}; unsupported={stats['unsupported']}, overlength={stats['overlength']}",
                    flush=True,
                )
        complete = stats["rows"] == a.limit
        passed = complete and not stats["unsupported"] and not stats["overlength"]
        write_json(a.output / "core_counts.json", dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))))
        summary = dict(
            stats,
            requested=a.limit,
            complete=complete,
            gate_passed=passed,
            unique_cores=len(counts),
            core_budget=5000 - 7 - len(COMPACT_GRAMMAR),
            failure_reasons=dict(reasons),
            token_format=TOKEN_FORMAT,
            note="Counts include all reconstructable rows, including overlength rows. No tokenizer exported; resolve scope before vocabulary construction.",
        )
        write_json(a.output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        if not passed:
            raise SystemExit(2)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
