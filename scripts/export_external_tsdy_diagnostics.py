"""Export a small diagnostic bundle from complete preflight shards (stdlib only)."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, default=Path("outputs/enhanced_tsdy_v3/preflight_2m"))
    p.add_argument("--input", type=Path, default=Path("data/shared_study_2m_v2/miner/corpus.txt"))
    p.add_argument("--output", type=Path, default=Path("outputs/enhanced_tsdy_v3/preflight_diagnostics"))
    a = p.parse_args()
    if a.output.exists():
        p.error("Output already exists; choose a new output directory")
    summary = json.loads((a.preflight / "summary.json").read_text())
    manifest = json.loads((a.preflight / "manifest.json").read_text())
    expected = math.ceil(summary["rows"] / manifest["shard_size"])
    shards = sorted(a.preflight.glob("shard_*.jsonl"))
    if len(shards) != expected:
        p.error(
            f"Incomplete shard set: {len(shards)} present; expected {expected}. Run on the cluster with the full shards."
        )
    if digest(a.input) != manifest["input_sha256"]:
        p.error("Input corpus hash differs from preflight")
    errors = []
    long = []
    rows = 0
    for i, path in enumerate(shards):
        expected_name = f"shard_{i*manifest['shard_size']:09d}.jsonl"
        if path.name != expected_name:
            p.error("Noncontiguous shard filenames")
        raw = path.read_bytes()
        sha = Path(str(path) + ".sha256")
        if not sha.exists() or hashlib.sha256(raw).hexdigest() != sha.read_text().strip():
            p.error(f"Checksum mismatch: {path}")
        for line in raw.splitlines():
            # Parse every record to validate row continuity as well as integrity.
            r = json.loads(line)
            if r["row"] != rows:
                p.error("Shard row continuity mismatch")
            rows += 1
            if not r["ok"]:
                errors.append(r)
            elif r["overlength"]:
                r["token_count"] = len(r["tokens"])
                long.append(r)
        if (i + 1) % 100 == 0:
            print(f"Checked {i+1}/{expected} shards", flush=True)
    if rows != summary["rows"] or len(errors) != summary["unsupported"] or len(long) != summary["overlength"]:
        p.error("Shard totals disagree with summary")
    wanted = {r["row"]: r for r in errors + long}
    with a.input.open() as source:
        for i, line in enumerate(source):
            if i in wanted:
                wanted[i]["smiles"] = line.strip()
    a.output.mkdir(parents=True)
    for name, data in [
        ("summary.json", summary),
        ("manifest.json", manifest),
        ("unsupported.json", errors),
        ("overlength.json", long),
        (
            "integrity.json",
            dict(
                shards=expected,
                rows=rows,
                all_checksums_match=True,
                summary_counts_match=True,
                input_sha256=manifest["input_sha256"],
            ),
        ),
    ]:
        (a.output / name).write_text(json.dumps(data, indent=2) + "\n")
    print(f"Exported {len(errors)} unsupported and {len(long)} overlength records to {a.output}")


if __name__ == "__main__":
    main()
