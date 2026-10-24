"""Audit partitions and reconstruction before starting a shared-fragment study.

Run from the repository root with python -m scripts.audit_shared_fragmentation.
Input is a canonical SMILES text corpus, one molecule per line. This does not
train models or silently replace failed encodings with another representation.
"""

import argparse
from collections import Counter
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path

import chemicalgof
from rdkit import Chem, rdBase

from utils.cache_utils import file_digest
from utils.shared_fragmentation import (
    partition_molecule,
    serialize_fragsmiles,
    serialize_tsmiles,
    decode_tsid,
)
from utils.shared_ps import load_model


def audit(smiles, merge_model=None):
    counts = Counter()
    examples = []
    for smi in smiles:
        counts["molecules"] += 1
        try:
            base = partition_molecule(smi)
            partition = merge_model.apply(base) if merge_model is not None else base
            partition.validate()
            counts["partition_ok"] += 1
            counts["base_fragments"] += len(base.groups)
            counts["final_fragments"] += len(partition.groups)
            expected = Chem.MolToSmiles(partition.mol, isomericSmiles=True)
        except Exception as exc:
            counts["partition_error"] += 1
            if len(examples) < 20:
                examples.append({"smiles": smi, "stage": "partition", "error": str(exc)})
            continue
        for representation in ("fragsmiles", "TSID", "TSDY", "TSIS"):
            try:
                text = (
                    serialize_fragsmiles(partition)
                    if representation == "fragsmiles"
                    else serialize_tsmiles(partition, representation)
                )
                counts[representation + "_encoded"] += 1
                if representation == "TSDY":
                    continue  # No claim of unique reconstruction without attachment IDs.
                decoded = (
                    ".".join(chemicalgof.decode(c) for c in text.split(";"))
                    if representation == "fragsmiles"
                    else decode_tsid(text)
                )
                match = Chem.CanonSmiles(decoded) == expected
                counts[representation + ("_roundtrip_ok" if match else "_roundtrip_mismatch")] += 1
                if not match and len(examples) < 20:
                    examples.append({"smiles": smi, "stage": representation, "encoded": text, "decoded": decoded})
            except Exception as exc:
                counts[representation + "_error"] += 1
                if len(examples) < 20:
                    examples.append({"smiles": smi, "stage": representation, "error": str(exc)})
    return {"counts": dict(counts), "first_failures": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--ps-merges-path", type=Path)
    parser.add_argument("--require-roundtrip", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new audit output path")
    model = load_model(args.ps_merges_path) if args.ps_merges_path else None
    with args.input.open() as stream:
        result = audit(islice((line.strip() for line in stream if line.strip()), args.limit), model)
    result["provenance"] = {
        "input_sha256": file_digest(args.input),
        "ps_merges_sha256": file_digest(args.ps_merges_path) if args.ps_merges_path else None,
        "rdkit": rdBase.rdkitVersion,
        "chemicalgof": version("chemicalgof"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["counts"], indent=2))
    if args.require_roundtrip and any(v for k, v in result["counts"].items() if k.endswith(("_error", "_mismatch"))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
