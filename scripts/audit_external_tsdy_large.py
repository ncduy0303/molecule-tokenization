"""Fixed-row, multiprocessing reconstruction and coverage audit; no training."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
from statistics import mean
from rdkit import Chem, RDLogger, rdBase
from utils.external_tsdy import (
    encode,
    model_tokens,
    decode_tokens,
    verbose_model_tokens,
    COMPACT_GRAMMAR,
    molecules_equal,
    TOKEN_FORMAT,
)
from utils.shared_fragmentation import partition_molecule, serialize_tsmiles
from utils.tsmiles_tokenizer import split_tsmiles_tokens
from utils.cache_utils import file_digest


def worker(item):
    RDLogger.DisableLog("rdApp.*")
    group, row, smiles = item
    result = dict(group=group, row=row, smiles=smiles)
    try:
        partition = partition_molecule(smiles)
        result["canonical"] = Chem.MolToSmiles(partition.mol)
        text = encode(smiles)
        tokens = model_tokens(text)
        if not molecules_equal(decode_tokens(tokens), result["canonical"]):
            raise ValueError("Compact token roundtrip mismatch")
        cores = [s for c in json.loads(text)["components"] for s in c["cores"]]
        base = split_tsmiles_tokens(serialize_tsmiles(partition, "TSDY"), True)
        result.update(
            ok=True,
            cores=cores,
            base_fragments=[t for t in base if t not in ("&", "^", ".")],
            compact_length=len(tokens),
            verbose_length=len(verbose_model_tokens(text)),
            base_length=len(base),
        )
    except Exception as exc:
        result.update(ok=False, error=str(exc))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train-limit", type=int, default=10000)
    p.add_argument("--validation-limit", type=int, default=2000)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists")
    source = Path("data/shared_study_2m_v2/miner/corpus.txt")
    with source.open() as f:
        raw = list(islice(f, a.train_limit + a.validation_limit))
    if len(raw) != a.train_limit + a.validation_limit:
        raise ValueError("Too few rows")
    items = [("pilot_train" if i < a.train_limit else "pilot_validation", i, s.strip()) for i, s in enumerate(raw)]
    hashes = {str(source): file_digest(source)}
    for task in ("hiv", "bbbp"):
        suffix = "_repair2" if task == "hiv" else ""
        folder = Path(f"data/shared_study_2m_v2/base_fragsmiles_{task}{suffix}")
        csvpath = folder / f"{task}.csv"
        splits = json.loads((folder / f"{task}_split_indices.json").read_text())
        hashes[str(csvpath)] = file_digest(csvpath)
        hashes[str(folder / f"{task}_split_indices.json")] = file_digest(folder / f"{task}_split_indices.json")
        with csvpath.open() as f:
            rows = list(csv.DictReader(f))
        for split in ("train", "validation", "test"):
            items.extend((task + "_" + split, i, rows[i]["smiles"]) for i in splits[split])
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(worker, items, chunksize=32), 1):
            results.append(r)
            if i % 5000 == 0:
                print(f"Audited {i}/{len(items)}", flush=True)
    train = [r for r in results if r["group"] == "pilot_train" and r["ok"]]
    cc = Counter(t for r in train for t in r["cores"])
    bc = Counter(t for r in train for t in r["base_fragments"])
    cv = set(t for t, n in sorted(cc.items(), key=lambda x: (-x[1], x[0]))[: 5000 - 7 - len(COMPACT_GRAMMAR)])
    bv = set(t for t, n in sorted(bc.items(), key=lambda x: (-x[1], x[0]))[: 5000 - 7 - 3])
    groups = {}
    for group in sorted({r["group"] for r in results}):
        allrows = [r for r in results if r["group"] == group]
        good = [r for r in allrows if r["ok"]]
        m = dict(
            attempted=len(allrows),
            roundtrip_ok=len(good),
            failures=len(allrows) - len(good),
            failure_reasons=dict(Counter(r["error"] for r in allrows if not r["ok"])),
        )
        for label, key, vocab in [("enhanced", "cores", cv), ("base", "base_fragments", bv)]:
            total = sum(len(r[key]) for r in good)
            m[label + "_fragment_unk_pct"] = (
                100 * sum(t not in vocab for r in good for t in r[key]) / total if total else None
            )
        for style in ("compact", "verbose", "base"):
            lengths = [r[style + "_length"] for r in good]
            m[style] = (
                dict(mean_tokens=mean(lengths), max_tokens=max(lengths), over_510=sum(n > 510 for n in lengths))
                if lengths
                else {}
            )
        groups[group] = m
    identities = {
        g: {r["canonical"] for r in results if r["group"] == g and "canonical" in r}
        for g in ("pilot_train", "pilot_validation")
    }
    report = dict(
        token_format=TOKEN_FORMAT,
        rdkit=rdBase.rdkitVersion,
        source_hashes=hashes,
        groups=groups,
        vocab=dict(
            budget=5000,
            reserved_special=7,
            reserved_compact_grammar=len(COMPACT_GRAMMAR),
            unique_training_cores=len(cc),
            unique_training_base_fragments=len(bc),
            retained_cores=len(cv),
            retained_base_fragments=len(bv),
        ),
        pilot_molecular_overlap=len(identities["pilot_train"] & identities["pilot_validation"]),
        notes=[
            "Fixed prefix slices; failures do not shift selection.",
            "Downstream uses the frozen index files; rows absent from those files are not audited.",
            "Coverage uses matched 10K pilot vocabularies, not full 2M vocabularies.",
            "No downstream labels used for representation or vocabulary fitting. Metrics on common supported rows only.",
        ],
    )
    a.output.mkdir(parents=True)
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (a.output / "rows.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
