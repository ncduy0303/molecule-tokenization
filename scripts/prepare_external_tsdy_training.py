"""CPU-only, resumable preparation of frozen enhanced-TSDY model inputs.

No resampling. Unsupported chemistry blocks export. Truncation is explicit and
reported. The only downstream exclusion is the source-bound HIV train complex,
and the same scope is used for the optional base-TSDY/fragSMILES controls.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
from importlib.metadata import version

from utils.cache_utils import file_digest, value_digest, read_metadata, validate_indices
from utils.external_tsdy import TOKEN_FORMAT, encode, model_tokens, decode_tokens, molecules_equal
from utils.external_tsdy_tokenizer import ExternalTSDYTokenizer, export_tokenizer, tokenizer_identity
from scripts.prepare_external_tsdy import write_json

SPLITS = ("train", "validation", "test")
_worker_tokenizer = None
_worker_kind = None


def load_tokenizer(path, kind):
    if kind == "enhanced":
        return ExternalTSDYTokenizer.from_pretrained(str(path), local_files_only=True)
    if kind == "TSDY":
        from utils.tsmiles_tokenizer import TSMILESTokenizer

        return TSMILESTokenizer.from_pretrained(str(path), local_files_only=True)
    from utils.fragsmiles_tokenizer import FragSMILESTokenizer

    return FragSMILESTokenizer.from_pretrained(str(path), local_files_only=True)


def init_worker(path, kind):
    global _worker_tokenizer, _worker_kind
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _worker_tokenizer, _worker_kind = load_tokenizer(path, kind), kind


def encode_record(item):
    record, max_length, policy = item
    result = dict(source_row=record["source_row"])
    try:
        tok = _worker_tokenizer
        if _worker_kind == "enhanced":
            doc = encode(record["text"])
            words = model_tokens(doc)
            if not molecules_equal(record["text"], decode_tokens(words)):
                raise ValueError("Actual token-stream reconstruction failed")
            cores = [c for component in json.loads(doc)["components"] for c in component["cores"]]
        else:
            words = tok.tokenize(record["text"])
            cores = [w for w in words if w not in ("&", "^", ".", ";")]
        ids = tok.convert_tokens_to_ids(words)
        capacity = max_length - tok.num_special_tokens_to_add(pair=False)
        overlength = len(ids) > capacity
        result.update(
            full_content_tokens=len(ids),
            overlength=overlength,
            unknown_tokens=ids.count(tok.unk_token_id),
            core_count=len(cores),
            unknown_cores=sum(tok.convert_tokens_to_ids(c) == tok.unk_token_id for c in cores),
        )
        if overlength and policy == "error":
            raise ValueError(f"{len(ids)} content tokens exceeds capacity {capacity}; policy=error")
        # Same right truncation used by the frozen benchmark, explicitly audited.
        ids = tok.build_inputs_with_special_tokens(ids[:capacity])
        result.update(
            ok=True,
            input_ids=ids,
            attention_mask=[1] * len(ids),
            truncated=overlength,
            retained_unknown_tokens=ids.count(tok.unk_token_id),
        )
        if "labels" in record:
            result["labels"] = record["labels"]
        else:
            # Match frozen TSDY/fragSMILES MLM masking, including maskable UNK.
            # DataCollatorForLanguageModeling consumes this field before model.forward.
            protected = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
            result["special_tokens_mask"] = [int(i in protected) for i in ids]
    except Exception as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}", text=record["text"])
    return result


def arrow_rows(paths, classification):
    for path in paths:
        with open(path) as source:
            for line in source:
                record = json.loads(line)
                if not record["ok"]:
                    raise ValueError("Cannot export failed row")
                row = {k: record[k] for k in ("input_ids", "attention_mask")}
                if classification:
                    row["labels"] = record["labels"]
                else:
                    row["special_tokens_mask"] = record["special_tokens_mask"]
                yield row


def prepare_task(
    root,
    groups,
    provider,
    source_identity,
    tokenizer_path,
    kind,
    task,
    workers=8,
    shard_size=2048,
    length_policy="error",
    smoke=False,
    resume=False,
):
    """Provider(split,start,stop) returns exactly the specified frozen rows."""
    from datasets import Dataset, DatasetDict, Features, Sequence, Value

    root = Path(root)
    tok = load_tokenizer(tokenizer_path, kind)
    identity = dict(
        task=task,
        kind=kind,
        tokenizer=tokenizer_identity(tok),
        source=source_identity,
        max_length=512,
        length_policy=length_policy,
        smoke=smoke,
        shard_size=shard_size,
        mlm_masking_policy="frozen_cls_sep_pad_protected_unk_maskable" if task == "mlm" else None,
        code={
            p: file_digest(p)
            for p in (
                "scripts/prepare_external_tsdy_training.py",
                "utils/external_tsdy_tokenizer.py",
                "utils/external_tsdy.py",
                "utils/shared_fragmentation.py",
            )
        },
        versions={p: version(p) for p in ("rdkit", "transformers", "datasets", "networkx")},
    )
    if root.exists():
        if not resume or json.loads((root / "manifest.json").read_text()) != identity:
            raise ValueError(f"{root}: existing preparation differs or --resume was not supplied")
    else:
        root.mkdir(parents=True)
        write_json(root / "manifest.json", identity)
    lock = root / ".lock"
    lock.mkdir()  # A killed writer requires explicit inspection before removing this lock.
    try:
        metrics = {}
        paths = {}
        with ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(str(tokenizer_path), kind)
        ) as pool:
            for split in SPLITS:
                stats = Counter(
                    rows=0,
                    failed=0,
                    overlength=0,
                    tokens=0,
                    unknown_tokens=0,
                    cores=0,
                    unknown_cores=0,
                    retained_unknown_tokens=0,
                )
                outliers = []
                examples = []
                paths[split] = []
                folder = root / "shards" / split
                folder.mkdir(parents=True, exist_ok=True)
                for start in range(0, len(groups[split]), shard_size):
                    stop = min(start + shard_size, len(groups[split]))
                    path = folder / f"{start:09d}.jsonl"
                    sha = Path(str(path) + ".sha256")
                    expected = groups[split][start:stop]
                    if path.exists() and sha.exists():
                        if file_digest(path) != sha.read_text().strip():
                            raise ValueError(f"Corrupt preparation shard: {path}")
                        results = [json.loads(line) for line in path.read_text().splitlines()]
                    else:
                        items = provider(split, start, stop)
                        if [r["source_row"] for r in items] != expected:
                            raise ValueError("Source provider changed frozen row order")
                        results = list(pool.map(encode_record, [(r, 512, length_policy) for r in items], chunksize=32))
                        if [r["source_row"] for r in results] != expected:
                            raise ValueError("Worker results are incomplete or out of order")
                        payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in results).encode()
                        tmp = Path(str(path) + ".tmp")
                        tmp.write_bytes(payload)
                        if tmp.read_bytes() != payload:
                            raise ValueError("Shard write verification failed")
                        os.replace(tmp, path)
                        sha.write_text(file_digest(path) + "\n")
                    if [r["source_row"] for r in results] != expected:
                        raise ValueError(f"Unexpected source row identities: {path}")
                    paths[split].append(str(path))
                    for r in results:
                        stats["rows"] += 1
                        if r.get("overlength"):
                            stats["overlength"] += 1
                            outliers.append(r["source_row"])
                        if not r["ok"]:
                            stats["failed"] += 1
                            if len(examples) < 20:
                                examples.append(r)
                            continue
                        stats["tokens"] += r["full_content_tokens"]
                        stats["unknown_tokens"] += r["unknown_tokens"]
                        stats["cores"] += r["core_count"]
                        stats["unknown_cores"] += r["unknown_cores"]
                        stats["retained_unknown_tokens"] += r["retained_unknown_tokens"]
                    print(
                        f'{task}/{kind}/{split}: {stop}/{len(groups[split])}; failures={stats["failed"]}; long={stats["overlength"]}',
                        flush=True,
                    )
                metrics[split] = dict(stats, overlength_source_rows=outliers, failure_examples=examples)
        passed = all(m["failed"] == 0 and m["rows"] == len(groups[s]) and m["rows"] > 0 for s, m in metrics.items())
        write_json(root / "audit.json", dict(passed=passed, identity=identity, splits=metrics))
        if not passed:
            raise ValueError(f'Preparation gate failed; inspect {root / "audit.json"}. No ready dataset exported.')
        # Never rebuild an already completed immutable Arrow export on resume.
        if (root / "ready.json").exists():
            from omegaconf import OmegaConf
            from datamodules.molecule_datasets.prepared import load_prepared

            load_prepared(
                OmegaConf.create(dict(prepared_path=str(root), prepared_task=task, max_length=512, allow_smoke=smoke)),
                tok,
            )
            return
        # A prior interrupted Arrow export may be safely replaced; no ready marker exists.
        import shutil

        build = root / "arrow_build"
        if build.exists():
            shutil.rmtree(build)
        features = Features(dict(input_ids=Sequence(Value("int32")), attention_mask=Sequence(Value("int8"))))
        classification = task != "mlm"
        if classification:
            features["labels"] = Sequence(Value("float32"))
        else:
            features["special_tokens_mask"] = Sequence(Value("int8"))
        data = DatasetDict(
            {
                s: Dataset.from_generator(
                    arrow_rows,
                    gen_kwargs=dict(paths=paths[s], classification=classification),
                    features=features,
                    cache_dir=str(build),
                )
                for s in SPLITS
            }
        )
        target = root / "dataset"
        if target.exists():
            shutil.rmtree(target)
        data.save_to_disk(str(target))
        files = {str(p.relative_to(root)): file_digest(p) for p in target.rglob("*") if p.is_file()}
        files.update({name: file_digest(root / name) for name in ("manifest.json", "audit.json")})
        write_json(
            root / "ready.json",
            dict(
                passed=True,
                task=task,
                tokenizer=tokenizer_identity(tok),
                max_length=512,
                smoke=smoke,
                split_sizes={s: len(data[s]) for s in SPLITS},
                files=files,
            ),
        )
    finally:
        lock.rmdir()


def mlm_inputs(preflight, smoke_limit):
    """Use the exact existing selection and enforce tokenizer/MLM separation."""
    from datasets import load_dataset

    corpus = Path("data/shared_study_2m_v2/miner/corpus.txt")
    corpus_identity = read_metadata(corpus)
    manifest = json.loads((Path(preflight) / "manifest.json").read_text())
    if file_digest(corpus) != manifest["input_sha256"]:
        raise ValueError("Tokenizer fitting corpus changed")
    ti = Path("data/shared_study_2m_v2/miner/subset_indices.json")
    mi = Path("data/shared_study_2m_v2/base_TSDY_mlm/subset_indices.json")
    tokenizer_meta = read_metadata(ti)
    mlm_meta = read_metadata(mi)
    t = json.loads(ti.read_text())
    m = json.loads(mi.read_text())
    if corpus_identity["indices"] != file_digest(ti) or corpus_identity["selection"] != tokenizer_meta:
        raise ValueError("Vocabulary fitting and tokenizer selection differ")
    if mlm_meta["tokenizer_indices"] != file_digest(ti) or mlm_meta["source"] != tokenizer_meta["source"]:
        raise ValueError("MLM and tokenizer selection identities differ")
    groups = {s: m[s] for s in SPLITS}
    validate_indices({"tokenizer": t["indices"]}, t["total_source"], {"tokenizer": tokenizer_meta["corpus_size"]})
    validate_indices(
        groups,
        t["total_source"],
        dict(train=mlm_meta["train_size"], validation=mlm_meta["val_size"], test=mlm_meta["test_size"]),
        t["indices"],
    )
    if smoke_limit:
        groups = {s: ids[:smoke_limit] for s, ids in groups.items()}
    source = mlm_meta["source"]
    raw = load_dataset(source["dataset"], revision=source["revision"], split=source["split"])
    if len(raw) != t["total_source"] or raw._fingerprint != t["source_fingerprint"]:
        raise ValueError("Pinned source dataset contents/fingerprint changed")
    selected = {s: raw.select(ids) for s, ids in groups.items()}

    def provider(split, start, stop):
        texts = selected[split][start:stop][source["smiles_column"]]
        return [dict(source_row=i, text=text) for i, text in zip(groups[split][start:stop], texts)]

    return (
        groups,
        provider,
        dict(
            selection=mlm_meta,
            indices_sha256=file_digest(mi),
            tokenizer_indices_sha256=file_digest(ti),
            excluded_rows=[],
            split_rows_sha256=value_digest(groups),
        ),
    )


def downstream_inputs(task, kind, smoke_limit):
    contract = json.loads(Path("configurations/audits/external_tsdy_scope.json").read_text())
    for file, sha in contract["source_hashes"].items():
        if file_digest(file) != sha:
            raise ValueError(f"Frozen audit source changed: {file}")
    source = Path("data/shared_study_2m_v2") / (
        "base_fragsmiles_hiv_repair2" if task == "hiv" else "base_fragsmiles_bbbp"
    )
    csvpath = source / f"{task}.csv"
    index = source / f"{task}_split_indices.json"
    rows = list(csv.DictReader(csvpath.open()))
    indices = json.loads(index.read_text())
    groups = {s: indices[s][:] for s in SPLITS}
    validate_indices(groups, len(rows))
    target = "HIV_active" if task == "hiv" else "p_np"
    # The frozen CSVs have no missing labels; avoid changing row numbering.
    if any(r[target] not in ("0", "1", "0.0", "1.0") for r in rows):
        raise ValueError("Unexpected/missing frozen labels")
    exclusions = []
    for expected in contract["known_unsupported"]:
        if expected["group"] != f"{task}_train":
            continue
        import hashlib

        row = expected["row"]
        if hashlib.sha256(rows[row]["smiles"].encode()).hexdigest() != expected["smiles_sha256"]:
            raise ValueError("Scoped HIV molecule identity changed")
        groups["train"].remove(row)
        exclusions.append(expected)
    if smoke_limit:
        # Fixed diagnostic subset with both labels if available, never used for final metrics.
        for split, ids in groups.items():
            chosen = []
            for label in (0, 1):
                chosen.extend([i for i in ids if int(float(rows[i][target])) == label][: max(1, smoke_limit // 2)])
            groups[split] = sorted(chosen)
    texts = {}
    source_checks = {str(csvpath): file_digest(csvpath), str(index): file_digest(index)}
    if kind != "enhanced":
        suffix = "_repair2" if kind == "fragsmiles" and task == "hiv" else ""
        cached = Path(f"data/shared_study_2m_v2/base_{kind}_{task}{suffix}")
        control_tok = load_tokenizer(Path(f"outputs/shared_study_2m_v2/base_{kind}/tokenizer"), kind)
        if file_digest(cached / f"{task}.csv") != file_digest(csvpath):
            raise ValueError("Control CSV differs from the frozen shared source")
        if json.loads((cached / f"{task}_split_indices.json").read_text()) != indices:
            raise ValueError("Control splits differ from enhanced splits")
        for split in SPLITS:
            prefix = "fragsmiles" if kind == "fragsmiles" else "tsmiles_exocyclic_TSDY"
            file = cached / f"{task}_canon_{prefix}_{split}_seed42.txt"
            meta = read_metadata(file)
            if meta["representation"] != control_tok.init_kwargs.get("molecular_representation"):
                raise ValueError("Control tokenizer and cached representation differ")
            if meta["indices"] != file_digest(cached / f"{task}_split_indices.json"):
                raise ValueError("Control strings and selected indices differ")
            lines = file.read_text().splitlines()
            if len(lines) != len(indices[split]):
                raise ValueError("Control string count mismatch")
            texts[split] = dict(zip(indices[split], lines))
            source_checks[str(file)] = file_digest(file)

    def provider(split, start, stop):
        return [
            dict(
                source_row=i,
                text=rows[i]["smiles"] if kind == "enhanced" else texts[split][i],
                labels=[float(rows[i][target])],
            )
            for i in groups[split][start:stop]
        ]

    return (
        groups,
        provider,
        dict(
            files=source_checks,
            excluded_rows=exclusions,
            split_rows_sha256=value_digest(groups),
            scope_sha256=file_digest("configurations/audits/external_tsdy_scope.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, default=Path("outputs/enhanced_tsdy_v3_1/preflight_2m"))
    p.add_argument("--tokenizer-output", type=Path, default=Path("outputs/enhanced_tsdy_v3_1/tokenizer"))
    p.add_argument("--output", type=Path, default=Path("data/enhanced_tsdy_v3_1"))
    p.add_argument("--tasks", nargs="+", choices=["mlm", "hiv", "bbbp"], default=["mlm", "hiv", "bbbp"])
    p.add_argument(
        "--controls", action="store_true", help="Also prepare matched HIV controls using frozen base tokenizers"
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--length-policy", choices=["error", "truncate"], default="error")
    p.add_argument("--smoke-limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.workers < 1 or a.smoke_limit < 0:
        p.error("Invalid workers/smoke limit")
    export_tokenizer(a.preflight, a.tokenizer_output)
    if a.smoke_limit and a.output == Path("data/enhanced_tsdy_v3_1"):
        p.error("Choose a separate --output for smoke data")
    for task in a.tasks:
        kinds = ["enhanced"] + (["TSDY", "fragsmiles"] if a.controls and task == "hiv" else [])
        for kind in kinds:
            groups, provider, identity = (
                mlm_inputs(a.preflight, a.smoke_limit)
                if task == "mlm"
                else downstream_inputs(task, kind, a.smoke_limit)
            )
            tokenizer_path = (
                a.tokenizer_output if kind == "enhanced" else Path(f"outputs/shared_study_2m_v2/base_{kind}/tokenizer")
            )
            prepare_task(
                a.output / f"{kind}_{task}",
                groups,
                provider,
                identity,
                tokenizer_path,
                kind,
                task,
                workers=a.workers,
                length_policy=a.length_policy,
                smoke=bool(a.smoke_limit),
                resume=a.resume,
            )


if __name__ == "__main__":
    main()
