"""Prepare validated Arrow splits for one tokenizer and dataset."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from importlib.metadata import version

from utils.cache_utils import file_digest
from utils.external_tsdy import encode, model_tokens, decode_tokens, molecules_equal
from utils.external_tsdy_tokenizer import ExternalTSDYTokenizer, tokenizer_identity
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
                "datamodules/molecule_datasets/prepare.py",
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
