"""Frozen-vocabulary intrinsic evaluation. No training; failures remain in denominators."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib
import json
import logging
from pathlib import Path
import random
import statistics
import time
from rdkit import Chem

ROOT = Path("outputs/intrinsic_study")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def conditions():
    items = []
    base = Path("data/pretrained_tokenizers")
    for domain in ("SMILY", "SAFE"):
        for alg in ("BPE", "SBPE", "SPCATT", "SMIRK", "SMIRK_GPE", "SMIRK_SGPE", "SMIRK_SPCATT", "APE", "SPE"):
            path = (
                "data/ape_tokenizer/mikemayuare/vocab.json"
                if alg == "APE"
                else str(base / ("SMILY_SMIRK" if alg == "SMIRK" else f"CANONICAL_{domain}_{alg}_5K"))
            )
            items.append(
                dict(
                    name=f"CA_{domain}_{alg}",
                    path=path,
                    representation="safe" if domain == "SAFE" else "smiles",
                    slicer="brics",
                    kind="ape" if alg == "APE" else None,
                )
            )
    for variant in ("TSSA", "TSDY", "TSID", "TSIS"):
        for slicer, suffix in [("brics", ""), ("mmpa", "_M")]:
            items.append(
                dict(
                    name=f"CA_{variant}_{slicer}",
                    path=str(base / f"CANONICAL_TSMILES_{variant}{suffix}_5K"),
                    representation="tsmiles",
                    variant=variant,
                    slicer=slicer,
                )
            )
    for name in ("FRAGSMILES_5K", "PS_FRAGSMILES_5K_20K_MF1", "PS_FRAGSMILES_5K_20K_MF2"):
        items.append(
            dict(
                name="CA_" + name,
                path=str(base / ("CANONICAL_" + name)),
                representation="old_ps" if name.startswith("PS") else "fragsmiles",
            )
        )
    for mode in ("base", "ps"):
        for variant in ("fragsmiles", "TSDY", "TSID", "TSIS"):
            items.append(
                dict(
                    name=f"shared_{mode}_{variant}",
                    path=f"outputs/shared_study_2m_v2/{mode}_{variant}/tokenizer",
                    representation="shared",
                    variant=variant,
                    ps_path="outputs/shared_study_2m_v2/miner/ps_merges.json" if mode == "ps" else None,
                )
            )
    items.append(dict(name="enhanced_TSDY", path="outputs/enhanced_tsdy_v3_1/tokenizer", representation="enhanced"))
    for item in items:
        p = Path(item["path"])
        files = (
            [p]
            if p.is_file()
            else [
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix in (".json", ".txt") and not f.name.startswith("corpus")
            ]
        )
        if item.get("ps_path"):
            files.append(Path(item["ps_path"]))
        item["hashes"] = {str(f): digest(f) for f in files}
    return items


def inputs(pubchem_limit, downstream_limit):
    """Fixed sample from preserved old PubChem test strings; overlap NOT assumed absent."""
    source = Path("data/pubchem10m_mlm_train/canon_smiles_test.txt")
    lines = source.read_text().splitlines()
    rng = random.Random(42)
    selected = sorted(rng.sample(range(len(lines)), min(pubchem_limit, len(lines))))
    rows = [dict(group="pubchem", id=i, smiles=lines[i]) for i in selected]
    files = {str(source): digest(source)}
    for task, folder in [("hiv", "base_fragsmiles_hiv_repair2"), ("bbbp", "base_fragsmiles_bbbp")]:
        root = Path("data/shared_study_2m_v2") / folder
        p = root / f"{task}.csv"
        idx = root / f"{task}_split_indices.json"
        data = list(csv.DictReader(p.open()))
        splits = json.loads(idx.read_text())
        files.update({str(f): digest(f) for f in (p, idx)})
        for split in ("train", "validation", "test"):
            indices = splits[split]
            if downstream_limit:
                indices = indices[:downstream_limit]
            rows.extend(dict(group=f"{task}_{split}", id=i, smiles=data[i]["smiles"]) for i in indices)
    return rows, dict(
        files=files,
        seed=42,
        pubchem_selection=selected,
        heldout_status="historical overlap unverified; no universal unseen-corpus claim",
    )


def effective_vocab(tok):
    if type(tok).__name__ == "SmirkTokenizerFast":
        return {tok.convert_ids_to_tokens(i): i for i in range(len(tok))}
    return tok.get_vocab()


def load(item):
    p = Path(item["path"])
    if item.get("kind") == "ape":
        from apetokenizer.ape_tokenizer import APETokenizer

        t = APETokenizer()
        t.load_vocabulary(str(p))
        return t
    c = json.loads((p / "tokenizer_config.json").read_text())["tokenizer_class"]
    if (p / "smirk_pcatt_meta.json").exists():
        c = "SmirkPCATTTokenizer"
    mapping = {
        "PreTrainedTokenizerFast": "transformers",
        "SmirkTokenizerFast": "smirk",
        "GreedTok": "pcatt.hf.greedtok",
        "SMILES_SPE_Tokenizer": "utils.spe_tokenizer",
        "TSMILESTokenizer": "utils.tsmiles_tokenizer",
        "FragSMILESTokenizer": "utils.fragsmiles_tokenizer",
        "PSFragSMILESTokenizer": "utils.ps_fragsmiles_tokenizer",
        "SmirkPCATTTokenizer": "utils.smirk_pcatt_tokenizer",
        "ExternalTSDYTokenizer": "utils.external_tsdy_tokenizer",
    }
    cls = getattr(importlib.import_module(mapping[c]), c)
    if c == "SmirkTokenizerFast":
        return cls(tokenizer_file=str(p / "tokenizer.json"))
    return cls.from_pretrained(str(p), local_files_only=True)


class Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class Words(list):
    pass


def raw_decode(words, fn):
    try:
        words.raw_decoded = fn()
    except Exception as e:
        words.raw_decoded = None
        words.raw_error = str(e)


def decode_frag_words(words):
    import chemicalgof

    parts = [[]]
    for word in words:
        if word == ";":
            parts.append([])
        else:
            parts[-1].append(word)
    return ".".join(chemicalgof.decode(part) for part in parts)


def convert(smi, item, tok, ps):
    rep = item["representation"]
    decode = None
    cores = None
    if rep == "enhanced":
        from utils.external_tsdy import encode, model_tokens, decode_tokens

        doc = encode(smi)
        words = Words(model_tokens(doc))
        words.raw_decoded = decode_tokens(words)
        cores = [x for c in json.loads(doc)["components"] for x in c["cores"]]
        return words, lambda w: decode_tokens(w), cores
    if rep == "shared":
        from utils.shared_fragmentation import encode_shared, decode_tsid

        text = encode_shared(smi, item["variant"], ps)
        if item["variant"] in ("TSID", "TSIS"):
            decode = lambda w: decode_tsid("".join(w))
    elif rep == "safe":
        import safe

        text = safe.encode(smi, canonical=True, slicer=item["slicer"], ignore_stereo=True)
        decode = lambda w: safe.decode("".join(w))
    elif rep == "tsmiles":
        from utils import tsmiles_utils as u

        parts = []
        for c in u._standardize(smi).split("."):
            with u.encoding_timeout(u.ENCODE_TIMEOUT_SECONDS):
                tree = u._encode_single_component(c, u._ALG_MAP[(item["slicer"], item["variant"])])
                if tree.mol is None:
                    raise ValueError("Empty t-SMILES tree")
                value = u._extract_variant(tree, item["variant"])
                if not value:
                    raise ValueError("Empty t-SMILES encoding")
                parts.append(value.strip())
        text = ".".join(parts)
        # Upstream t-SMILES decoder compatibility has not been verified here.
    elif rep == "old_ps":
        text = tok._get_ps_tokenizer().encode(smi, canonize=True, random=False, capitalize_legacy=True)
    elif rep == "fragsmiles":
        import chemicalgof

        text = ";".join(chemicalgof.encode(c, canonical=True) for c in smi.split("."))
    else:
        text = smi
    if rep in ("fragsmiles", "old_ps") or (rep == "shared" and item["variant"] == "fragsmiles"):
        import chemicalgof
        from utils.fragsmiles_utils import split_fragsmiles_tokens

        if rep == "old_ps":
            words = [w for c in text.split(";") for w in chemicalgof.split(c) if w]
        else:
            words = [w for w in split_fragsmiles_tokens(text, getattr(tok, "preserve_components", False)) if w]
        decode = decode_frag_words
        words = Words(words)
        raw_decode(words, lambda: ".".join(chemicalgof.decode(c) for c in text.split(";")))
        cores = [w for w in words if not (w.startswith("<") and w.endswith(">")) and w != ";"]
        return words, decode, cores
    if item.get("kind") == "ape":
        return text, None, None
    # Preserve native token IDs; SMIRK's tokenize/convert route can return None for OOV.
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Unexpected batch size")
        ids = ids[0]
    inverse = {v: k for k, v in effective_vocab(tok).items()}
    words = Words([inverse[i] for i in ids])
    words.native_ids = ids
    if rep == "safe":
        import safe

        raw_decode(words, lambda: safe.decode(text))
    elif rep == "shared" and item["variant"] in ("TSID", "TSIS"):
        raw_decode(words, lambda: decode_tsid(text))
    elif rep == "smiles":
        words.raw_decoded = smi
    if rep == "smiles":
        decode = lambda w: tok.convert_tokens_to_string(w)
    if hasattr(tok, "_greedtok") or type(tok).__name__ == "GreedTok":
        decode = None
    if rep in ("tsmiles", "shared"):
        cores = [w for w in words if w not in ("&", "^", ".")]
    return words, decode, cores


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q
    i = int(pos)
    f = pos - i
    return v[i] * (1 - f) + v[min(i + 1, len(v) - 1)] * f


def summarize(records, vocab_size):
    ok = [r for r in records if r["ok"]]
    lengths = [r["tokens"] for r in ok]
    total = sum(lengths)
    heavy = sum(r["heavy_atoms"] for r in ok)
    ids = {i for r in ok for i in r["used_ids"]}
    checked = [r for r in ok if r["representation_roundtrip"] is not None]
    post = [r for r in ok if r["vocabulary_roundtrip"] is not None]
    return dict(
        attempted=len(records),
        successful=len(ok),
        failed=len(records) - len(ok),
        mean_length=statistics.mean(lengths) if lengths else None,
        median_length=statistics.median(lengths) if lengths else None,
        p95_length=percentile(lengths, 0.95),
        p99_length=percentile(lengths, 0.99),
        tokens_per_heavy_atom=total / heavy if heavy else None,
        unknown_occurrence_rate=sum(r["unknown"] for r in ok) / total if total else None,
        molecules_with_unknown=sum(r["unknown"] > 0 for r in ok),
        molecule_unknown_rate=sum(r["unknown"] > 0 for r in ok) / len(ok) if ok else None,
        all_unknown_molecules=sum(r["all_unknown"] for r in ok),
        over_510=sum(r["tokens"] > 510 for r in ok),
        fragment_units=sum(r.get("core_count") or 0 for r in ok),
        unknown_fragment_units=sum(r.get("unknown_cores") or 0 for r in ok),
        vocabulary_size=vocab_size,
        used_token_ids=len(ids),
        vocabulary_utilization=len(ids) / vocab_size if vocab_size else None,
        representation_roundtrip_checked=len(checked),
        representation_roundtrip_pass=sum(r["representation_roundtrip"] for r in checked),
        vocabulary_roundtrip_checked=len(post),
        vocabulary_roundtrip_pass=sum(r["vocabulary_roundtrip"] for r in post),
    )


def evaluate(job):
    item, rows, folder, identity = job
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ready = folder / "complete.json"
    if ready.exists():
        saved = json.loads(ready.read_text())
        if saved["identity"] != identity or saved["records_sha256"] != digest(folder / "records.jsonl"):
            raise ValueError("Existing result mismatch")
        return item["name"]
    from utils.external_tsdy import molecules_equal
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    tok = load(item)
    ape = item.get("kind") == "ape"
    vocab = tok.vocabulary if ape else effective_vocab(tok)
    unknown = vocab.get(tok.unk_token)
    ps = None
    if item.get("ps_path"):
        from utils.shared_ps import load_model

        ps = load_model(item["ps_path"])
    warning = Warnings()
    logging.getLogger().addHandler(warning)
    results = []
    started = time.perf_counter()
    with (folder / "records.jsonl").open("w") as stream:
        for row in rows:
            r = dict(group=row["group"], id=row["id"], ok=False)
            warning.messages = []
            try:
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is None:
                    raise ValueError("Invalid source SMILES")
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                heavy = mol.GetNumHeavyAtoms()
                words, decoder, cores = convert(smi, item, tok, ps)
                if ape:
                    ids = tok(words, add_special_tokens=False)["input_ids"]
                else:
                    ids = getattr(words, "native_ids", None)
                if not ape and ids is None:
                    ids = tok.convert_tokens_to_ids(words)
                if warning.messages:
                    raise ValueError("Adapter warning: " + "; ".join(warning.messages))
                if not ids:
                    raise ValueError("Empty token stream")
                n = ids.count(unknown) if unknown is not None else 0
                raw = (
                    (molecules_equal(smi, words.raw_decoded) if words.raw_decoded is not None else False)
                    if hasattr(words, "raw_decoded")
                    else None
                )
                post = None
                decode_error = getattr(words, "raw_error", None)
                if decoder:
                    # Raw representation and vocabulary decoding are audited separately.
                    if n:
                        post = False
                    else:
                        try:
                            post = molecules_equal(smi, decoder([tok.convert_ids_to_tokens(i) for i in ids]))
                        except Exception as e:
                            post = False
                            decode_error = str(e)
                r.update(
                    ok=True,
                    tokens=len(ids),
                    heavy_atoms=heavy,
                    tokens_per_heavy_atom=len(ids) / heavy if heavy else None,
                    unknown=n,
                    all_unknown=n == len(ids),
                    used_ids=sorted(set(ids)),
                    representation_roundtrip=raw,
                    vocabulary_roundtrip=post,
                    decoder_status="checked" if decoder else "not_audited_or_ambiguous",
                    decode_error=decode_error,
                    core_count=len(cores) if cores is not None else None,
                    unknown_cores=(
                        sum(tok.convert_tokens_to_ids(c) == unknown for c in cores) if cores is not None else None
                    ),
                )
            except Exception as e:
                r.update(error=f"{type(e).__name__}: {e}", smiles=row["smiles"])
            stream.write(json.dumps(r) + "\n")
            results.append(r)
    logging.getLogger().removeHandler(warning)
    groups = {
        s: summarize([r for r in results if r["group"] == s], len(vocab)) for s in sorted({r["group"] for r in results})
    }
    write(folder / "summary.json", dict(condition=item, groups=groups, seconds=time.perf_counter() - started))
    write(ready, dict(identity=identity, records_sha256=digest(folder / "records.jsonl")))
    return item["name"]


def aggregate(output, items):
    records = {
        i["name"]: [json.loads(s) for s in (output / i["name"] / "records.jsonl").read_text().splitlines()]
        for i in items
    }
    successful = [{(r["group"], r["id"]) for r in rs if r["ok"]} for rs in records.values()]
    common = set.intersection(*successful)
    result = {}
    for i in items:
        name = i["name"]
        native = json.loads((output / name / "summary.json").read_text())
        v = next(iter(native["groups"].values()))["vocabulary_size"]
        result[name] = dict(
            native=native["groups"],
            common={
                g: summarize([r for r in records[name] if r["group"] == g and (g, r["id"]) in common], v)
                for g in native["groups"]
            },
        )
    write(output / "comparison.json", dict(common_rows=len(common), conditions=result))
    # Scientific figures, no model inference or W&B writes.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result)
    vals = [
        (
            result[n]["native"]["pubchem"]["mean_length"]
            if result[n]["native"]["pubchem"]["mean_length"] is not None
            else float("nan")
        )
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.22)))
    ax.barh(names, vals)
    ax.set_xlabel("Mean content tokens per successfully encoded molecule")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "pubchem_lengths.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--pubchem-limit", type=int, default=10000)
    p.add_argument("--downstream-limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conditions", nargs="*")
    a = p.parse_args()
    items = conditions()
    if a.conditions:
        known = {i["name"] for i in items}
        assert set(a.conditions) <= known, "Unknown condition"
        items = [i for i in items if i["name"] in a.conditions]
    rows, source = inputs(a.pubchem_limit, a.downstream_limit)
    from importlib.metadata import version

    code = {str(f): digest(f) for folder in ("utils", "scripts") for f in Path(folder).rglob("*.py")}
    manifest = dict(
        conditions=items,
        source=source,
        code=code,
        versions={k: version(k) for k in ("rdkit", "transformers", "tokenizers", "chemicalgof")},
        downstream_limit=a.downstream_limit,
        pubchem_limit=a.pubchem_limit,
    )
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output manifest changed; use a new directory")
    write(path, manifest)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for name in pool.map(evaluate, [(i, rows, str(a.output / i["name"]), manifest) for i in items]):
            print("Completed", name, flush=True)
    aggregate(a.output, items)


if __name__ == "__main__":
    main()
