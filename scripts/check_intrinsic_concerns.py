"""Offline provenance, pairwise SAFE and reconstruction checks. Never retrains models."""

import json, csv, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from scripts.evaluate_intrinsic import load, convert, summarize, write, digest
from utils.external_tsdy import molecules_equal

ROOT = Path("outputs/intrinsic_study/full")
OUT = Path("outputs/intrinsic_study/focused_checks")
_CACHE = {}


def molecular_properties(m):
    return dict(
        atoms=m.GetNumAtoms(),
        bonds=m.GetNumBonds(),
        charge=sum(a.GetFormalCharge() for a in m.GetAtoms()),
        radicals=sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()),
        hydrogens=sum(a.GetTotalNumHs() for a in m.GetAtoms()),
        elements=dict(Counter(a.GetSymbol() for a in m.GetAtoms())),
    )


def classify(job):
    item, row, smiles = job
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    name = item["name"]
    r = dict(condition=name, group=row["group"], id=row["id"], source=smiles)
    try:
        if name not in _CACHE:
            ps = None
            if item.get("ps_path"):
                from utils.shared_ps import load_model

                ps = load_model(item["ps_path"])
            _CACHE[name] = (load(item), ps)
        tok, ps = _CACHE[name]
        m = Chem.MolFromSmiles(smiles)
        source = Chem.MolToSmiles(m, isomericSmiles=True)
        words, _, _ = convert(source, item, tok, ps)
        decoded = getattr(words, "raw_decoded", None)
        r["decoded"] = decoded
        if decoded is None:
            r.update(category="decode_error", error=getattr(words, "raw_error", "No decoded output"))
            return r
        if molecules_equal(source, decoded):
            r["category"] = "not_reproduced"
            return r
        d = Chem.MolFromSmiles(decoded)
        if d is None:
            r["category"] = "invalid_decoded_smiles"
            return r
        r["source_properties"] = molecular_properties(m)
        r["decoded_properties"] = molecular_properties(d)
        Chem.RemoveStereochemistry(m)
        Chem.RemoveStereochemistry(d)
        r["category"] = (
            "stereo_only" if molecules_equal(Chem.MolToSmiles(m), Chem.MolToSmiles(d)) else "non_stereo_mismatch"
        )
    except Exception as e:
        r.update(category="replay_error", error=f"{type(e).__name__}: {e}")
    return r


def artifact_check():
    rows = []
    keys = ["model", "pre_tokenizer", "normalizer", "post_processor", "decoder", "added_tokens"]
    for root in ["data/pretrained_tokenizers", "data/pretrained_models", "outputs"]:
        for p in Path(root).rglob("tokenizer.json"):
            if "SMIRK" not in str(p).upper() or not any(x in str(p).upper() for x in ["GPE", "SGPE"]):
                continue
            a = json.loads(p.read_text())
            signature = hashlib.sha256(json.dumps({k: a.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()
            rows.append(dict(path=str(p), file_sha256=digest(p), semantic_sha256=signature))
    runs = json.loads(Path("outputs/intrinsic_inventory/wandb_runs.json").read_text())
    runs = [
        dict(id=r["id"], name=r["name"], url=r["url"], state=r["state"], score=r["summary"].get("test/roc_auc"))
        for r in runs
        if "SMILY-SMIRK-" in (r["name"] or "") and any(x in r["name"] for x in ["-GPE-", "-SGPE-"])
    ]
    write(OUT / "artifact_inventory.json", dict(artifacts=rows, wandb_runs=runs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for p, h in manifest["source"]["files"].items():
        assert digest(p) == h, p
    comparison = json.loads((ROOT / "comparison.json").read_text())
    artifact_check()
    metrics = {}
    for alg in ["BPE", "SBPE", "SPCATT", "SMIRK", "SMIRK_GPE", "SMIRK_SGPE", "SMIRK_SPCATT", "APE", "SPE"]:
        names = ["CA_SMILY_" + alg, "CA_SAFE_" + alg]
        records = []
        for n in names:
            records.append(
                {
                    (r["group"], r["id"]): r
                    for r in map(json.loads, (ROOT / n / "records.jsonl").read_text().splitlines())
                }
            )
        common = {k for k, r in records[0].items() if r["ok"] and records[1][k]["ok"]}
        metrics[alg] = {}
        for group in comparison["conditions"][names[0]]["native"]:
            pair = {}
            for n, rs in zip(names, records):
                vocab = comparison["conditions"][n]["native"][group]["vocabulary_size"]
                pair[n] = summarize([r for k, r in rs.items() if k in common and k[0] == group], vocab)
            metrics[alg][group] = pair
    write(OUT / "safe_matched.json", metrics)
    raw = Path("data/pubchem10m_mlm_train/canon_smiles_test.txt").read_text().splitlines()
    sources = {"pubchem": raw}
    for task, folder in [("hiv", "base_fragsmiles_hiv_repair2"), ("bbbp", "base_fragsmiles_bbbp")]:
        data = list(csv.DictReader(open(f"data/shared_study_2m_v2/{folder}/{task}.csv")))
        for split in ["train", "validation", "test"]:
            sources[f"{task}_{split}"] = [r["smiles"] for r in data]
    jobs = []
    totals = {}
    for item in manifest["conditions"]:
        name = item["name"]
        # SAFE raw reconstruction is representation-level and shared across its tokenizer variants.
        if name.startswith("CA_SAFE_") and name != "CA_SAFE_BPE":
            continue
        bad = []
        for line in (ROOT / name / "records.jsonl").open():
            row = json.loads(line)
            if row.get("representation_roundtrip") is False:
                bad.append(row)
        if bad:
            totals[name] = len(bad)
        jobs.extend((item, r, sources[r["group"]][r["id"]]) for r in bad)
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    with ProcessPoolExecutor(max_workers=4) as pool, (OUT / "reconstruction_cases.jsonl").open("w") as stream:
        for i, result in enumerate(pool.map(classify, jobs, chunksize=64), 1):
            stream.write(json.dumps(result) + "\n")
            counts[result["condition"]][result["category"]] += 1
            key = result["condition"] + "/" + result["category"]
            if len(examples[key]) < 5:
                examples[key].append(result)
            if i % 1000 == 0:
                print(f"Classified {i}/{len(jobs)}", flush=True)
    write(
        OUT / "reconstruction_summary.json",
        dict(
            expected=totals,
            counts=dict(counts),
            examples=dict(examples),
            safe_replay="SAFE_BPE represents the shared SAFE conversion; other variants are not counted repeatedly",
        ),
    )
    write(
        OUT / "complete.json",
        dict(
            passed=True,
            cases=len(jobs),
            source_manifest_sha256=digest(ROOT / "manifest.json"),
            audit_code_sha256=digest(__file__),
        ),
    )


if __name__ == "__main__":
    main()
