"""Replay raw representation mismatches; keep vocabulary loss separate."""

import json
import argparse
from collections import Counter
from pathlib import Path
from rdkit import Chem, RDLogger
from scripts.evaluate_intrinsic import inputs, conditions, load, convert, write, digest
from utils.external_tsdy import molecules_equal

ROOT = Path("outputs/intrinsic_study/reconstruction_audit")


def classify(source, decoded):
    if decoded is None:
        return "decoder_error", {}
    a, b = Chem.MolFromSmiles(source), Chem.MolFromSmiles(decoded)
    if a is None or b is None:
        return "invalid_decoded_smiles", {}
    if molecules_equal(source, decoded):
        return "now_equivalent", {}
    Chem.RemoveStereochemistry(a)
    Chem.RemoveStereochemistry(b)
    if molecules_equal(Chem.MolToSmiles(a), Chem.MolToSmiles(b)):
        return "stereo_only", {}

    def features(m):
        return dict(
            elements=dict(Counter(x.GetSymbol() for x in m.GetAtoms())),
            atoms=m.GetNumAtoms(),
            bonds=m.GetNumBonds(),
            charge=sum(x.GetFormalCharge() for x in m.GetAtoms()),
            radicals=sum(x.GetNumRadicalElectrons() for x in m.GetAtoms()),
            hydrogens=sum(x.GetTotalNumHs() for x in m.GetAtoms()),
            components=len(Chem.GetMolFrags(m)),
        )

    return "nonstereo_molecular_difference", dict(source=features(a), decoded=features(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    rows, provenance = inputs(10000, 0)
    lookup = {(r["group"], r["id"]): r["smiles"] for r in rows}
    ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for item in conditions():
        if args.names and item["name"] not in args.names:
            continue
        original = Path("outputs/intrinsic_study/full") / item["name"] / "records.jsonl"
        records = [json.loads(l) for l in original.open()]
        failed = [r for r in records if r.get("representation_roundtrip") is False]
        if not failed:
            continue
        target = ROOT / (item["name"] + ".jsonl")
        print(item["name"], len(failed), flush=True)
        tok = load(item)
        ps = None
        if item.get("ps_path"):
            from utils.shared_ps import load_model

            ps = load_model(item["ps_path"])
        counts = Counter()
        differences = Counter()
        with target.open("w") as stream:
            for i, r in enumerate(failed):
                source = Chem.MolToSmiles(Chem.MolFromSmiles(lookup[r["group"], r["id"]]), isomericSmiles=True)
                decoded = None
                error = None
                try:
                    words, _, _ = convert(source, item, tok, ps)
                    decoded = getattr(words, "raw_decoded", None)
                    error = getattr(words, "raw_error", None)
                    category, details = classify(source, decoded)
                except Exception as e:
                    category, details = "replay_exception", {}
                    error = str(e)
                counts[category] += 1
                if details:
                    differences.update(k for k in details["source"] if details["source"][k] != details["decoded"][k])
                stream.write(
                    json.dumps(
                        dict(
                            group=r["group"],
                            id=r["id"],
                            source=source,
                            decoded=decoded,
                            category=category,
                            error=error,
                            details=details,
                        )
                    )
                    + "\n"
                )
                if i % 1000 == 0:
                    print(" ", i, dict(counts), flush=True)
        summaries[item["name"]] = dict(
            original_raw_failures=len(failed),
            categories=dict(counts),
            feature_differences=dict(differences),
            records_sha256=digest(original),
            audit_sha256=digest(target),
        )
        write(ROOT / ("summary_subset.json" if args.names else "summary.json"), summaries)
    write(
        ROOT / ("manifest_subset.json" if args.names else "manifest.json"),
        dict(
            inputs=provenance,
            script_sha256=digest(__file__),
            scope="Raw representation failures only. Post-vocabulary losses are not replayed. Feature differences overlap; nonstereo differences do not alone prove a connectivity change.",
            rdkit=__import__("rdkit").__version__,
        ),
    )


if __name__ == "__main__":
    main()
